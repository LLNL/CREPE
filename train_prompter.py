import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from clip import clip

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import json
import datetime
from colorama import init, Fore, Style
# Initialize colorama
init()

from dataloader_prompter import VGPredicateDataset
from models.models import PromptLearner, TextEncoder

def freeze_model(model):
  for param in model.parameters():
      param.requires_grad = False

def train(args):

    train_dataset = VGPredicateDataset( predicate_dict_dir='datasets/pred_dicts_train_cmr', 
                                    images_dir='./datasets/images',
                                    device=args.device, debug=False, test=False, num_predicates=50)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
    clip_model, _ = clip.load("ViT-B/32", device=args.device)
    freeze_model(clip_model)
    clip_model.eval()
    # Initialize the models
    prompt_learner = PromptLearner(clip_model, n_ctx=args.n_contex_vectors, device=args.device, token_position=args.token_position)
    text_encoder = TextEncoder(clip_model).to(args.device)
    text_encoder.eval()

    prompt_learner = prompt_learner.to(args.device)
    if args.device == 'cuda':
        text_encoder = nn.DataParallel(text_encoder)

    if args.resume_training:
        model_path = os.path.join(args.checkpoints_dir_prompt, f'{args.which_epoch}_prompt_learner.pth')
        # Check if the model file exists
        if os.path.exists(model_path):
            prompt_learner.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(Fore.GREEN + f"Loaded model file '{model_path}'")
            # Reset colorama color settings
            print(Style.RESET_ALL)
        else:
            # If the model file does not exist, print a warning and load random weights
            print(Fore.RED + f"Warning: Model file '{model_path}' not found. Loading random weights instead.")
            # Reset colorama color settings
            print(Style.RESET_ALL)

    # the training
    prompt_learner.train()
    
    # Define your optimizer and specify the learning rate
    optimizer = optim.SGD(prompt_learner.parameters(), lr=args.learning_rate)

    epoch_losses = []
    for epoch in tqdm(range(args.which_epoch+1, args.epochs), desc="Epochs"):
        epoch_loss = 0
        num_batches = 0  
        for data_dict in tqdm(trainloader):

            union_img_emb = data_dict['union_img_emb'].float().squeeze(0).to(args.device)
            union_CMR_emb = data_dict['union_cmr_emb'].float().squeeze(0).to(args.device)
            phrases = data_dict['phrases']

            # Add context to the prompts
            union_prompts, union_tokenized_prompts = prompt_learner(phrases, union_img_emb)
            union_prompt_emb = text_encoder(union_prompts.half(), union_tokenized_prompts)

            # Compute cosine similarity
            cos_sim_positive = F.cosine_similarity(union_img_emb, union_prompt_emb, dim=-1)
            cos_sim_negative = F.cosine_similarity(union_img_emb, union_CMR_emb, dim=-1)
            
            # Compute exponentials
            exp_cos_sim_postive = torch.exp(cos_sim_positive)
            exp_cos_sim_negative = torch.exp(cos_sim_negative)
            
            # Compute InfoNCE loss
            loss = -torch.log(exp_cos_sim_postive / (exp_cos_sim_postive + exp_cos_sim_negative))
            loss = loss.mean()
            
            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Batch {num_batches + 1}: Total Loss = {loss.item():.4f},")

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{args.epochs}: Total Loss = {avg_loss:.4f}, ")

        model_path = os.path.join(args.out_dir, 'checkpoints')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if epoch % args.save_freq == 0:
            torch.save(prompt_learner.state_dict(), os.path.join(model_path, f'{epoch}_prompt_learner.pth'))

        # Save training progress to a text file
        with open(os.path.join(args.out_dir, 'training_progress.txt'), 'a') as f:
            f.write(f"\nEpoch {epoch + 1}/{args.epochs}: Total Loss = {avg_loss:.4f}\n")

        fig, axs = plt.subplots(1, 1, figsize=(8, 12))
        fig.tight_layout(pad=5.0)

        axs.plot(epoch_losses, label='Total Loss')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Total Loss')
        axs.legend()


        if args.resume_training:
            plt.savefig(os.path.join(args.out_dir, f'res_{args.which_epoch}_epoch_losses.png'))
        else:
            plt.savefig(os.path.join(args.out_dir, 'epoch_losses.png'))

    torch.save(prompt_learner.state_dict(), os.path.join(model_path, f'{args.epochs}_prompt_learner.pth'))

def main(args):
    train(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set hyperparameters for the model")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA if available')
    parser.add_argument('--use_DDP', type=bool, default=False, help='Use Distributed Data Parallel for training')
    parser.add_argument('--resume_training', type=bool, default=False, help='Resume training from a checkpoint')
    parser.add_argument('--checkpoints_dir_prompt', type=str, default='checkpoints', help='Directory for saving prompter model checkpoints')
    parser.add_argument('--which_epoch', type=int, default=0, help='Epoch number for loading the prompter model checkpoint')
    
    # Model parameters
    parser.add_argument('--C', type=int, default=1, choices=[1], help='Cross modal retrival no of crops')
    parser.add_argument('--K', type=int, default=1, choices=[1], help='Cross modal top K retrival ')
    parser.add_argument('--n_contex_vectors', type=int, default=4, help='Number of context vectors')
    parser.add_argument('--token_position', type=str, default='middle', help='position of the learnable context token')
    parser.add_argument('--num_predicates', type=int, default=5, help='Number of predicates of interest')

    # Input/output and logging parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory for input files')
    parser.add_argument('--out_dir', type=str, default='output', help='Output directory to save progress, plots, and model checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency for saving the model')
    
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.resume_training:
        args.out_dir = os.path.dirname(args.checkpoints_dir_prompt)
    else:
        args.out_dir = f"{args.out_dir}/{current_time}"

    print(f"Time and Date of the run: {current_time}\nArguments: {args}")
    
    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.resume_training:
        args_path = os.path.join(args.out_dir, f'res_{args.which_epoch}_args.json')
    else:
        args_path = os.path.join(args.out_dir, 'args.json')
    # Save the arguments to a JSON file
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Save the arguments to a JSON file
    with open(os.path.join(args.out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    main(args)