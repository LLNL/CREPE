import os
import argparse
import datetime
from colorama import init, Fore, Style
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from dataloader_classifier import VGDataset
from models.models import PredicateEstimator
from test_classifier import evaluate

# Initialize colorama
init()

def freeze_model(model):
  for param in model.parameters():
      param.requires_grad = False

def train(args):

    ####################### Initialize data #######################
    train_dataset = VGDataset(split='train', case=1, data_root=args.data_dir, num_predicates=args.num_predicates)
    val_dataset = VGDataset(split='val', case=1, data_root=args.data_dir, num_predicates=args.num_predicates)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    ####################### Initialize the models #######################

    predicate_estimator = PredicateEstimator(args, args.device, isTest=False, resume=args.resume_training)
    print(PredicateEstimator)
    ####################### Initialize optimizer #######################

    predicate_estimator.set_train()
    # Define your optimizer and specify the learning rate
    optimizer = predicate_estimator.configure_optimizers(optimizer_name=args.optimizer, lr=args.learning_rate)

    ####################### Evaluate the model #######################

    if args.which_epoch_cls != 0:
        pretrain_epoch = args.which_epoch_cls
    else:
        pretrain_epoch = -1
    # Evaluate the model on the validation set before training
    val_loss = validate(predicate_estimator, val_data_loader, train_dataset.ind_to_predicates, 'val', pretrain_epoch)
    
    ####################### Train the model #######################
    predicate_losses = []
    for epoch in tqdm(range(args.which_epoch_cls+1, args.train_epochs), desc="Epochs"):
        epoch_predicate_loss = 0
        num_batches = 0  
        for data_dict in tqdm(train_data_loader):
            if data_dict['img_name'][0] == 'None':
                continue
            predicate_estimator.set_input(data_dict)
    
            predicate_logits = predicate_estimator.get_estimate()
            
            predicate_loss = predicate_estimator.get_loss(predicate_logits)

            optimizer.zero_grad()
            predicate_loss.backward()
            optimizer.step()
            # print(f"Batch {num_batches + 1}: Predicate Loss = {predicate_loss.item():.4f}")

            epoch_predicate_loss += predicate_loss.item()
            num_batches += 1

        avg_predicate_loss = epoch_predicate_loss / num_batches
        
        predicate_losses.append(avg_predicate_loss)

        print(f"Epoch {epoch + 1}/{args.train_epochs}: Predicate Loss = {avg_predicate_loss:.4f}")

        model_path = os.path.join(args.out_dir, 'checkpoints_classifier')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if epoch % args.save_freq == 0:
            if args.learnable_UVTransE:
                if args.is_TTT == False:
                    torch.save({
                        'epoch': epoch,
                        'prompt_learner': predicate_estimator.prompt_learner.state_dict(),
                        'UVTransE': predicate_estimator.uv_transe.state_dict(),
                        'classifier': predicate_estimator.classifier.state_dict(),

                    }, os.path.join(model_path, f'{epoch}_combined.pth'))
                else:
                    torch.save({
                        'epoch': epoch,
                        'UVTransE': predicate_estimator.uv_transe.state_dict(),
                        'classifier': predicate_estimator.classifier.state_dict(),

                    }, os.path.join(model_path, f'{epoch}_combined.pth'))
            else:
                torch.save({
                    'epoch': epoch,
                    'prompt_learner': predicate_estimator.prompt_learner.state_dict(),
                    'classifier': predicate_estimator.classifier.state_dict(),

                }, os.path.join(model_path, f'{epoch}_combined.pth'))

            # torch.save(predicate_estimator.classifier.state_dict(), os.path.join(model_path, f'{epoch}_combined.pth'))

        # Save training progress to a text file
        with open(os.path.join(args.out_dir, 'classifier_training_report.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}/{args.train_epochs}: Predicate Loss = {avg_predicate_loss:.4f}\n")

        fig, axs = plt.subplots(1, 1, figsize=(8, 12))
        fig.tight_layout(pad=5.0)

        axs.plot(predicate_losses, label='Predicate Loss')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Predicate Loss')
        axs.legend()
        plt.savefig(os.path.join(args.out_dir, 'classifier_epoch_losses.png'))

        # ####################### Evaluate the model #######################
        # # Evaluate the model on the validation set
        val_loss = validate(predicate_estimator, val_data_loader, train_dataset.ind_to_predicates, 'val', epoch)

def validate(predicate_estimator, val_data_loader, ind_to_predicates, mode, epoch):
    predicate_estimator.set_eval()

    val_loss, eval_res = evaluate(predicate_estimator, val_data_loader, ind_to_predicates, return_loss=True)

    # Appending to file
    with open(os.path.join(args.out_dir, 'classifier_val_report.txt'), 'a') as file1:
        file1.write(f"Epoch:\t{epoch}\n")
        file1.write(eval_res)
        file1.write(f"\nValidation_loss:\t{val_loss}\n")

    predicate_estimator.set_train()
    return val_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set hyperparameters for the model")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='No of training Epochs for the classifier')
    parser.add_argument('--which_epoch', type=int, default=0, help='Epoch number for loading the prompter and UVTransE model checkpoint')
    parser.add_argument('--resume_training', type=bool, default=False, help='Resume training from a checkpoint')
    parser.add_argument('--which_epoch_cls', type=int, default=0, help='Epoch number for loading the classifier model checkpoint')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA if available')

    # Model parameters
    parser.add_argument('--n_context_vectors', type=int, default=8, help='Number of context vectors')
    parser.add_argument('--token_position', type=str, default='front', help='Position of the learnable context token')
    parser.add_argument('--learnable_UVTransE',type=bool, default=False, help='Load pretrained learnable layers for UVTransE')
    parser.add_argument('--randomize_UVTransE',type=bool, default=False, help='randomly initialize UVTransE')
    parser.add_argument('--is_halu_prompting',type=bool, default=False, help='Use halucinator for prompting')
    parser.add_argument('--update_prompt_learner',type=bool, default=False, help='update the propmt learner')
    parser.add_argument('--update_UVTransE',type=bool, default=False, help='update the layers for UVTransE')
    parser.add_argument('--is_non_linear',type=bool, default=False, help='Non linearity in the classifier')

    parser.add_argument('--num_predicates', type=int, default=50, help='Number of predicates of interest')


    parser.add_argument('--is_TTT',type=bool, default=False, help='Use TTT model for getting the union features')
    # Input/output and logging parameters
    parser.add_argument('--checkpoints_dir_prompt', type=str, default='checkpoints', help='Directory to the prompter and UVTransE')
    parser.add_argument('--log_dir_cls', type=str, default='checkpoints', help='Directory to the classifier')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory for input files')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency for saving the model')

    # Parse the arguments
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Time and Date of the run: {current_time}\nArguments: {args}")

    if args.resume_training:
        args.out_dir = args.log_dir_cls
    else:
        args.out_dir = f"{args.out_dir}/classifier/{current_time}"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)     

    if args.is_TTT:
            args.learnable_UVTransE = True
    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    if args.resume_training:
        args_path = os.path.join(args.out_dir, f'res_{args.which_epoch_cls}_classifier_args.json')
    else:
        args_path = os.path.join(args.out_dir, 'classifier_args.json')
    # Save the arguments to a JSON file
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    train(args)