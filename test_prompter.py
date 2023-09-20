import os
import argparse
import datetime
from colorama import init, Fore, Style
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import clip
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer

import wandb

from dataloader_prompter import VGPredicateDataset
from models.models import PromptLearner, TextEncoder, UVTransE

# Initialize colorama
init()

def freeze_model(model):
  for param in model.parameters():
      param.requires_grad = False

def test(args, is_pretrain=False):

    predicate_dataset =VGPredicateDataset( predicate_dict_dir='datasets/pred_dicts', 
                                    images_dir='./datasets/images',
                                    device=args.device, test=True, num_predicates=args.num_predicates)

    data_loader = DataLoader(predicate_dataset, batch_size=args.batch_size, shuffle=True)
        
    
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    freeze_model(clip_model)
    clip_model.eval()
    # Initialize the models
    prompt_learner = PromptLearner(clip_model, n_ctx=args.n_context_vectors, device=args.device, token_position=args.token_position)
    text_encoder = TextEncoder(clip_model).to(args.device)
    text_encoder.eval()

    if args.learnable_UVTransE:
        uv_transe = UVTransE()
        uv_transe = uv_transe.to(args.device)
        uv_transe.eval()

    if is_pretrain:
        args.epoch == -1

    
    if args.learnable_UVTransE:
        # Construct the model path
        model_path = os.path.join(args.checkpoint_dir, f'{args.epoch}_combined_models.pth')
    else:
        # Construct the model path
        model_path = os.path.join(args.checkpoint_dir, f'{args.epoch}_prompt_learner.pth')


    # Check if the model file exists
    if os.path.exists(model_path):
        if args.learnable_UVTransE:
            # Construct the model path
            loaded_state_dicts = torch.load(model_path, map_location=args.device)
            # Load the saved state dict into the model
            prompt_learner.load_state_dict(loaded_state_dicts['prompt_learner'])
            uv_transe.load_state_dict(loaded_state_dicts['UVTransE'], strict=False)
        else:
            # Load the saved state dict into the model
            prompt_learner.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        # If the model file does not exist, print a warning and load random weights
        print(Fore.RED + f"Warning: Model file '{model_path}' not found. Loading random weights instead.")
        # Reset colorama color settings
        print(Style.RESET_ALL)
    
    prompt_learner.to(args.device)
    
    prompt_learner.eval()
    text_encoder.eval()
    
    predicate_feat_dict={}
    for data_dict in tqdm(data_loader):
        anchor_union_emb = data_dict['anchor_union_emb'].to(args.device).float()
        anchor_sub_emb = data_dict['anchor_sub_emb'].to(args.device).float()
        anchor_phrase = data_dict['anchor_phrase']
        anchor_obj_emb = data_dict['anchor_obj_emb'].to(args.device).float()

        gt_predicate_labels = data_dict["anchor_gt_predicate_id"].cpu().detach().numpy()
        # Add context to the prompts
        anchor_prompts, anchor_tokenized_prompts = prompt_learner(anchor_phrase, anchor_union_emb)
        anchor_feats = text_encoder(anchor_prompts.half(), anchor_tokenized_prompts)

        # Do UVTransE
        if args.learnable_UVTransE:
            anchor_predicate_feats = uv_transe(anchor_feats.float(), anchor_sub_emb, anchor_obj_emb)
        else:
            anchor_predicate_feats   = anchor_feats - (anchor_sub_emb + anchor_obj_emb)


        for i in range(len(gt_predicate_labels)):
            gt_predicate_label = predicate_dataset.predicate_categories[gt_predicate_labels[i]]
            if gt_predicate_label in predicate_feat_dict:
                predicate_feat_dict[gt_predicate_label].append(anchor_predicate_feats.cpu().detach().numpy()[i])
            else:
                predicate_feat_dict[gt_predicate_label] = []
                predicate_feat_dict[gt_predicate_label].append(anchor_predicate_feats.cpu().detach().numpy()[i])

    temp_dict = {}
    for key, value in predicate_feat_dict.items():
        # Merge arrays along the first dimension
        temp_dict[key] = np.stack(value)
    
    predicate_feat_dict = temp_dict

    perform_tsne(args, predicate_feat_dict)

    # perform_umap(args, predicate_feat_dict)

    interpret_prompts(prompt_learner)
    interpret_conditional_prompts(prompt_learner)

def test_TTT(args, is_pretrain=False):

    predicate_dataset =VGPredicateDataset( predicate_dict_dir='datasets/pred_dicts', 
                                    images_dir='./datasets/images',
                                    device=args.device, test=True, num_predicates=args.num_predicates)

    data_loader = DataLoader(predicate_dataset, batch_size=args.batch_size, shuffle=True)
        
    
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    freeze_model(clip_model)
    clip_model.eval()
    # Initialize the models
    if args.learnable_UVTransE:
        uv_transe = UVTransE()
        uv_transe = uv_transe.to(args.device)
        uv_transe.eval()

    if is_pretrain:
        args.epoch == -1

    
    if args.learnable_UVTransE:
        # Construct the model path
        model_path = os.path.join(args.checkpoint_dir, f'{args.epoch}_combined_models.pth')
    else:
        # Construct the model path
        model_path = os.path.join(args.checkpoint_dir, f'{args.epoch}_prompt_learner.pth')


    # Check if the model file exists
    if os.path.exists(model_path):
        if args.learnable_UVTransE:
            # Construct the model path
            loaded_state_dicts = torch.load(model_path, map_location=args.device)
            uv_transe.load_state_dict(loaded_state_dicts['UVTransE'], strict=False)
    else:
        # If the model file does not exist, print a warning and load random weights
        print(Fore.RED + f"Warning: Model file '{model_path}' not found. Loading random weights instead.")
        # Reset colorama color settings
        print(Style.RESET_ALL)

    predicate_feat_dict={}
    for data_dict in tqdm(data_loader):
        anchor_union_emb = data_dict['anchor_union_emb'].to(args.device).float()
        anchor_sub_emb = data_dict['anchor_sub_emb'].to(args.device).float()
        anchor_phrase = data_dict['anchor_phrase']
        anchor_obj_emb = data_dict['anchor_obj_emb'].to(args.device).float()

        gt_predicate_labels = data_dict["anchor_gt_predicate_id"].cpu().detach().numpy()
        
        
        anchor_feats = predicate_dataset.CLIP_txt_encode(anchor_phrase)
        # Do UVTransE
        if args.learnable_UVTransE:
            anchor_predicate_feats = uv_transe(anchor_feats.float(), anchor_sub_emb, anchor_obj_emb)
        else:
            anchor_predicate_feats   = anchor_feats - (anchor_sub_emb + anchor_obj_emb)


        for i in range(len(gt_predicate_labels)):
            gt_predicate_label = predicate_dataset.predicate_categories[gt_predicate_labels[i]]
            if gt_predicate_label in predicate_feat_dict:
                predicate_feat_dict[gt_predicate_label].append(anchor_predicate_feats.cpu().detach().numpy()[i])
            else:
                predicate_feat_dict[gt_predicate_label] = []
                predicate_feat_dict[gt_predicate_label].append(anchor_predicate_feats.cpu().detach().numpy()[i])

    temp_dict = {}
    for key, value in predicate_feat_dict.items():
        # Merge arrays along the first dimension
        temp_dict[key] = np.stack(value)
    
    predicate_feat_dict = temp_dict

    perform_tsne(args, predicate_feat_dict)


def perform_tsne(args, predicate_feat_dict):
    
    # Combine the predicate features into a single array
    data = np.concatenate(list(predicate_feat_dict.values()))

    # Create labels for the predicate features
    labels = np.concatenate([np.array([k] * len(v)) for k, v in predicate_feat_dict.items()])

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        plt.scatter(
            reduced_data[labels == label, 0],
            reduced_data[labels == label, 1],
            label=f'{label} (n={np.sum(labels == label)})',
            alpha=0.5,  # Set transparency
        )

    # Add axis labels, legend, and title
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    if args.pretrain or args.epoch==-1:
        plt.title(f't-SNE plot for predicate features with randomly initialized conditional learnable prompting model\n# ctx vectors: {args.n_context_vectors}')
    else:        
        plt.title(f't-SNE plot for predicate features of epoch-{args.epoch} conditional learnable prompting model\n# ctx vectors: {args.n_context_vectors}')

    # Save the plot and show it
    plt.tight_layout()

    fig_pth = os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-predicate_feat_tsne.png')
    plt.savefig(fig_pth, dpi=300)

    # Calculate the silhouette score
    score = silhouette_score(reduced_data, labels)
    print("Silhouette score:", score)
    # Save the score to a text file
    with open(os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-silhouette_score.txt'), 'w') as f:
        f.write(str(score))

def perform_umap(args, predicate_feat_dict):
    
    # Combine the predicate features into a single array
    data = np.concatenate(list(predicate_feat_dict.values()))

    # Create labels for the predicate features
    labels = np.concatenate([np.array([k] * len(v)) for k, v in predicate_feat_dict.items()])

    # Perform t-SNE dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_data = reducer.fit_transform(data)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        plt.scatter(
            reduced_data[labels == label, 0],
            reduced_data[labels == label, 1],
            label=f'{label} (n={np.sum(labels == label)})',
            alpha=0.5,  # Set transparency
        )
    
    # Add axis labels, legend, and title
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    if args.pretrain or args.epoch==-1:
        plt.title(f'UMAP plot for predicate features with randomly initialized conditional learnable prompting model\n# ctx vectors: {args.n_context_vectors}')
    else:        
        plt.title(f'UMAP plot for predicate features of epoch-{args.epoch} conditional learnable prompting model\n# ctx vectors: {args.n_context_vectors}')

    # Save the plot and show it
    plt.tight_layout()

    fig_pth = os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-predicate_feat_UMAP.png')
    plt.savefig(fig_pth, dpi=300)

    # Calculate the silhouette score
    score = silhouette_score(reduced_data, labels)
    print("Silhouette score:", score)
    # Save the score to a text file
    with open(os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-silhouette_score.txt'), 'w') as f:
        f.write(str(score))

def interpret_prompts(prompt_learner, topk=5):

    tokenizer = SimpleTokenizer()
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)

    token_embedding = clip_model.token_embedding.weight
    print(f"Size of token embedding: {token_embedding.shape}")

    ctx = prompt_learner.ctx
    ctx = ctx.float()
    print(f"Size of context: {ctx.shape}")

    # Decode the token embeddings to words
    word_ids = torch.arange(token_embedding.size(0))
    words = [tokenizer.decode([idx.item()]) for idx in word_ids]

    if ctx.dim() == 2:
        # Generic context
        distance = torch.cdist(ctx, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        with open(os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-prompt_interpret.txt'), 'w') as f:
            for m, idxs in enumerate(sorted_idxs):
                words = [tokenizer.decoder[idx.item()] for idx in idxs]
                dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
                print(f"{m+1}: {words} {dist}")

                f.write(f"{m+1}: {words} {dist}\n")

    elif ctx.dim() == 3:
        # Class-specific context
        raise NotImplementedError

def interpret_conditional_prompts(prompt_learner):

    tokenizer = SimpleTokenizer()
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)

    token_embedding = clip_model.token_embedding.weight
    print(f"Size of token embedding: {token_embedding.shape}")

    # Decode the token embeddings to words
    word_ids = torch.arange(token_embedding.size(0))
    words = [tokenizer.decode([idx.item()]) for idx in word_ids]

    def interpret_ctx(ctx, topk=5):
        
        ctx = ctx.squeeze(0)
        
        if ctx.dim() == 2:
            # Generic context
            distance = torch.cdist(ctx, token_embedding)
            print(f"Size of distance matrix: {distance.shape}")
            sorted_idxs = torch.argsort(distance, dim=1)
            sorted_idxs = sorted_idxs[:, :topk]

            with open(os.path.join(os.path.dirname(args.checkpoint_dir), f'{args.epoch}-prompt_interpret.txt'), 'w') as f:
                for m, idxs in enumerate(sorted_idxs):
                    words = [tokenizer.decoder[idx.item()] for idx in idxs]
                    dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
                    print(f"{m+1}: {words} {dist}")

                    f.write(f"{m+1}: {words} {dist}\n")

        elif ctx.dim() == 3:
            # Class-specific context
            raise NotImplementedError

    predicate_dataset =VGPredicateDataset( predicate_dict_dir='datasets/pred_dicts', 
                                    images_dir='./datasets/images',
                                    device=args.device, test=True, num_predicates=args.num_predicates)

    data_loader = DataLoader(predicate_dataset, batch_size=1, shuffle=True)

    entity_ids = {}
    for data_dict in tqdm(data_loader):
        anchor_union_emb = data_dict['anchor_union_emb'].to(args.device).float()
        anchor_phrase = data_dict['anchor_phrase']

        gt_predicate_ids = data_dict["anchor_gt_predicate_id"].cpu().detach().numpy()[0]

        sub_label, obj_label = anchor_phrase[0].split(' ')
        predicate_label = predicate_dataset.predicate_categories[gt_predicate_ids]
        true_phrase = f'{sub_label} {predicate_label} {obj_label}'

        print(true_phrase)
        # Add context to the prompts
        
        prompt_embeddings, tokenized_prompts, ctx_shifted = prompt_learner(anchor_phrase, anchor_union_emb, True)
        
        interpret_ctx(ctx_shifted)
        
        assert False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set hyperparameters for the model")
    parser.add_argument('--n_context_vectors', type=int, default=8, help='Number of context vectors')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA if available')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number for loading model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory for input files')
    parser.add_argument('--pretrain', action='store_true', help='Use random model')
    parser.add_argument('--token_position', type=str, default='front', help='position of the learnable context token')
    parser.add_argument('--learnable_UVTransE',type=bool, default=False, help='use learnable layers for UVTransE')
    parser.add_argument('--num_predicates', type=int, default=5, help='Number of predicates of interest')

    parser.add_argument('--train_TTT', type=bool, default=False, help='Train to cluster TTT where phrase if (sub obj)')
 

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(f"Time and Date of the run: {current_time}\nArguments: {args}")

    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

    if args.train_TTT:
        args.learnable_UVTransE = True
        test_TTT(args)
    else:
        test(args)