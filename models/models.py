import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import clip
from collections import OrderedDict
import os
from colorama import init, Fore, Style

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx=16, token_position='front', device='cpu'):
        super().__init__()
        
        self.clip_model = clip_model

        self.token_position = token_position

        self.device = device
        dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]

        print("Initializing a generic context")
        ctx_init = " ".join(["X"] * n_ctx)
        
        # use given words to initialize context vectors
        prompt = clip.tokenize(ctx_init).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        print(f'Initial context: "{ctx_init}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors).to(self.device)  # to be optimized

        # Initialize the meta-Net
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, ctx_dim))
        ])).to(self.device)

        self.dtype = dtype
        self.n_ctx = n_ctx

    def compute_prefix_sufix(self, phrases):
        
        prompt_dummy = " ".join(["X"] * self.n_ctx)

        phrases = [phrase.replace("_", " ") for phrase in phrases]

        if self.token_position == 'front':
            prompts = [prompt_dummy + " " + name + "." for name in phrases]
        elif self.token_position == "middle":
            prompts = []
            for name in phrases:
                prompt = name.split(' ')[0] + " " + prompt_dummy + " " + name.split(' ')[1] + "." 
                prompts.append(prompt)

        # Tokenize the prompt with the dummy preffix added
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # Embed the tokens
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        if self.token_position == "front":
            # Split the prefix and suffix from the embeddings
            # prefix is start of sentence[SOS]: suffix is the actual phrase with EOS
            token_prefix = embedding[:, :1, :]
            token_suffix = embedding[:, 1 + self.n_ctx :, :]

            return token_prefix, token_suffix

        elif self.token_position == "middle":
            token_prefix = embedding[:, :1, :]
            token_sub = embedding[:, 1:2, :]
            token_obj = embedding[:, 2+ self.n_ctx :3+ self.n_ctx, :] # The '.' is accounted into suffix
            token_suffix = embedding[:, 3+ self.n_ctx:, :]
            
            return token_prefix, token_sub, token_obj, token_suffix

    def forward(self, phrases, im_features, eval=False):
        ctx = self.ctx  # (n_ctx, ctx_dim)

        # Compute the image-based bias using the meta-net
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)

        # If the context tensor is 2D, expand it to match the batch size
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(bias.shape[0], -1, -1)  # (batch, n_ctx, ctx_dim)

        # Add the bias to the context tensor
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)

        if self.token_position == 'front':
            # Compute the prefix (SOS) and suffix (EOS) tokens for the phrases
            prefix, suffix = self.compute_prefix_sufix(phrases)

            # Concatenate the prefix, context, and suffix to form the new prompt
            prompts = torch.cat(
                [
                    prefix,  # (batch, 1, dim)
                    ctx_shifted,     # (batch, n_ctx, ctx_dim)
                    suffix,  # (batch, *, dim)
                ],
                dim=1,
            )
        
        elif self.token_position == "middle":
            # Compute the prefix (SOS) sub, obj, and suffix (EOS) tokens for the phrases
            prefix, sub, obj, suffix = self.compute_prefix_sufix(phrases)
            # Concatenate the prefix, sub, context, obj, and suffix to form the new prompt
            prompts = torch.cat(
                [
                    prefix, 
                    sub,
                    ctx_shifted,    
                    obj,
                    suffix,
                ],
                dim=1,
            )
        
        if eval:
            return prompts, self.tokenized_prompts, ctx_shifted
        
        return prompts, self.tokenized_prompts

class UVTransE(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, is_classifier=False):
        """Set up network module in UnionModel"""
        super(UVTransE, self).__init__()
        self.name = 'Union'
        self.in_dim = in_dim

        # For subject
        self.sub_w = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(True), 
            nn.Linear(512, 256), )
        # For object
        self.obj_w = nn.Sequential(
            nn.Linear(self.in_dim, 512), 
            nn.ReLU(True), 
            nn.Linear(512, 256), )

        # For union
        self.union_w = nn.Sequential(
            nn.Linear(self.in_dim, 512), 
            nn.ReLU(True), 
            nn.Linear(512, 256), )

        # Finally 2-layer fully connected layer for predicting relation
        self.final_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, out_dim), )

    def forward(self, union_f, sub_f, obj_f):
        sub_enc = self.sub_w(sub_f)
        obj_enc = self.obj_w(obj_f)
        union_enc = self.union_w(union_f)

        pred_enc = union_enc - sub_enc - obj_enc

        pred = self.final_fc(pred_enc)

        return pred
    
class Classifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=50, isNonLinear=False):
        super(Classifier, self).__init__()

        self.isNonLinear = isNonLinear

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.layer1(x)

        if self.isNonLinear:
            x = F.relu(x)

        x = self.layer2(x)
        return x

class PredicateEstimator:
    def __init__(self, args, device, isTest=False, resume=False):
        super(PredicateEstimator, self).__init__()

        self.args = args
        self.device = device

        clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
        
        clip_model.eval()

        self.prompt_learner = PromptLearner(clip_model, n_ctx=args.n_context_vectors, device=args.device, token_position=args.token_position).to(args.device)
        self.text_encoder = TextEncoder(clip_model).to(args.device)

        self.load_model(args, isTest, resume)

    def load_model(self, args, isTest=False, resume=False):

        self.classifier = Classifier(input_size=512, hidden_size=256, output_size=args.num_predicates, isNonLinear=args.is_non_linear).to(args.device)

        # Load UVTransE model if learnable
        if args.learnable_UVTransE:
            self.uv_transe = UVTransE().to(args.device)
            model_path = os.path.join(args.checkpoints_dir_prompt, f'{args.which_epoch}_combined_models.pth')
        else:
            model_path = os.path.join(args.checkpoints_dir_prompt, f'{args.which_epoch}_prompt_learner.pth')

        # Load saved state dict into the prompt learner model
        if os.path.exists(model_path):
            loaded_state_dicts = torch.load(model_path, map_location=args.device)
            if args.learnable_UVTransE:
                if args.randomize_UVTransE == False:
                    print("Loading UVTransE weights")
                    self.uv_transe.load_state_dict(loaded_state_dicts['UVTransE'], strict=False)
                
                if args.is_TTT == False:
                    print("Loading prompt learner weights")
                    self.prompt_learner.load_state_dict(loaded_state_dicts['prompt_learner'])
            else:
                if args.is_TTT == False:
                    print("Loading prompt learner weights")
                    self.prompt_learner.load_state_dict(loaded_state_dicts)
        else:
            print(Fore.RED + f"Warning: Model file '{model_path}' not found. Loading random weights instead.")
            print(Style.RESET_ALL)

        # Load the classifier model
        if isTest or resume:
            classifier_model_path = os.path.join(args.log_dir_cls, 'checkpoints_classifier', f'{args.which_epoch_cls}_combined.pth')
        else:
            classifier_model_path = 'None'
            
        if os.path.exists(classifier_model_path):

            loaded_state_dicts = torch.load(classifier_model_path, map_location=args.device)

            if args.is_TTT == False:
                print("Loading prompt learner weights")
                self.prompt_learner.load_state_dict(loaded_state_dicts['prompt_learner'])
            if args.learnable_UVTransE:
                print("Loading UVTransE weights")
                self.uv_transe.load_state_dict(loaded_state_dicts['UVTransE'])

            print("Loading classifier weights")
            self.classifier.load_state_dict(loaded_state_dicts['classifier'])
        else:
            print(Fore.RED + f"Warning: Classifier Model file '{classifier_model_path}' not found. Loading random weights instead.")
            print(Style.RESET_ALL)

    def configure_optimizers(self, optimizer_name='adam', lr=0.001):

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.classifier.parameters(), lr=lr)
        else:
            raise ValueError('Optimizer not supported')

        if self.args.update_prompt_learner and self.args.is_TTT == False:
            self.optimizer.add_param_group({'params': self.prompt_learner.parameters()})
        if self.args.update_UVTransE:
            self.optimizer.add_param_group({'params': self.uv_transe.parameters()})
        return self.optimizer

    def set_eval(self,):

        self.classifier.eval()
        if self.args.is_TTT == False:
            self.prompt_learner.eval()
        if self.args.learnable_UVTransE:
            self.uv_transe.eval()
    
    def set_train(self,):

        self.classifier.train()
        if self.args.update_prompt_learner and self.args.is_TTT == False:
            self.prompt_learner.train()
        if self.args.learnable_UVTransE and self.args.update_UVTransE:
            self.uv_transe.train()

    def set_input(self, data_dict):        
        self.union_img_emb = data_dict['union_emb'].float().squeeze(0).to(self.device)
        self.sub_emb = data_dict['sub_emb'].float().squeeze(0).to(self.device)
        phrases = data_dict['phrases']
        # Convert to a single list
        self.phrases = [value for tuple in phrases for value in tuple]
        self.obj_emb = data_dict['obj_emb'].float().squeeze(0).to(self.device)
        self.gt_predicate_id = data_dict["gt_predicate_ids"].squeeze(0).to(self.device) # in the range [0, 49] for 50 predicates ignoring the N/R predicate
        
        if self.args.is_TTT:
            self.phrases_emb = data_dict['phrases_emb'].float().squeeze(0).to(self.device)

    def get_estimate(self):
        if self.args.is_TTT == False:
            prompts, tokenized_prompts = self.prompt_learner(self.phrases, self.union_img_emb)
            union_emb = self.text_encoder(prompts.half(), tokenized_prompts)
        else:
            union_emb = self.phrases_emb

        if self.args.learnable_UVTransE:
            predicate_f = self.uv_transe(union_emb.float(), self.sub_emb.float(), self.obj_emb.float())
        else:            
            predicate_f = union_emb - (self.sub_emb + self.obj_emb)

        predicate_logits = self.classifier(predicate_f)

        return predicate_logits
    
    def get_loss(self, predicate_logits=None):
        if predicate_logits is None:
            predicate_logits = self.get_estimate()
        loss = F.cross_entropy(predicate_logits, self.gt_predicate_id)
        return loss
    
    def get_test_result(self, numpy=True, num_classes=50):
        predicate_logits = self.get_estimate()
        predicate_prob = F.softmax(predicate_logits, dim=1)

        # if the desired number of classes is 51, add a zero column(N/R predicate) to the beginning of the predicate_prob tensor
        if num_classes == 51:
            zeros = torch.zeros(predicate_prob.shape[0], 1).to(self.device)
            # concatenate the tensor of zeros with the original tensor along the last dimension
            predicate_prob = torch.cat((zeros, predicate_prob), dim=-1)
        if numpy:
            predicate_prob = predicate_prob.cpu().detach().numpy()
        return predicate_prob

