import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import random
import torch
from torchvision.transforms import functional as F
import clip
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.model_selection import train_test_split
import glob

class VGPredicateDataset(torch.utils.data.Dataset):
    def __init__(self, predicate_dict_dir='datasets/pred_dicts', images_dir='./datasets/images', 
                 device='cpu', debug=False, test=False, num_predicates=None):
        super().__init__()
        
        self.test = test
        self.debug = debug
        self.device = device
        self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=device)
        
        if test:
            predicate_dict_dir = 'datasets/pred_dicts_test'
        else:
            predicate_dict_dir = 'datasets/pred_dicts_train'
            
        self.triplet_data_dict = {}
        dict_paths = glob.glob(f'{predicate_dict_dir}/*')
        for dict_file in dict_paths:
            predicate_name = os.path.basename(dict_file).split('_')[0]
            with open(dict_file, "rb") as f:
                self.triplet_data_dict[predicate_name] = pickle.load(f)

        self.images_dir=images_dir
        self.entity_f = np.load(f'datasets/labels_CLIP.npy', allow_pickle=True)

        with open('datasets/VG-SGG-dicts-with-attri.json') as json_file:
            vg_dicts = json.load(json_file)

            idx_to_labels = vg_dicts["idx_to_label"]
            idx_to_predicate = vg_dicts["idx_to_predicate"]
            self.entity_categories = list(idx_to_labels.values())
            self.predicate_categories = list(idx_to_predicate.values())
            
        if num_predicates is None:
            predicates_of_interest = self.predicate_categories
        else:
            with open('datasets/predicate_counts_splits.json') as json_file:
                predicate_splits = json.load(json_file)
            predicates_of_interest = predicate_splits[f'{num_predicates}_predicates']
            
        #######################################################################################

        # Split the triplet data for each category
        
        self.triplet_data = []
        for category in predicates_of_interest:
            self.triplet_data += self.triplet_data_dict[category]
            print(f'# of "{category}" triplets: {len(self.triplet_data_dict[category])}')

        random.shuffle(self.triplet_data)
        print('Total Triplet data: ', len(self.triplet_data))

    def CLIP_txt_encode(self, promt):
        text = clip.tokenize(promt).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_text(text).squeeze(0)
        
    def CLIP_img_encode(self, image):        
        image = self.clip_processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(image).squeeze(0)

    def __getitem__(self, idx):
        anchor_data = self.triplet_data[idx]
        gt_predicate_id = anchor_data["gt_predicate_id"]
        
        # Select anchor
        anchor_union_emb, anchor_sub_label, anchor_sub_emb, anchor_obj_label, anchor_obj_emb, anchor_gt_predicate_id = self.extract_embeddings(anchor_data)
        anchor_phrase = f'{anchor_sub_label} {anchor_obj_label}'  # Merging subject and object labels

        # Create adverserial phrases for anchor by replacing subject and object labels
        sub_adver = random.choice([x for x in self.entity_categories if x != anchor_sub_label])
        obj_adver = random.choice([x for x in self.entity_categories if x != anchor_obj_label])
        anchor_sub_adverserial_phrase = f'{sub_adver} {anchor_obj_label}'
        anchor_obj_adverserial_phrase = f'{anchor_sub_label} {obj_adver}'

        gt_predicate_label = self.predicate_categories[gt_predicate_id]

        # Randomly choose a positive predicate
        positive_data = random.choice(self.triplet_data_dict[gt_predicate_label])
        positive_union_emb, positive_sub_label, positive_sub_emb, positive_obj_label, positive_obj_emb, positive_gt_predicate_id = self.extract_embeddings(positive_data)
        positive_phrase = f'{positive_sub_label} {positive_obj_label}' 

        # Randomly choose a negative predicate
        remaining_gt_predicates = [predicate for predicate in self.predicate_categories if predicate != gt_predicate_label]
        negative_predicate = random.choice(remaining_gt_predicates)
        negative_data = random.choice(self.triplet_data_dict[negative_predicate])
        negative_union_emb, negative_sub_label, negative_sub_emb, negative_obj_label, negative_obj_emb, negative_gt_predicate_id = self.extract_embeddings(negative_data)
        negative_phrase = f'{negative_sub_label} {negative_obj_label}' 

        out_dict = {
            'anchor_union_emb': anchor_union_emb,
            'anchor_sub_emb': anchor_sub_emb,
            'anchor_sub_label': anchor_sub_label,
            'anchor_obj_label': anchor_obj_label,
            'anchor_obj_emb': anchor_obj_emb,
            'anchor_phrase': anchor_phrase,
            'anchor_gt_predicate_id': anchor_gt_predicate_id,

            'positive_union_emb': positive_union_emb,
            'positive_sub_emb': positive_sub_emb,
            'positive_sub_label': positive_sub_label,
            'positive_obj_label': positive_obj_label,
            'positive_obj_emb': positive_obj_emb,
            'positive_phrase': positive_phrase,
            'positive_gt_predicate_id': positive_gt_predicate_id,

            'negative_union_emb': negative_union_emb,
            'negative_sub_emb': negative_sub_emb,
            'negative_sub_label': negative_sub_label,
            'negative_obj_label': negative_obj_label,
            'negative_obj_emb': negative_obj_emb,
            'negative_phrase': negative_phrase,
            'negative_gt_predicate_id': negative_gt_predicate_id,

            'anchor_sub_adverserial_phrase': anchor_sub_adverserial_phrase,
            'anchor_obj_adverserial_phrase': anchor_obj_adverserial_phrase,
        }
        
        if self.debug:
            self.display(anchor_data, positive_data, negative_data, out_dict)
        return out_dict
    
    def extract_embeddings(self,data):
        # load the images and convert them to tensors
        img_path = data["img_name"]

        sub_id_ = int(data["sub_id"])
        obj_id_ = int(data["obj_id"])

        sub_label = self.entity_categories[data["sub_id"]]
        obj_label = self.entity_categories[data["obj_id"]]

        sub_txt_emb = self.entity_f[sub_id_].squeeze(0)
        obj_txt_emb = self.entity_f[obj_id_].squeeze(0)

        union_img_embedding = data["union_img_embedding"]
        
        
        return union_img_embedding, sub_label, sub_txt_emb, obj_label, obj_txt_emb, data["gt_predicate_id"]

    def __len__(self):
        return len(self.triplet_data)

    def display(self, anchor_data, positive_data, negative_data, out_dict):
        
        def compare_embeddings(data, out_dict, data_type):
            # load the images and convert them to tensors
            img_path = data["img_name"]
            img = Image.open(os.path.join(self.images_dir, img_path))
            
            img_tensor = F.to_tensor(img)
            img_np = np.array(img_tensor.permute(1, 2, 0).numpy()*255, dtype=np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            sub_label = self.entity_categories[data["sub_id"]]
            obj_label = self.entity_categories[data["obj_id"]]
            gt_predicate_label = self.predicate_categories[data["gt_predicate_id"]]

            sub_bbox = data["sub_bbox"]
            obj_bbox = data["obj_bbox"]
            union_bbox = data["union_bbox"]

            union_img = F.crop(img, *union_bbox)

            union_img_embedding = self.CLIP_img_encode(union_img)
            sub_txt_emb = self.CLIP_txt_encode(sub_label)        
            obj_txt_emb = self.CLIP_txt_encode(obj_label)

            union_sim = cos_sim(np.expand_dims(out_dict[f'{data_type}_union_emb'], axis=0), 
                                np.expand_dims(union_img_embedding.cpu().detach().numpy(), axis=0))
            sub_sim = cos_sim(np.expand_dims(out_dict[f'{data_type}_sub_emb'], axis=0), 
                              np.expand_dims(sub_txt_emb.cpu().detach().numpy(), axis=0))
            obj_sim = cos_sim(np.expand_dims(out_dict[f'{data_type}_obj_emb'], axis=0), 
                              np.expand_dims(obj_txt_emb.cpu().detach().numpy(), axis=0))
            
            print(f'\n\n{data_type} text:\t {sub_label} {gt_predicate_label} {obj_label}' )
            print(f'{data_type} union embedding similarity:\t{union_sim}')
            print(f'{data_type} sub embedding similarity:\t{sub_sim}')
            print(f'{data_type} obj embedding similarity:\t{obj_sim}')
            
            # draw the union bbox in green
            cv2.rectangle(img_np, tuple(union_bbox[:2]), tuple(union_bbox[2:]), (0, 255, 0), 2)
            cv2.putText(img_np, gt_predicate_label, (union_bbox[0], union_bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # draw the sub bbox in red
            cv2.rectangle(img_np, tuple(sub_bbox[:2]), tuple(sub_bbox[2:]), (255, 0, 0), 2)
            cv2.putText(img_np, sub_label, (sub_bbox[0], sub_bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # draw the obj bbox in blue
            cv2.rectangle(img_np, tuple(obj_bbox[:2]), tuple(obj_bbox[2:]), (0, 0, 255), 2)
            cv2.putText(img_np, obj_label, (obj_bbox[0], obj_bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(f'{data_type}.jpg', img_np)

        compare_embeddings(anchor_data, out_dict, 'anchor')
        compare_embeddings(positive_data, out_dict, 'positive')
        compare_embeddings(negative_data, out_dict, 'negative')

if __name__ == "__main__":
    # Initialize dataset and get a random item
    dataset = VGPredicateDataset( predicate_dict_dir='datasets/datasets/pred_dicts_train', 
                                    images_dir='./datasets/images',
                                    device='cpu', debug=True, test=False, num_predicates=5)
    data = dataset[80]