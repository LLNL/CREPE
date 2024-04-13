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
            predicate_dict_dir = 'datasets/pred_dicts_test_cmr'
        else:
            predicate_dict_dir = 'datasets/pred_dicts_train_cmr'
            
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
            
        self.predicates_of_interest = list(predicates_of_interest.keys())

        #######################################################################################

        # Split the triplet data for each category
        
        self.triplet_data = []
        for category in predicates_of_interest:
            # train_data, test_data = train_test_split(self.triplet_data_dict[category], test_size=0.2, random_state=42)
            # if self.test:
            #     self.triplet_data_dict[category] = test_data
            #     self.triplet_data += test_data
            # else:
            #     self.triplet_data_dict[category] = train_data
            #     self.triplet_data += train_data

            self.triplet_data += self.triplet_data_dict[category]

            
            print(f'No of "{category}" triplets: {len(self.triplet_data_dict[category])}')
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
        
        # load the images and convert them to tensors
        img_path = anchor_data["img_name"]

        sub_id_ = int(anchor_data["sub_id"])
        obj_id_ = int(anchor_data["obj_id"])

        sub_label = self.entity_categories[anchor_data["sub_id"]]
        obj_label = self.entity_categories[anchor_data["obj_id"]]

        sub_txt_emb = self.entity_f[sub_id_].squeeze(0)
        obj_txt_emb = self.entity_f[obj_id_].squeeze(0)

        phrase = f'{sub_label} {obj_label}'

        union_img_embedding = anchor_data["union_img_embedding"]
        union_cmr_embedding = anchor_data["union_cmr_embedding"]

        out_dict = {

            'union_img_emb': union_img_embedding,
            'union_cmr_emb': union_cmr_embedding,
            'sub_emb': sub_txt_emb,
            'obj_emb': obj_txt_emb,
            'sub_label': sub_label,
            'obj_label': obj_label,
            'phrases': phrase,
            'gt_predicate_id': gt_predicate_id,
            'im_width': anchor_data['im_width'],
            'im_height': anchor_data['im_height'],
            'img_name': img_path,

        }
        
        if self.debug:
            self.display(anchor_data, out_dict)
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
        union_cmr_embedding = data["union_cmr_embedding"]
        
        
        return union_img_embedding, union_cmr_embedding, sub_label, sub_txt_emb, obj_label, obj_txt_emb, data["gt_predicate_id"]

    def __len__(self):
        return len(self.triplet_data)

    def display(self, anchor_data, out_dict):
        
        def compare_embeddings(data, out_dict, data_type):
            # load the images and convert them to tensors
            img_name = data["img_name"]

            image_path = os.path.join(self.images_dir, img_name)
            img = cv2.imread(image_path)
            img_np = cv2.resize(img, (out_dict['im_height'], out_dict['im_width']), interpolation = cv2.INTER_AREA)

            sub_label = self.entity_categories[data["sub_id"]]
            obj_label = self.entity_categories[data["obj_id"]]
            gt_predicate_label = self.predicate_categories[data["gt_predicate_id"]]

            sub_bbox = data["sub_bbox"]
            obj_bbox = data["obj_bbox"]
            union_bbox = data["union_bbox"]
            phrases = out_dict["phrases"]
                        
            # Define the coordinates of the union bounding box
            x1, y1, x2, y2 = data["union_bbox"]
            # Crop the union image from the original image
            union_img = img_np[y1:y2, x1:x2]
            union_img_pil = Image.fromarray(union_img)

            # Save the PIL image
            union_img_pil.save(f'{data_type}_union.jpg')
            union_img_embedding = self.CLIP_img_encode(union_img_pil)
            sub_txt_emb = self.CLIP_txt_encode(sub_label)        
            obj_txt_emb = self.CLIP_txt_encode(obj_label)

            union_sim = cos_sim(np.expand_dims(out_dict[f'union_img_emb'], axis=0), 
                                np.expand_dims(union_img_embedding.cpu().detach().numpy(), axis=0))
            sub_sim = cos_sim(np.expand_dims(out_dict['sub_emb'], axis=0), 
                              np.expand_dims(sub_txt_emb.cpu().detach().numpy(), axis=0))
            obj_sim = cos_sim(np.expand_dims(out_dict['obj_emb'], axis=0), 
                              np.expand_dims(obj_txt_emb.cpu().detach().numpy(), axis=0))
            
            print(f'\n\n{data_type} text:\t {sub_label} {gt_predicate_label} {obj_label}' )
            print(f'{data_type} phrase:\t {phrases}' )
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

if __name__ == "__main__":
    # Initialize dataset and get a random item
    dataset = VGPredicateDataset( predicate_dict_dir='datasets/datasets/pred_dicts_train_cmr', 
                                    images_dir='./datasets/images',
                                    device='cpu', debug=True, test=False, num_predicates=50)
    data = dataset[80]