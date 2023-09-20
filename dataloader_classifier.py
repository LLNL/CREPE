from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys, glob
from tkinter import image_names
import numpy as np
import cv2
import json

import PIL.Image as Image
import pickle
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import clip

from tqdm import tqdm

class VGDataset(data.Dataset):
    def __init__(self, data_root='/p/gpfs1/rakshith/datasets/VG/np_files', 
                    split='val', case=1, feature_type='prompt', num_predicates=50):
        """
        Data loader for the VG dataset

        Parameters
        ----------
        data_root (str): Path to the dataset folder
        split (str) : Choice for the data split 'val' and 'train'  
        case (int) : Choice of different eval cases 
            1 : Predicate Classification (taking ground truth bounding boxes and labels as inputs)
        feature_type (str) : Choice for different feature types 
            1.prompt: Use features learned from prompting; size 512
            2.CK: Use CLIP C_K features; size 512
            3.frcnn: fasterRCNN features; size 4096
        """

        self.data_root = data_root
        self.split = split
        self.case = case
        self.feature_type = feature_type
        
        
        self.entity_f_path = 'datasets/'

        self.root_path = f'{data_root}/case_{case}/{split}'
        
        with open(f'{data_root}/{split}_image_names.pickle', 'rb') as handle:
            self.img_names = pickle.load(handle)

        self.total_images = (len(self.img_names)//10)*10
        if split == 'train':
            self.step = 5772
        
        elif split =='val':
            self.step = 5000
        
        elif split == 'test':
            self.step = 2644
        
        self.start_idx = 0
        self.end_idx = self.step
        self._load_data(self.start_idx, self.end_idx)
        print(f'Total number of {split} triplets for case {case}: {len(self.img_names)}\n')


        with open('metrics/ind_to_predicates.pkl', "rb") as input_file:
            self.ind_to_predicates = pickle.load(input_file)
            
        with open('datasets/VG-SGG-dicts-with-attri.json') as json_file:
            vg_dicts = json.load(json_file)

        self.idx_to_label = vg_dicts["idx_to_label"]
        self.idx_to_predicate = vg_dicts["idx_to_predicate"]
        self.idx_to_predicate['0'] = 'N/R'
        self.idx_to_label['0'] = 'BG'

        self.num_predicates = num_predicates
        with open('datasets/predicate_idx_splits.json') as json_file:
            predicate_splits = json.load(json_file)
            self.predicates_of_interest = list(predicate_splits[f'{num_predicates}_predicates'].values())

        print('Total number of predicates: ', len(self.predicates_of_interest))
        for k,v in predicate_splits[f'{num_predicates}_predicates'].items():
            print(k,v)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=self.device)

    def _load_data(self, start_idx, end_idx):

        if self.feature_type == 'prompt' or self.feature_type == 'CK':
            # entity_f is a list of features that follows idx_to_label with idx starting from 0 (airplane features is at index 0)
            # This list doesnt include features background class
            self.entity_f = np.load(f'{self.entity_f_path}labels_CLIP.npy', allow_pickle=True)
            # The union features have an unwanted dimension in position 1
            self.union_f = np.load(f'{self.root_path}/{start_idx}_{end_idx}_CLIP_img_union_f.npy', allow_pickle=True)
        
        elif self.feature_type == 'frcnn':
            self.entity_f = np.load(f'{self.root_path}/{start_idx}_{end_idx}_frcnn_entity_f.npy', allow_pickle=True)
            self.union_f = np.load(f'{self.root_path}/{start_idx}_{end_idx}_frcnn_union_f.npy', allow_pickle=True)
        
        self.entity_labels = np.load(f'{self.root_path}/{start_idx}_{end_idx}_entity_labels.npy', allow_pickle=True)
        self.entity_bb = np.load(f'{self.root_path}/{start_idx}_{end_idx}_entity_bb.npy', allow_pickle=True)
        self.entity_scores = np.load(f'{self.root_path}/{start_idx}_{end_idx}_entity_scores.npy', allow_pickle=True)
        self.rel_pair_idxs = np.load(f'{self.root_path}/{start_idx}_{end_idx}_rel_pair_idxs.npy', allow_pickle=True)
        
        self.gt_bb = np.load(f'{self.root_path}/{start_idx}_{end_idx}_gt_bb.npy', allow_pickle=True)
        self.gt_labels = np.load(f'{self.root_path}/{start_idx}_{end_idx}_gt_labels.npy', allow_pickle=True)
        self.gt_predicates = np.load(f'{self.root_path}/{start_idx}_{end_idx}_gt_predicates.npy', allow_pickle=True)
        self.gt_rels = np.load(f'{self.root_path}/{start_idx}_{end_idx}_gt_rels.npy', allow_pickle=True)
        
        self.img_sizes = np.load(f'{self.root_path}/{start_idx}_{end_idx}_img_sizes.npy', allow_pickle=True)

    def __getitem__(self, idx):
        
        """
        Outputs the data for the given index referincing an image in the VG dataset:
        'img_name': name of the image,            
        'im_width', 'im_height': width and height of the image,
        'phrases': Sub and Obj labels as a string with space in between (in the order of the rel_pair_idx)
        'sub_emb': CLIP text embedding of the subject, (shape: (num_rels, 512))
        'obj_emb': CLIP text embedding of the object, (shape: (num_rels, 512))            
        'union_emb': CLIP image embedding of the union of the subject and object, (shape: (num_rels, 512))
        'gt_predicate_ids': predicate ids of the ground truth relationships, (shape: (num_rels, 1)) 
                                    ****** the values are in the range [0, 49] ********** We subtract 1 as we ignore the N/R predicate

        'rel_pair_idx': indices of the subject and object in the image, (shape: (num_rels, 2)), the N/R predicate is ignored
        'gt_rels': ground truth relationships, (shape: (num_gt_rels, 2))
        'gt_boxes': ground truth bounding boxes for the entities (sub, obj), (shape: (num_gt_enti, [x1, y1, x2, y2])) 
        'gt_classes': ground truth labels for the entities (sub, obj), (shape: (num_gt_enti, 1))
        'pred_boxes': predicted bounding boxes for the entities (sub, obj), (shape: (num_pred_enti, [x1, y1, x2, y2]))
        'pred_classes': predicted labels for the entities (sub, obj), (shape: (num_pred_enti, 1))
        'obj_probs': predicted scores for the entities (sub, obj), (shape: (num_pred_enti, 151)) including the background class
        """

        a = idx//self.step
        if idx == self.end_idx or idx > self.end_idx:
            self.start_idx = self.step*a
            self.end_idx = self.step*(a+1)
            self._load_data(self.start_idx, self.end_idx)

        img_name = self.img_names[idx]

        idx = idx - self.step*a
        _rel_pair_idxs = []
        sub_f, obj_f, union_f = [], [], []
        sub_bb, obj_bb, union_bb = [], [],[]
        im_width, im_height = [], []
        phrases = []
        phrases_f = []
        predicates = []
        
        im_width = self.img_sizes[idx][1]
        im_height = self.img_sizes[idx][0]

        for i , (a,b) in enumerate(self.rel_pair_idxs[idx]):        
            
            # ignore the 'N/R' predicate 
            if self.gt_predicates[idx][i] == 0 or self.gt_predicates[idx][i] not in self.predicates_of_interest:
                continue

            if self.num_predicates < 50:
                # Replace the predicate ID to select from the splits list
                predicates.append(self.predicates_of_interest.index(self.gt_predicates[idx][i]))
            else:
                # Reduce the predicate value by 1 to account for the 'N/R' predicate
                predicates.append(self.gt_predicates[idx][i] - 1)
            union_f.append(self.union_f[idx][i])
            
            # -1 if for accounting for the 'BG' class which is not there in the features
            sub_idx = self.entity_labels[idx][a]-1
            obj_idx = self.entity_labels[idx][b]-1

            sub_label = self.idx_to_label[str(self.entity_labels[idx][a])]
            obj_label = self.idx_to_label[str(self.entity_labels[idx][b])]

            phrase = sub_label + ' ' + obj_label

            phrase_f = self.CLIP_txt_encode(phrase).cpu().numpy()
            phrases_f.append(phrase_f)

            phrases.append(phrase)
            sub_f.append(self.entity_f[sub_idx][0])
            obj_f.append(self.entity_f[obj_idx][0])
            
            _rel_pair_idxs.append([a,b])
            # Converting x1,y1,x2,y2 to x1,x2,y1,y2; sub obj and union are in x1,x2,y1,y2
            sub_bb.append([self.entity_bb[idx][a][0], self.entity_bb[idx][a][2], 
                                self.entity_bb[idx][a][1], self.entity_bb[idx][a][3]])
            obj_bb.append([self.entity_bb[idx][b][0], self.entity_bb[idx][b][2], 
                                self.entity_bb[idx][b][1], self.entity_bb[idx][b][3]])
            _union_bb = self._calculate_union_bb(self.entity_bb[idx][a], self.entity_bb[idx][b])
            union_bb.append(_union_bb)

        sub_bb = np.array(sub_bb).astype(int)
        obj_bb = np.array(obj_bb).astype(int)
        union_bb = np.array(union_bb).astype(int)

        if len(union_f) == 0:
            temp_dict = {
                'img_name': 'None',            
                'im_width': np.array(im_width).astype(int),
                'im_height': np.array(im_height).astype(int),
                'phrases': phrases,
                'sub_emb': np.array(sub_f).astype(np.float32),
                'obj_emb': np.array(obj_f).astype(np.float32),            
                'union_emb': union_f,
                'gt_predicate_ids': np.array(predicates).astype(int), # the values are in the range [0, 49] We subtract 1 as we ignore the N/R predicate
                'rel_pair_idx': np.array(_rel_pair_idxs).astype(int),
                'gt_rels': np.array(self.gt_rels[idx]).astype(int),
                'gt_boxes': np.array(self.gt_bb[idx]).astype(int),
                'gt_classes': np.array(self.gt_labels[idx]).astype(int),
                'pred_boxes': np.array(self.entity_bb[idx]).astype(int),
                'pred_classes': np.array(self.entity_labels[idx]).astype(int),
                'obj_probs': np.array(self.entity_scores[idx]).astype(np.float32),
            }
            return temp_dict
        
        union_f = np.array(union_f).astype(np.float32)
        if union_f.ndim == 3:
            union_f = union_f[:,0,:]
        

        temp_dict = {
            'img_name': img_name,            
            'im_width': np.array(im_width).astype(int),
            'im_height': np.array(im_height).astype(int),
            'phrases': phrases,
            'phrases_emb': np.array(phrases_f).astype(np.float32),
            'sub_emb': np.array(sub_f).astype(np.float32),
            'obj_emb': np.array(obj_f).astype(np.float32),            
            'union_emb': union_f,
            'gt_predicate_ids': np.array(predicates).astype(int), # the values are in the range [0, 49] We subtract 1 as we ignore the N/R predicate
            'rel_pair_idx': np.array(_rel_pair_idxs).astype(int),
            'gt_rels': np.array(self.gt_rels[idx]).astype(int),
            'gt_boxes': np.array(self.gt_bb[idx]).astype(int),
            'gt_classes': np.array(self.gt_labels[idx]).astype(int),
            'pred_boxes': np.array(self.entity_bb[idx]).astype(int),
            'pred_classes': np.array(self.entity_labels[idx]).astype(int),
            'obj_probs': np.array(self.entity_scores[idx]).astype(np.float32),
        }
        return temp_dict

    def _calculate_union_bb(self, box1, box2):
        """
        Calculate the union of the box1 and box2

        Parameters
        ----------
        box1 (list): [x1,y1,x2,y2]
        box2 (list): [x1,y1,x2,y2]

        Returns
        -------
        union_box(list): [x1,x2,y1,y2]

        """

        x1 = min(box1[0], box2[0])
        x2 = max(box1[2], box2[2])
        y1 = min(box1[1], box2[1])
        y2 = max(box1[3], box2[3])

        return [x1,x2,y1,y2]

    def __len__(self):
        
        return self.total_images
    
    def display(self, data_dict, feature_check=False):
        img_name = data_dict['img_name'] 
        im_width = int(data_dict['im_width'] )
        im_height = int(data_dict['im_height'] )

        phrases = data_dict['phrases']
        gt_rels = data_dict['gt_rels']
        gt_labels = data_dict['gt_classes']
        gt_bb = data_dict['gt_boxes']

         
        rel_pair_idx = data_dict['rel_pair_idx']
        pred_labels = data_dict['pred_classes']
        predicates = data_dict['gt_predicate_ids']
        pred_bb = data_dict['pred_boxes']

        sub_f = data_dict['sub_emb']
        obj_f = data_dict['obj_emb']
        union_f = data_dict['union_emb']

        img = cv2.imread(f'/p/lustre1/rakshith/datasets/SGG/VG/VG_100K/{img_name}')
        img = cv2.resize(img, (im_width, im_height), interpolation = cv2.INTER_AREA)

        for i, (a,b) in enumerate(rel_pair_idx):
            img_1 = img.copy()

            phrase = phrases[i]
            sub_label = self.idx_to_label[str(pred_labels[a])]
            pred_label = self.idx_to_predicate[str(predicates[i]+1)] # +1 as we reduced the value to account for removing N/R predicate
            obj_label = self.idx_to_label[str(pred_labels[b])]
            txt = f'{sub_label} {pred_label} {obj_label}'                
            print(f'Pred:\t{txt}\t{phrase} \n')
            
            if predicates[i] != 0:

                union_bb = self._calculate_union_bb(pred_bb[a], pred_bb[b])
                
                img_1 = cv2.rectangle(img_1, (pred_bb[a][0], pred_bb[a][1]), (pred_bb[a][2], pred_bb[a][3]), (255, 0, 0), 2)
                img_1 = cv2.rectangle(img_1, (pred_bb[b][0], pred_bb[b][1]), (pred_bb[b][2], pred_bb[b][3]), (0, 255, 0), 2)                
                img_1 = cv2.rectangle(img_1, (union_bb[0], union_bb[2]), (union_bb[1], union_bb[3]), (0, 0, 255), 2)
                
                if feature_check:                    
                    import clip
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model,preprocess = clip.load(name='ViT-B/32', device=device)
                    
                    text_token = clip.tokenize([sub_label, obj_label]).to(device)

                    cropped_union = Image.fromarray(img[union_bb[2]:union_bb[3], union_bb[0]:union_bb[1],:])
                    pros_union = preprocess(cropped_union).to(device).unsqueeze(0)
                    with torch.no_grad():
                            union_features = model.encode_image(pros_union).cpu().detach().numpy()
                        
                            text_features = model.encode_text(text_token).cpu().detach().numpy()

                    union_sim = cos_sim(np.expand_dims(union_features[0], axis=0), 
                                        np.expand_dims(union_f[i], axis=0))
                    sub_sim = cos_sim(np.expand_dims(text_features[0], axis=0), 
                                    np.expand_dims(sub_f[i], axis=0))
                    obj_sim = cos_sim(np.expand_dims(text_features[1], axis=0), 
                                    np.expand_dims(obj_f[i], axis=0))
                    
                    print(f'union embedding cos similarity:\t{union_sim}')
                    print(f'sub embedding cos similarity:\t{sub_sim}')
                    print(f'obj embedding cos similarity:\t{obj_sim}')
   
                img_1 = cv2.putText(img_1, txt, (union_bb[0], union_bb[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f'{i}_test.jpg', img_1)

        print('\n\n')
        for i, (a,b,c) in enumerate(gt_rels):

            img_gt = img.copy()
            
            union_bb = self._calculate_union_bb(gt_bb[a], gt_bb[b])
                
            img_gt = cv2.rectangle(img_gt, (gt_bb[a][0], gt_bb[a][1]), (gt_bb[a][2], gt_bb[a][3]), (255, 0, 0), 2)
            img_gt = cv2.rectangle(img_gt, (gt_bb[b][0], gt_bb[b][1]), (gt_bb[b][2], gt_bb[b][3]), (0, 255, 0), 2)                
            img_gt = cv2.rectangle(img_gt, (union_bb[0], union_bb[2]), (union_bb[1], union_bb[3]), (0, 0, 255), 2)
                
            sub_label = self.idx_to_label[str(gt_labels[a])]
            pred_label = self.idx_to_predicate[str(c)]
            obj_label = self.idx_to_label[str(gt_labels[b])]
            txt = f'{sub_label} {pred_label} {obj_label}'

            img_gt = cv2.putText(img_gt, txt, (union_bb[0], union_bb[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print('GT:\t', txt)
            cv2.imwrite(f'{i}_test_gt.jpg', img_gt)
        print('\n\n')

    def CLIP_txt_encode(self, promt):
        text = clip.tokenize(promt).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_text(text).squeeze(0)
        
if __name__ == '__main__':
    dataset = VGDataset(split='train', case=1, data_root='/p/lustre1/rakshith/datasets/SGG/VG/np_files_2', num_predicates=50)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print('Total number of datapoints: ', len(dataset))
    # data = dataset.__getitem__(19980)

    # print('\n\nImg_name: ', data['img_name'])
    # print('predicted_entities: ', data['pred_classes'].shape, 'gt_entities: ', data['gt_classes'].shape)
    # print('predicted_entities_probabilities: ', data['obj_probs'].shape)
    # print('possible_relations: ', data['rel_pair_idx'].shape, 'gt_relations: ', data['gt_rels'].shape)
    # print('gt_predicates_possible_relations: ', data['gt_predicate_ids'].shape)
    # print('Sub_features: ',data['sub_emb'].shape, 'obj_features: ',data['obj_emb'].shape, 'union_features: ',data['union_emb'].shape)
    # print('\n\n')


    # dataset.display(data, feature_check=True)

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        if data['img_name'] == 'None':
            continue
        
        dataset.display(data, feature_check=True)
        assert False

    count = 0
    for data in tqdm(data_loader):

        if data['img_name'][0] == 'None':
            continue
        
        count += 1

        print('\n\nImg_name: ', data['img_name'])
        print('predicted_entities: ', data['pred_classes'].shape, 'gt_entities: ', data['gt_classes'].shape)
        print('predicted_entities_probabilities: ', data['obj_probs'].shape)
        print('possible_relations: ', data['rel_pair_idx'].shape, 'gt_relations: ', data['gt_rels'].shape)
        print('gt_predicates_possible_relations: ', data['gt_predicate_ids'].shape)
        print('Sub_features: ',data['sub_emb'].shape, 'obj_features: ',data['obj_emb'].shape, 'union_features: ',data['union_emb'].shape)
        print('\n\n')

    print(count)