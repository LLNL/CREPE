import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
import pickle
from abc import ABC, abstractmethod
import tqdm as tqdm

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def compute_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)
        
    def register_container(self, mode):
        # self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}
        
        self.result_dict[mode + '_recall'] = {}
        for key in [5, 10, 15, 20, 25, 30, 50, 100]:
            self.result_dict[mode + '_recall'][key] = []

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        rel_pair_idx = local_container['rel_pair_idx']
        rel_probs = local_container['rel_probs']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_probs = local_container['obj_probs']

        iou_thres = global_container['iou_thres']

        # print('rel_pair_idx', rel_pair_idx.shape, 'obj_probs', obj_probs.shape, 'rel_probs', rel_probs.shape)
        # Sort the predicted possible based on the (sub_score*pred_score*obj_score)
        pred_rel_inds, obj_scores, rel_scores, _ = sort_the_relations(rel_pair_idx, obj_probs, torch.from_numpy(rel_probs))
        # print('pred_rel_inds', pred_rel_inds.shape, 'obj_scores', obj_scores.shape, 'rel_scores', rel_scores.shape)

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1))) # (#predted relations, (sub_id, obj_id, pred_lab))
        # print('pred_rels', pred_rels.shape)
        pred_scores = rel_scores[:,1:].max(1) # #predted relations
        # print('pred_scores',pred_scores.shape)

        # print('\ngt_rels', gt_rels.shape, 'pred_rels', pred_rels.shape)
        # print('gt_boxes',gt_boxes.shape, 'pred_boxes',pred_boxes.shape)
        # Converts (m1, (sub_id, obj_id, pred_label)) to (m1, (sub_label, obj_label, pred_label))
        # Transforms (#entities, bbox) to (#triplets, bbox)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)


        # print('\ngt_triplets',gt_triplets.shape, 'pred_triplets',pred_triplets.shape)
        # print('gt_triplet_boxes',gt_triplet_boxes.shape, 'pred_triplet_boxes',pred_triplet_boxes.shape)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # print(len(pred_to_gt[:k]), len(match), gt_rels.shape[0])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)

        return local_container

"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

        self.rtypes = {
            'Head': ['on', 'has', 'of', 'wearing', 'in', 'near', 'with', 'behind', 'holding'],
            'Mid': ['wears', 'above', 'sitting on', 'under', 'riding', 'in front of', 'standing on'],
            'Tail': ['attached to', 'belonging to', 'at', 'walking on', 'carrying', 'over', 'watching',
                    'hanging from', 'for', 'looking at', 'parked on', 'eating', 'laying on', 'between', 
                    'covering', 'and', 'covered in', 'using', 'along', 'on back of', 'to', 'mounted on', 
                    'part of', 'lying on', 'walking in', 'from', 'painted on', 'growing on', 'across', 
                    'against', 'made of', 'playing', 'flying in', 'says']
        }
    def register_container(self, mode):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        # self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        # self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}
        # self.result_dict[mode + '_mean_recall_HMT'] = {'Head':{20: 0.0, 50: 0.0, 100: 0.0},
        #                                                 'Mid': {20: 0.0, 50: 0.0, 100: 0.0},
        #                                                 'Tail': {20: 0.0, 50: 0.0, 100: 0.0}}
        
        self.result_dict[mode + '_recall_hit'] = {}
        self.result_dict[mode + '_recall_count'] = {}
        self.result_dict[mode + '_mean_recall'] = {}
        self.result_dict[mode + '_mean_recall_collect'] = {}
        self.result_dict[mode + '_mean_recall_list'] = {}
        self.result_dict[mode + '_mean_recall_HMT'] = {'Head':{}, 'Mid': {}, 'Tail': {}}
        for key in [5, 10, 15, 20, 25, 30, 50, 100]:
            self.result_dict[mode + '_mean_recall'][key] = 0.0
            self.result_dict[mode + '_mean_recall_collect'][key] = [[] for i in range(self.num_rel)]
            self.result_dict[mode + '_mean_recall_list'][key] = []
            self.result_dict[mode + '_mean_recall_HMT']['Head'][key] = 0.0
            self.result_dict[mode + '_mean_recall_HMT']['Mid'][key] = 0.0
            self.result_dict[mode + '_mean_recall_HMT']['Tail'][key] = 0.0
            
    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:

            result_str += '----------------------- HMT ------------------------\n'
            for rtype, recalls in self.result_dict[mode + '_mean_recall_HMT'].items():
                result_str += f'SGG eval {rtype}: '
                for k, v in recalls.items():
                    result_str += '   mR @ %d: %.4f; ' % (k, float(v))
                result_str += '\n'
            result_str += '\n'
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode):
        no_samples_rtype = {'Head': len(self.rtypes['Head']),
                            'Mid': len(self.rtypes['Mid']),
                            'Tail':len(self.rtypes['Tail'])}
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

                for rtype, relations in self.rtypes.items():
                    if self.rel_name_list[idx] in relations:
                        self.result_dict[mode + '_mean_recall_HMT'][rtype][k] += (tmp_recall/no_samples_rtype[rtype])

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        # self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_accumulate_recall'] = {}
        for key in [5, 10, 15, 20, 25, 30, 50, 100]:
            self.result_dict[mode + '_accumulate_recall'][key] = []

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    try:
        triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    except:
        print(relations)
        print(classes, sub_id, ob_id)

        print(classes.shape, sub_id.shape, ob_id.shape)
        assert False
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores

def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    # print('\ngt_pred_triplet_CM ', keeps.shape)
    # print('#gt_with_pred_match ', np.where(gt_has_match)[0].shape, 'gt_idx_pred_match ', keeps[gt_has_match].shape)
    
    # Iterate over the GT triplets that matches with the predicted triplets
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        # Extract all the predicted boxes that match the current GT triplet
        boxes = pred_boxes[keep_inds]
        
        # Of these pred boxes filter the ones that satisfies the IOU condition the current GT bboxe
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = compute_iou(gt_box_union[None], box_union)[0] >= iou_thres
        else:

            sub_iou = compute_iou(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = compute_iou(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)
        
        # print(np.where(keep_inds)[0], np.where(keep_inds)[0][inds])

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    # print('pred_to_gt ', len(pred_to_gt))
    return pred_to_gt

def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, mode):

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)

    return 


def sort_the_relations(rel_pair_idx, obj_class_prob, rel_class_prob):

    # print('rel_pair_idx ', rel_pair_idx.shape, 'obj_class_prob ', obj_class_prob.shape, 'rel_class_prob ', rel_class_prob.shape)
    # Getting the object ids from the object probabilities ignoring the background
    obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
    obj_pred = obj_pred + 1 # adding 1 as the background was ignored in previous step
    
    # sorting triples according to score production
    # Splitting the obj_scores into sub and obj based on the rel_pair_idx
    obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
    obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
    # print('obj_scores0 ', obj_scores0.shape, 'obj_scores1 ', obj_scores1.shape)
    # Getting the relation ids from the relation probabilities ignoring the N/R
    rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
    rel_class = rel_class + 1 # adding 1 as the N/R was ignored in previous step
    # print('rel_scores ', rel_scores.shape, 'rel_class ', rel_class.shape)

    # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
    triple_scores = rel_scores * obj_scores0 * obj_scores1
    _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
    rel_pair_idx = rel_pair_idx[sorting_idx] # (#rel, 2)

    rel_class_prob = rel_class_prob[sorting_idx]# (#rel, #rel_class)
    rel_labels = rel_class[sorting_idx] # (#rel, )

    return rel_pair_idx, obj_scores.numpy(), rel_class_prob.numpy(), rel_labels.numpy()

if __name__ == "__main__":

    # with open('metrics/local_container.pkl', "rb") as input_file:
    #     local_container = pickle.load(input_file)
    with open('metrics/ind_to_predicates.pkl', "rb") as input_file:
        ind_to_predicates = pickle.load(input_file)
    # with open('metrics/groundtruths.pkl', "rb") as input_file:
    #     groundtruths = pickle.load(input_file)
    # with open('metrics/predictions.pkl', "rb") as input_file:
    #     predictions = pickle.load(input_file)

    modes = ['predcls', 'sgcls', 'sgdet']
    result_dict = {}


    # prepare all inputs
    global_container = {}
    global_container['zeroshot_triplet'] = None
    global_container['result_dict'] = result_dict
    global_container['mode'] = modes[0]
    global_container['multiple_preds'] = False
    global_container['num_rel_category'] = 51
    global_container['iou_thres'] = 0.5
    global_container['attribute_on'] = False
    global_container['num_attributes'] = 201


    num_rel_category = 51
    result_dict = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode=modes[0])
    mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
    mean_recall.register_container(mode=modes[0])


    for _ in range(10):
        
        # Keep creating local containers with the predictions
        local_container = eval_recall.calculate_recall(global_container, local_container, modes[0])

        mean_recall.collect_mean_recall_items(global_container, local_container, mode=modes[0])

    mean_recall.calculate_mean_recall(mode = modes[0])
    
    string = eval_recall.generate_print_string(mode = modes[0])
    string += mean_recall.generate_print_string(mode = modes[0])

    print(string)