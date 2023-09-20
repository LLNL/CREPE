import os
import argparse
import datetime
import pickle

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import cv2

from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataloader_classifier import VGDataset
from models.models import PredicateEstimator
from metrics.metrics import SGRecall, SGMeanRecall


def compute_classification_accuracy(predicted_probabilities, target_classes):

    # Ignore the background class
    if not isinstance(predicted_probabilities, torch.Tensor):
        predicted_probabilities = torch.from_numpy(predicted_probabilities)
    
    predicted_classes = torch.max(predicted_probabilities, dim=1)[1] # Torch.max returns (values, indices)


    # Ignore the background class while computing the accuracy
    if args.num_predicates == 50:
        # Get the indices of background class from the target classes
        relation_indices = torch.where(target_classes != 0)[0]
        # Filter the predicted classes by ignoring the background indices
        predicted_classes = torch.index_select(predicted_classes, dim=0, index=relation_indices)
        target_classes = torch.index_select(target_classes, dim=0, index=relation_indices)


    correct_predictions = torch.eq(predicted_classes, target_classes).sum().item()

    if len(target_classes) == 0:
        return 1
    accuracy = correct_predictions / len(target_classes)
    return accuracy


def evaluate(predicate_estimator, test_data_loader, ind_to_predicates, mode='predcls', return_loss=False, compute_mR=True):
    
    num_rel_category = 51
    result_dict = {}

    # prepare all inputs
    global_container = {}
    global_container['zeroshot_triplet'] = None
    global_container['result_dict'] = result_dict
    global_container['mode'] = mode
    global_container['multiple_preds'] = False
    global_container['num_rel_category'] = num_rel_category
    global_container['iou_thres'] = 0.5
    # We dont care about these two
    global_container['attribute_on'] = False
    global_container['num_attributes'] = 201

    
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode=mode)
    mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
    mean_recall.register_container(mode=mode)

    print(f'Total number of images for evaluation {len(test_data_loader)}')
    
    total_loss = 0.0
    accuracy_list = []
    # Check the num of predicates(51) for GT and predictions
    for data_dict in tqdm(test_data_loader):

        if data_dict['img_name'][0] == 'None':
            continue
        
        local_container = {}
        predicate_estimator.set_input(data_dict)

        with torch.no_grad():
        
            prediction = predicate_estimator.get_test_result(numpy=True, num_classes=args.num_predicates)
            if return_loss:
                predicate_logits = predicate_estimator.get_estimate()
                predicate_loss = predicate_estimator.get_loss(predicate_logits)
                total_loss += predicate_loss

            accuracy_list.append(compute_classification_accuracy(prediction, data_dict['gt_predicate_ids'][0]))

        if compute_mR:
            # values = data_dict['gt_predicate_ids'][0].numpy().astype(int) # (num_pred_entities(num_pred_entities-1),) 506
            # prediction = np.eye(51)[values]
            # prediction = torch.randint(low=0, high=49, size=prediction.shape, dtype=torch.int32).numpy()
            
            # values = data_dict['gt_predicate_ids'][0].numpy().astype(int)+1 # (num_pred_entities(num_pred_entities-1),) 506
            # prediction = np.eye(51)[values]

            local_container['gt_rels'] = data_dict['gt_rels'][0] # Tiplet ids in form (num_gt_rel, (sub_id-obj_id-pred_id))
            local_container['gt_boxes'] = data_dict['gt_boxes'][0] # (num_gt_entities, (x,y,x,y))
            local_container['gt_classes'] = data_dict['gt_classes'][0] # detected obj labels (num_entities,)
            
            local_container['obj_probs'] = data_dict['obj_probs'][0] #  # (num_pred_entities, 151)
            local_container['rel_pair_idx'] = data_dict['rel_pair_idx'][0] # (num_valid_relations, (sub_id-obj_id))
            local_container['rel_probs'] = prediction # (num_valid_relations, 51)
            local_container['pred_boxes'] = data_dict['pred_boxes'][0] # (num_pred_entities, (x,y,x,y))
            local_container['pred_classes'] = data_dict['pred_classes'][0] # (num_pred_entities,)
            
            if data_dict['img_name'][0] == '2326431.jpg':
                # print(data_dict['img_name'])
                # assert False
                continue
            # Keep creating local containers with the predictions
            local_container = eval_recall.calculate_recall(global_container, local_container, mode)

            mean_recall.collect_mean_recall_items(global_container, local_container, mode=mode)

    string = ''

    string += 'Accuracy: {:.4f}\n'.format(np.mean(accuracy_list))
    if compute_mR:
        mean_recall.calculate_mean_recall(mode = mode)
        
        string += eval_recall.generate_print_string(mode = mode)
        string += mean_recall.generate_print_string(mode = mode)

    print(string)

    if return_loss:
        return total_loss / len(test_data_loader), string
    return string

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="Set hyperparameters for the model")


    parser = argparse.ArgumentParser(description="Set hyperparameters for the model")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--which_epoch', type=int, default=0, help='Epoch number for loading the prompter model checkpoint')
    parser.add_argument('--which_epoch_cls', type=int, default=0, help='Epoch number for loading the classifier model checkpoint')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA if available')

    # Model parameters
    parser.add_argument('--n_context_vectors', type=int, default=8, help='Number of context vectors')
    parser.add_argument('--token_position', type=str, default='front', help='Position of the learnable context token')
    parser.add_argument('--learnable_UVTransE',type=bool, default=False, help='Load pretrained learnable layers for UVTransE')
    parser.add_argument('--randomize_UVTransE',type=bool, default=False, help='randomly initialize UVTransE')
    parser.add_argument('--is_halu_prompting',type=bool, default=False, help='Use halucinator for prompting')
    parser.add_argument('--num_predicates', type=int, default=50, help='Number of predicates of interest')
    parser.add_argument('--is_non_linear',type=bool, default=False, help='Non linearity in the classifier')

    # Input/output and logging parameters
    parser.add_argument('--checkpoints_dir_prompt', type=str, default='checkpoints', help='Directory to the prompter and UVTransE')
    parser.add_argument('--log_dir_cls', type=str, default='checkpoints', help='Directory to the classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory for input files')


    parser.add_argument('--is_TTT',type=bool, default=False, help='Use TTT model for getting the union features')
    # Parse the arguments
    args = parser.parse_args()
    
    if args.is_TTT:
        args.learnable_UVTransE = True

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Time and Date of the run: {current_time}\nArguments: {args}")

    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"


    with open('metrics/ind_to_predicates.pkl', "rb") as input_file:
        ind_to_predicates = pickle.load(input_file)

    ####################### Initialize data #######################
    test_dataset = VGDataset(split='test', case=1, data_root=args.data_dir, num_predicates=args.num_predicates)

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ####################### Initialize the models #######################

    predicate_estimator = PredicateEstimator(args, args.device, isTest=True)
    predicate_estimator.set_eval()
    string = evaluate(predicate_estimator, test_data_loader, ind_to_predicates, compute_mR=True)

    # print(string)
    # Appending to file
    with open(os.path.join(args.log_dir_cls, 'classifier_test_report.txt'), 'a') as file1:
        file1.write(f"Epoch-{args.which_epoch_cls} No-correction\n")
        file1.write(string)
        file1.write(f"\n**************************************************\n\n")


    # Open the text file
    with open(os.path.join(args.log_dir_cls, 'classifier_val_report.txt')) as f:
        lines = f.readlines()

    # Extract the accuracy values
    accuracies = []
    for line in lines:
        if "Accuracy: " in line:
            accuracies.append(float(line.split(": ")[1]))

    # Plot the accuracies
    plt.plot(accuracies)
    plt.title("Model Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(args.log_dir_cls, 'val_accuracy_epoch.png'))