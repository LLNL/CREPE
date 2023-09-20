# CREPE: CLIP Representation Enhanced Predicate Estimation

![](https://img.shields.io/badge/pytorch-green)

This repository hosts the official PyTorch implementation for: "CREPE: Learnable Prompting With CLIP Improves Visual Relationship Prediction"

## Abstract

This work explores the challenges of leveraging large-scale vision language models, such as CLIP, for visual relationship prediction (VRP), a task vital in understanding the relations between objects in a scene based on both image features and text descriptors. Despite its potential, we find that CLIP's language priors are restrictive in effectively differentiating between various predicates for VRP. Towards this, we present CREPE (CLIP Representation Enhanced Predicate Estimation), which utilizes learnable prompts and a unique contrastive training strategy to derive reliable CLIP representations suited for VRP. CREPE can be seamlessly integrated into any VRP method. Our evaluations on the Visual Genome benchmark illustrate that using representations from CREPE significantly enhances the performance of vanilla VRP methods, such as UVTransE and VCTree, even without additional calibration techniques, showcasing its efficacy as a powerful solution to VRP. CREPE's performance on the Unrel benchmark reveals strong generalization to diverse and previously unseen predicate occurrences, despite lacking explicit training on such examples. 
## Installation
The project requires Python 3.8 or later and makes use of PyTorch for training the models.

1. Clone the repository:
```bash
git clone https://github.com/LLNL/CREPE
cd CREPE
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Dataset
We use the Visual Genome benchmark dataset for this project. You can download it from the [official website](http://visualgenome.org/). After downloading, place the dataset in the `./datasets` directory.

## Training the Models

### Training the Prompter
The `train_prompter.py` script is used to train the prompter model. You can start the training process using the following command:

```bash
python train_prompter.py --n_contex_vectors 8 --token_position 'middle' --num_predicates 50 \
        --epochs 500 --learning_rate 2e-3 --batch_size 256 \
        --use_cuda True --out_dir './output' --data_dir 'datasets/pred_dicts_train_cmr'
```

You can also specify other command line arguments as per your requirements.

### Training the Classifier
After training the prompter, the obtained features are used for training the classifier. You can train the classifier using the `train_classifier.py` script as follows:

```bash

python legacy_train_classifier.py --batch_size=1 --learning_rate 0.001 --which_epoch=500 --train_epochs 100 --save_freq 1 --use_cuda True \
        --n_context_vectors=8 --token_position middle --learnable_UVTransE True --update_UVTransE True --is_non_linear True --num_predicates=50 \
        --checkpoints_dir_prompt=output/2023-05-09_19-18-07/checkpoints --out_dir=output/2023-05-09_19-18-07 \
        --data_dir='datasets/VG/np_files'
```

Just like the prompter, you can specify other command line arguments as per your requirements.

## Acknowledgements

We thanks the authors of the UVTransE and CLIP papers for their inspiring and foundational work. Our sincere thanks also goes to the authors of the [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) repository helped us in deriving the FRCNN features and metrics implementation.

## Contact
For any queries, feel free to reach out at `rakshith.2905@gmail.com`.


---
