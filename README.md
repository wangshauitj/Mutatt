# PyTorch Implementation of Mutatt
# MUTATT:VISUAL-TEXTUAL MUTUAL GUIDANCE FOR REFERRING EXPRESSION COMPREHENSION (ICME-2020)
Shuai Wang, Fan Lyu, Wei Feng, Song Wang

## Introduction

This repository is Pytorch implementation of [MUTATT:VISUAL-TEXTUAL MUTUAL GUIDANCE FOR REFERRING EXPRESSION COMPREHENSION](https://arxiv.org/pdf/.pdf) in [ICME 2020](http:///).
It is built on the top of the [MattNet](https://arxiv.org/pdf/1801.08186.pdf) in PyTorch.

Refering Expressions are natural language utterances that indicate particular objects within a scene, e.g., "the woman in red sweater", "the man on the right", etc.
For robots or other intelligent agents communicating with people in the world, the ability to accurately comprehend such expressions will be a necessary component for natural interactions.
In this project, we address referring expression comprehension: localizing an image region described by a natural language expression. 

## Prerequisites

* Python 2.7
* Pytorch 0.2 or higher
* CUDA 8.0

## Installation

1. Clone the Mutatt repository

```
git clone --recursive https://github.com/wangshauitj/Mutatt
```

2. Prepare the submodules and associated data

* Mask R-CNN: Follow the instructions of my [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.
You could use `cv/mrcn_detection.ipynb` to test if you've get Mask R-CNN ready.

* REFER API and data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.

* refer-parser2: Follow the instructions of [refer-parser2](https://github.com/lichengunc/refer-parser2) to extract the parsed expressions using [Vicente's R1-R7 attributes](http://tamaraberg.com/papers/referit.pdf). **Note** this sub-module is only used if you want to train the models by yourself.


## Training
1. Prepare the training and evaluation data by running `tools/prepro.py`:

```
python tools/prepro.py --dataset refcoco --splitBy unc
```

2. Extract features using Mask R-CNN, where the `head_feats` are used in subject module training and `ann_feats` is used in relationship module training.

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_head_feats.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_ann_feats.py --dataset refcoco --splitBy unc
```

3. Detect objects/masks and extract features (only needed if you want to evaluate the automatic comprehension). We empirically set the confidence threshold of Mask R-CNN as 0.65.

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect.py --dataset refcoco --splitBy unc --conf_thresh 0.65
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect_to_mask.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_det_feats.py --dataset refcoco --splitBy unc
```

4. Train MAttNet with ground-truth annotation:

```bash
./experiments/scripts/train_mattnet.sh GPU_ID refcoco unc
```
During training, you may want to use `cv/inpect_cv.ipynb` to check the training/validation curves and do cross validation.

## Evaluation

Evaluate MAttNet with ground-truth annotation:

```bash
./experiments/scripts/eval_easy.sh GPUID refcoco unc
```

If you detected/extracted the Mask R-CNN results already (step 3 above), now you can evaluate the automatic comprehension accuracy using Mask R-CNN detection and segmentation:

```bash
./experiments/scripts/eval_dets.sh GPU_ID refcoco unc
./experiments/scripts/eval_masks.sh GPU_ID refcoco unc
```



