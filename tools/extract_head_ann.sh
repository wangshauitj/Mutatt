# CUDA_VISIBLE_DEVICES=$1 python tools/extract_mrcn_head_feats.py --dataset $2 --splitBy $3
CUDA_VISIBLE_DEVICES=$1 python tools/extract_mrcn_ann_feats.py --dataset $2 --splitBy $3