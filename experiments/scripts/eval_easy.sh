

GPU_ID=$1
DATASET=$2
SPLITBY=$3

# IMDB="coco_minus_refer"
# ITERS=1150000
# TAG="notime"
# NET="res101"
#ID="mrcn_cmr_with_st"
# ID="ablation_sub_11_loc_01"
ID="out_method_seed_0"

case ${DATASET} in
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
esac
