python train.py \
    -project mine \
    -dataset cub200 \
    -dataroot data \
    -base_mode ft_cos \
    -new_mode avg_cos \
    -epochs_base 0 \
    -gpu 4 \
    -batch_size_base 256 \
    -temperature 95 \
    -softmax_t 28 \
    -shift_weight 0.5 \
    -topk_keyword_dim 50 \
    -topk_similarity 15 \
    -num_workers 4 \
    -shot 1 \
    -keyword_threshold 0 \
    -soft_mode keyword_seg \
    -class_label /home/zongyao/fscil/data/id2label_cub_v4.csv

    # -shift_weight 0.5 \
    # -topk_keyword_dim 50 \
    # -topk_similarity 15 \