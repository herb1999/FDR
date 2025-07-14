python train.py \
    -project mine \
    -dataset cub200 \
    -dataroot data \
    -base_mode ft_cos \
    -new_mode avg_cos \
    -gamma 0.25 \
    -lr_base 0.01 \
    -lr_new 0.1 \
    -decay 0.0005 \
    -epochs_base 0 \
    -schedule Cosine \
    -tmax 100 \
    -gpu 4 \
    -batch_size_base 256 \
    -num_workers 4 \
    -shot 5 \
    -epochs_split 0 \
    -split_ratio 0.8 \
    -lambda_weight 20 \
    -keyword_threshold 0 \
    -soft_mode keyword_seg \
    -class_label /home/zongyao/fscil/data/id2label_cub_v4.csv \
    -temperature 36 \
    -softmax_t 22 \
    -shift_weight 0.2 \
    -topk_keyword_dim 37 \
    -topk_similarity 29

  # -freeze_keyword_attention \
    #   -temperature 50 \
    # -softmax_t 30 \
    # -shift_weight 0.25 \
    # -topk_keyword_dim 32 \
    # -topk_similarity 8 \