python train.py \
    -project mine \
    -dataset in1k_fscil \
    -dataroot /home/zongyao/Dataset/ImageNet1K/imagenet_fscil_benchmark_800base \
    -base_mode ft_cos \
    -new_mode avg_cos \
    -gamma 0.25 \
    -lr_base 0.004 \
    -lr_new 0.1 \
    -decay 0.0005 \
    -epochs_base 0 \
    -schedule Milestone \
    -milestones 50 100 150 200 250 300 \
    -gpu 2 \
    -batch_size_base 256 \
    -test_batch_size 256 \
    -num_workers 8 \
    -keyword_threshold 0 \
    -shot 5 \
    -soft_mode keyword_seg \
    -class_label /home/zongyao/fscil/data/id2label_in1k_fscil_v5.csv \
    -temperature 90 \
    -softmax_t 30 \
    -shift_weight 0.4 \
    -topk_keyword_dim 50 \
    -topk_similarity 30 \


    # -temperature 50 \
    # -softmax_t 48 \
    # -shift_weight 0.5 \
    # -topk_keyword_dim 32 \
    # -topk_similarity 16 \

    # -temperature 94.8 \
    # -softmax_t 28 \
    # -shift_weight 0.502 \
    # -topk_keyword_dim 45 \
    # -topk_similarity 14 \