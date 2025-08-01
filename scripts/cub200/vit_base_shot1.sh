python train.py \
    -project mine \
    -dataset cub200 \
    -dataroot data \
    -base_mode ft_cos \
    -new_mode avg_cos \
    -gamma 0.25 \
    -lr_base 0.004 \
    -lr_new 0.1 \
    -decay 0.0005 \
    -epochs_base 0 \
    -schedule Milestone \
    -milestones 50 100 150 200 250 300 \
    -gpu 4 \
    -temperature 32 \
    -batch_size_base 256 \
    -softmax_t 16 \
    -shift_weight 0.5 \
    -num_workers 4 \
    -shot 1 \
    -soft_mode no_calibration 
