import argparse
import importlib
import logging
import sys

from postprocess_path import set_save_path
from utils import Logger, pprint, set_gpu, set_logging, set_seed


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and network
    parser.add_argument('-project', type=str, default='base', choices=['teen', 'mine'])
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'in1k_fscil'])
    parser.add_argument('-dataroot', type=str, default='')
    parser.add_argument('-class_label', type=str, default='')  # Category label CSV file
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-feat_norm', action='store_true', help='If True, normalize the feature.')

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)

    ## optimizer & scheduler
    parser.add_argument('-optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('-schedule', type=str, default='Step', choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-tmax', type=int, default=600)  # Consine scheduler

    parser.add_argument('-not_data_init', action='store_true',
                        help='Whether to use average data embedding for initialization or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='Set to 0 to use all available training images for new classes')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot',
                                 'ft_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos',
                                 'avg_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=None, help='Loading model parameters from a specific directory')
    parser.add_argument('-only_do_incre', action='store_true', help='Load model and perform incremental learning...')

    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('-softmax_t', type=float, default=16)
    parser.add_argument('-shift_weight', type=float, default=0.5, help='Weights of delta prototypes')
    parser.add_argument('-soft_mode', type=str, default='soft_proto',
                        choices=['soft_proto', 'soft_proto_txt', 'no_calibration', 'same_seg', 'random_seg',
                                 'keyword_seg', 'keyword_seg_with_training'])
    parser.add_argument('-shot', type=int, default=1, help='Number of shots for each class')

    parser.add_argument('-topk_similarity', type=int, default=5)  # Top K similarity



    # ====keyword_seg====
    parser.add_argument('-topk_keyword_dim', type=int, default=50)  # Select top K keyword-influenced feature dimensions
    parser.add_argument('-use_keyword_weights',
                        action='store_true')  # Whether keyword affects the final weighting of the attribute calibration
    parser.add_argument('-keyword_threshold', type=float, default=0,
                        help='Threshold for keyword dimension selection')

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    set_save_path(args)

    logger = Logger(args, args.save_path)
    set_logging('INFO', args.save_path)
    logging.info(f"save_path: {args.save_path}")
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
