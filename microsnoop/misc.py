import argparse
import os


def get_embed_args_parser():
    parser = argparse.ArgumentParser('Microsnoop embedding', add_help=False)

    # 0. Train parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # 1. Model parameters
    parser.add_argument('--model_name', default='mae-vit-large-patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--in_chans', default=1, type=int,
                        help='images input chans')
    parser.add_argument('--embed_dim', default=256, type=int,
                        help='images embed dim')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images patch size')
    parser.add_argument('--diam_mean', default=30.0, type=float,
                        help='diam mean of cell instance')
    parser.add_argument('--checkpoint', default=None,
                        help='if give checkpoint, model will continue training from checkpoint')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more effic3ient (sometimes) transfer to GPU.')
    parser.add_argument('--print_freq', default=0, type=int)

    # 2. Dataset parameters
    parser.add_argument('--embed_dir', default=None, type=str,
                        help='path where to save, empty for no saving')

    # 3. distributed training parameters
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--nnode', default=1, type=int)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--node_rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    return parser


def check_chans(img_names, chan_names, task_name='Task'):
    img_names = [os.path.basename(img_name) for img_name in img_names]
    nchan = len(chan_names)
    pass_result_dict={True: 'True', False: 'False'}
    pass_flag = True
    for chani, chan_name in enumerate(chan_names):
        img_namesi = img_names[chani::nchan]
        check_flags = [chan_name.lower() in img_namei.lower() for img_namei in img_namesi]
        pass_flag *= all(check_flags)
    # print(task_name+':', pass_flag)
    return pass_result_dict[pass_flag]
