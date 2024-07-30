import argparse
import torch

parser = argparse.ArgumentParser(description='scatch')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")

parser.add_argument('--mode', default='client')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default=65361)

args = parser.parse_args()

if args.local_rank == -1:
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()