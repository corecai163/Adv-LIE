export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node 2 --master_port 4320 train1.py
#CUDA_VISIBLE_DEVICES=0,1  python3 single_train.py
