# ocr-pytorch

Implementation of Object-Contextual Representations for Semantic Segmentation (https://arxiv.org/abs/1909.11065) in PyTorch

## Usage

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=8890 train.py --batch 4 [ADE20K PATH]
