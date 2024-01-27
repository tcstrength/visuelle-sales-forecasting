#$!/bin/bash

python3 -m venv .venv

source ./.venv/bin/activate \
    && pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn \
    scipy wandb pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    pytorch-lightning transformers fairseq tqdm tensorboardX \
    && pip cache list && pip cache purge