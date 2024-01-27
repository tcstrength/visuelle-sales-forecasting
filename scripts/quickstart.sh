#!/bin/bash

source ./.venv/bin/activate && python3 train_POP.py \
    --wandb_entity thai-cuong1404 \
    --wandb_run cuong_dep_trai \
    --data_folder dataset/ \
    --epochs 100