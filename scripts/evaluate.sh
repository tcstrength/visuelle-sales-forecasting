source ./.venv/bin/activate && cd src && python forecast_POP.py \
    --wandb_entity thai-cuong1404 \
    --data_folder dataset/ \
    --ckpt_path ckpt/GTM_cuong_dep_trai---epoch=9---20-01-2024-22-35-48.ckpt \
    --epochs 3 