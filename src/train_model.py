# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22

    Description: Load VISUELLE data with an addition of Google Trends

%load_ext autoreload
%autoreload 2
%cd ../

import sys
sys.path.append(f"{os.getcwd()}/src")
"""

import os
import pytorch_lightning as L
import warnings
import numpy as np
from utils import misc_utils
from models import VisuelleDataset
# from models import VisForecastNet
from models import GTrendModel

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    device = misc_utils.get_pytorch_device()

    visuelle = VisuelleDataset(data_dir="dataset/")
    train_loader = visuelle.get_data_loader(train=True, batch_size=64)
    test_loader = visuelle.get_data_loader(train=False, batch_size=64)

    # from models import VisForecastNet # Refresh Module
    # attrs_dict = [visuelle.col_id_to_str, visuelle.fab_id_to_str, visuelle.cat_id_to_str]
    # vfn = VisForecastNet(
    #     hidden_dim=64, output_dim=12, embedding_dim=32, attrs_dict=attrs_dict, dropout=0.2,
    #     device = device
    # )

    L.seed_everything(21)
    model = GTrendModel(
        num_trends=3, trend_len=52, hidden_dim=32, output_dim=12, dropout=0.2
    )
    targets, _, _, gtrends = train_loader.dataset[0:3]
    model.forward(None, None, gtrends)

    trainer = L.Trainer(
        accelerator=device, devices=1, max_epochs=40, log_every_n_steps=5, 
        check_val_every_n_epoch=5
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    targets, _, _, gtrends = test_loader.dataset[0:]
    predictions = model.forward(None, None, gtrends)
    y_test = targets.detach().numpy()
    y_hat = predictions.detach().numpy()
    
    misc_utils.print_error_metrics(y_test, y_hat, y_test * visuelle.target_scaler, y_hat * visuelle.target_scaler)
    

# All the strategies apply the same seed=21
# Strategy 1:
    # Use Only GTrends
    # Epoch: 50
    # Dropout: 0.1
    # TimeDistributed 3 times and convolution to merge 3 trends
    # 0.032 66.35 34.27 66.35
# Strategy 2:
    # Use Only GTrends
    # Epoch: 50
    # Dropout: 0.1
    # Use Torch.Concat 3 trends into 3
    # 0.031 64.763 33.45 64.763
# Strategy 3:
    # Use Only GTrends
    # Epoch: 50
    # Use Torch.Concat 3 trends into 3
    # Remove one Relu and swap the Dropout to before Relu
    # Dropout: 0.2
    # 0.033 67.428 34.826 67.428
# Strategy 4:
    # Use GTrends combine with Text Embedding
    # Apply Fusion Network

# from models import ImageEncoder
# from models import TextEncoder

# import torch
# from models import TimeDistributed

# layer = TimeDistributed(torch.nn.Linear(52, 32))

# t1 = train_loader.dataset[0][3].reshape(1, 3, 52)
# t2 = layer(t1)
# t3 = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, padding=1)(t2)
# t3.shape
# inputs = train_loader.dataset[:10]
# inputs[3].shape

# image_encodings = ImageEncoder(32)(inputs[3])
# text_encodings = TextEncoder(32, attrs_dict)(inputs[1])

# torch.cat([image_encodings, text_encodings], dim=1).shape

# test = torch.nn.BatchNorm1d(64)
# test(torch.cat([image_encodings, text_encodings], dim=1)).shape

# vfn(inputs[1], inputs[2], inputs[3]).shape