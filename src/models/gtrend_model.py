# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22
"""

import torch
import torchvision
import pytorch_lightning as L
import transformers
from torch.nn import functional as F
from fairseq.optim.adafactor import Adafactor
from models import TimeDistributed


class GTrendModel(L.LightningModule):
    def __init__(
            self, num_trends: int, trend_len: int, hidden_dim: int, 
            output_dim: int, dropout: float
        ):

        super().__init__()

        self.hidden_dim = hidden_dim
        # self.encoder = torch.nn.Sequential(
        #     TimeDistributed(torch.nn.Linear(trend_len, hidden_dim)),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Conv1d(in_channels=num_trends, out_channels=1, kernel_size=3, padding=1)
        # )

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_trends * trend_len, hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.val_outputs = []
        self.val_targets = []


    def forward(self, attrs, temporals, gtrends):
        encodings = self.encoder(gtrends)
        encodings = encodings.reshape(-1, self.hidden_dim)
        predictions = self.decoder(encodings)
        return predictions


    def training_step(self, batch, batch_idx):
        targets, bert_embs, temporals, gtrends = batch
        predicts = self.forward(bert_embs, temporals, gtrends)
        loss = F.mse_loss(predicts, targets)
        self.log("train_loss", loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        targets, bert_embs, temporals, images = batch
        predicts = self.forward(bert_embs, temporals, images)
        loss = F.mse_loss(predicts, targets)
        self.log("val_loss", loss)
        self.val_outputs.extend(predicts)
        self.val_targets.extend(targets)
        return loss
    

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(),scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]