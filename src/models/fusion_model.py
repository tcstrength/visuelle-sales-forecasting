# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22
"""

import torch
import pytorch_lightning as L
import mlflow
from torch.nn import functional as F
from fairseq.optim.adafactor import Adafactor
from utils import misc_utils


class TemporalEncoder(torch.nn.Module):

    def __init__(self, temporal_len: int, hidden_dim: int):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(temporal_len, hidden_dim),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU()
        )

    
    def forward(self, temporals):
        return self.fc(temporals)
    

class TrendsEncoder(torch.nn.Module):
    def __init__(self, num_trends: int, trend_len: int, hidden_dim: int):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_trends * trend_len, hidden_dim),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU()
        )

    def forward(self, trends):
        return self.fc(trends)
    

class FusionNetwork(torch.nn.Module):
    def __init__(self, num_feature: int, hidden_dim: int, fushion_dim: int):
        super().__init__()
        input_dim = num_feature * hidden_dim
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, input_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(input_dim, fushion_dim)
        )

    def forward(self, temporal_encodings, gtrends_encodings, vtrends_encodings):
        concat = torch.concat([temporal_encodings, gtrends_encodings, vtrends_encodings], dim=1)
        return self.fc(concat)


class FusionModel(L.LightningModule):
    def __init__(
            self, num_trends: int, vtrend_len: int, gtrend_len: int, temporal_len: int, 
            hidden_dim: int, fusion_dim: int, output_dim: int, sales_scale: float
        ):

        super().__init__()
        self.sales_scale = sales_scale

        self.temporal_encoder = TemporalEncoder(temporal_len, hidden_dim)
        self.gtrends_encoder = TrendsEncoder(num_trends, gtrend_len, hidden_dim)
        self.vtrends_encoder = TrendsEncoder(num_trends, vtrend_len, hidden_dim)
        self.step = 0
        ### Features: temporal + gtrend + vtrend
        self.fusion_network = FusionNetwork(3, hidden_dim, fusion_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_dim, output_dim)
        )
        self.val_outputs = []
        self.val_targets = []


    def forward(self, temporals, gtrends, vtrends):
        params = (
            self.vtrends_encoder(vtrends),
            self.gtrends_encoder(gtrends),
            self.temporal_encoder(temporals)
        )
        features = self.fusion_network(*params)
        predictions = self.decoder(features)
        return predictions


    def training_step(self, batch, batch_idx):
        targets, _, bert_encs, temporals, gtrends = batch
        predicts = self.forward(bert_encs, temporals, gtrends)
        loss = F.mse_loss(predicts, targets)
        mlflow.log_metrics({
            "train_loss": loss.detach().cpu().numpy().item()
        })
        return loss
    

    def validation_step(self, batch, batch_idx):
        targets, _, bert_encs, temporals, gtrends = batch
        predicts = self.forward(bert_encs, temporals, gtrends)
        loss = F.mse_loss(predicts, targets)
        self.val_outputs.extend(predicts)
        self.val_targets.extend(targets)
        return loss
    

    def on_validation_epoch_end(self) -> None:
        predicts = torch.stack(self.val_outputs)
        targets = torch.stack(self.val_targets)
        loss = F.mse_loss(predicts, targets)

        wape = misc_utils.cal_wape(targets, predicts)
        mae = misc_utils.cal_mae(targets, predicts, self.sales_scale)
        mlflow.log_metrics({
            "val_loss": loss.detach().cpu().numpy().item(),
            "val_mae": mae,
            "val_wape": wape
        }, step=self.step)
        self.step = self.step + 1
        self.val_outputs.clear()
        self.val_targets.clear()
    

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(),scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]