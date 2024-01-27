# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22

    Description: Load VISUELLE data with an addition of Google Trends
"""

import torch
import torchvision
import pytorch_lightning as L
import transformers
from torch.nn import functional as F
from fairseq.optim.adafactor import Adafactor


### (n, embedding_dim)
class ImageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim: int, dropout: float=0.2):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)
        self.fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        emb = self.resnet(images)
        return self.fc(emb)
    

### (n,embedding_dim)
class TextEncoder(torch.nn.Module):
    def __init__(self, embedding_dim: int, attrs_dict: dict, device: str):
        super().__init__()
        self.bert_encoder = transformers.pipeline('feature-extraction', model='bert-base-uncased')
        self.fc = torch.nn.Linear(768, embedding_dim)
        self.attrs_dict = attrs_dict
        self.dropout = torch.nn.Dropout(0.1)
        self.device = device
    
    def forward(self, attrs: torch.tensor):
        texts = []
        for x in range(len(attrs)):
            col_id = attrs[x][0]
            fab_id = attrs[x][1]
            cat_id = attrs[x][2]
            col = self.attrs_dict[0][col_id]
            fab = self.attrs_dict[1][fab_id]
            cat = self.attrs_dict[2][cat_id]
            text = f"{col} {fab} {cat}"
            texts.append(text)
        
        embs = self.bert_encoder(texts)
        embs = [torch.tensor(x[0][1:-1]).mean(axis=0) for x in embs]
        embs = torch.stack(embs).to(self.device)
        return self.fc(embs)

### (n, hidden_dim)
class FusionNetwork(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_fusion: int):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_dim * num_fusion),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(embedding_dim * num_fusion, hidden_dim)
        )
    
    def forward(self, image_encodings, text_encodings):
        fusioned_tensor = torch.cat([image_encodings, text_encodings], dim=1)
        return self.fc(fusioned_tensor)


# class TimeEmbedder(torch.nn.Module):
#     def __init__(self, time_dim: int, embedding_dim: int):
#         super().__init__()
#         self.fc = torch.nn.Linear(time_dim, embedding_dim)
#         self.dropout = torch.nn.Dropout(0.1)

#     def forward(self, vector: torch.tensor):
#         return self.dropout(self.fc(vector))
    

class VisForecastNet(L.LightningModule):
    def __init__(
            self, hidden_dim: int, output_dim: int, embedding_dim: int, dropout: float, 
            attrs_dict: list, device: str
        ):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim, dropout=dropout)
        self.text_encoder = TextEncoder(embedding_dim, attrs_dict, device)
        self.fusion_network = FusionNetwork(embedding_dim, hidden_dim, 2) # 2 types of feature
        # self.time_embedder = TimeEmbedder(time_dim=4, embedding_dim=1)
        self.forecast_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

        self.val_outputs = []
        self.val_targets = []


    def forward(self, attrs, temporals, images):
        image_encodings = self.image_encoder(images)
        text_encodings = self.text_encoder(attrs)
        fusion_feature = self.fusion_network(image_encodings, text_encodings)
        return self.forecast_fc(fusion_feature)

        # text_embs = self.text_embedder(bert_embs)
        # time_embs = self.time_embedder(temporals)
        # image_embs = self.image_embedder(images)
        # logger.info("Text embeddings: ", text_embs.shape)
        # logger.info("Time embeddings: ", time_embs.shape)
        # logger.info("Image embeddings: ", image_embs.shape)
        # fusion_embs = torch.cat([
        #     image_embs, text_embs, image_embs
        # ], dim=1)

        # logger.info("Fusion embeddings: ", fusion_embs.shape)
        

    def training_step(self, batch, batch_idx):
        targets, bert_embs, temporals, images = batch
        predicts = self.forward(bert_embs, temporals, images)
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