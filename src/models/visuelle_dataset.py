# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22

    Description: Load VISUELLE data with an addition of Google Trends
"""

import os
import torch
import tqdm
import pandas as pd
import numpy as np
from loguru import logger
# from PIL import Image
# from torchvision import transforms as T
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class VisuelleDataset():

    def __init__(self, data_dir: str):
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")
        gtrends_path = os.path.join(data_dir, "gtrends.csv")
        cat_path = os.path.join(data_dir, "category_labels.pt")
        color_path = os.path.join(data_dir, "color_labels.pt")
        fabric_path = os.path.join(data_dir, "fabric_labels.pt")
        scaler_path = os.path.join(data_dir, "normalization_scale.npy")
        logger.info(f"====== DATASET ======")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Test path: {test_path}")
        logger.info(f"Category path: {cat_path}")
        logger.info(f"Color path: {color_path}")
        logger.info(f"Fabric path: {fabric_path}")

        self.data_dir = data_dir
        self.target_scaler = np.load(scaler_path)
        self.image_root = os.path.join(data_dir, "images")
        self.train_df = pd.read_csv(train_path, parse_dates=["release_date"])
        self.test_df = pd.read_csv(test_path, parse_dates=["release_date"])
        self.gtrends_df = pd.read_csv(gtrends_path, index_col="date", parse_dates=True)
        self.trend_len = 52
        self.cat_dict = torch.load(cat_path)
        self.cat_id_to_str = list(self.cat_dict.keys())
        self.col_dict = torch.load(color_path)
        self.col_id_to_str = list(self.col_dict.keys())
        self.fab_dict = torch.load(fabric_path)
        self.fab_id_to_str = list(self.fab_dict.keys())


    def __str_to_tensor(self, mapper: dict, values: list) -> torch.tensor:
        values = [mapper[x] for x in values]
        return torch.tensor(values)


    def __prepare_dataset(self, train: bool = True, sample_frac: float = None) -> TensorDataset:
        
        if train is True:
            data = self.train_df.copy()
        else:
            data = self.test_df.copy()

        if sample_frac is not None:
            data = data.sample(frac=sample_frac)
        
        # ImageNet mean & std
        # image_compose = T.Compose([
        #     T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # image_features = []
        gtrends = []
        for (_, row) in tqdm.tqdm(data.iterrows(), total=len(data), ascii=True):
            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            release_date = row['release_date']
            cat, col, fab = row['category'], row['color'], row['fabric']

            gtrend_start = release_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends_df.loc[gtrend_start:release_date][cat][-52:].values[:self.trend_len]
            col_gtrend = self.gtrends_df.loc[gtrend_start:release_date][col][-52:].values[:self.trend_len]
            fab_gtrend = self.gtrends_df.loc[gtrend_start:release_date][fab][-52:].values[:self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])
            gtrends.append(multitrends)
            # img_path = row["image_path"]
            # img = Image.open(os.path.join(self.image_root, img_path)).convert("RGB")
            # image_features.append(image_compose(img))

        gtrends = torch.tensor(np.array(gtrends), dtype=torch.float32)
        data = data.drop(columns=["external_code", "season", "release_date", "image_path"])
        temporals = torch.tensor(data.iloc[:, 13:17].values, dtype=torch.float32)
        targets = torch.tensor(data.iloc[:, :12].values, dtype=torch.float32)
        categories = self.__str_to_tensor(self.cat_dict, data.iloc[:].category)
        colors = self.__str_to_tensor(self.col_dict, data.iloc[:].color)
        fabrics = self.__str_to_tensor(self.fab_dict, data.iloc[:].fabric)
        # images = torch.stack(image_features)

        return TensorDataset(
            targets, torch.vstack([colors, fabrics, categories]).T, temporals, gtrends
        )
    

    def get_target_scaler(self):
        return 


    def get_data_loader(
            self, batch_size: int = 32, train: bool = True, 
            sample_frac: float = None) -> DataLoader:
        
        dataset = self.__prepare_dataset(train, sample_frac)
        loader = DataLoader(dataset, batch_size, shuffle=True, persistent_workers=True, num_workers=4)
        return loader

