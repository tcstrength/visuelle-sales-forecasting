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
from datetime import datetime
from datetime import timedelta
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

        self.vtrends_df = self.__combine_series(self.train_df, self.test_df)
        self.vtrend_len = 14 # 28 weeks
        self.acceptance_date = self.vtrends_df.index.min() + timedelta(days=self.vtrend_len * 7)

        self.gtrends_df = pd.read_csv(gtrends_path, index_col="date", parse_dates=True)
        self.gtrend_len = 52 # 52 weeks

        self.cat_dict = torch.load(cat_path)
        self.cat_id_to_str = list(self.cat_dict.keys())
        self.col_dict = torch.load(color_path)
        self.col_id_to_str = list(self.col_dict.keys())
        self.fab_dict = torch.load(fabric_path)
        self.fab_id_to_str = list(self.fab_dict.keys())


    def __str_to_tensor(self, mapper: dict, values: list) -> torch.tensor:
        values = [mapper[x] for x in values]
        return torch.tensor(values)
    

    def __combine_series(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        category = self.__generate_daily_series("category", train_df, test_df)
        color = self.__generate_daily_series("color", train_df, test_df)
        fabric = self.__generate_daily_series("fabric", train_df, test_df)
        df1 = color.rename(columns={"color": "attr"})
        df2 = category.rename(columns={"category": "attr"})
        df3 = fabric.rename(columns={"fabric": "attr"})
        df = pd.concat([df1, df2, df3])
        df_pivot = df.pivot(index="date", columns=["attr"], values=["sales"])
        df_pivot.columns = df_pivot.columns.droplevel(0)
        df_pivot = df_pivot.fillna(0)
        df_pivot = df_pivot.sort_index()
        return df_pivot
    

    def __generate_daily_series(
            self, attr: str, train_df: pd.DataFrame, 
            test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the train and test dataset, then calculate total sales for each category,
        color and fabric overtime for each week
        """
        df = pd.concat([train_df, test_df])
        min_date = df["release_date"].min()
        max_date = df["release_date"].max() + timedelta(days=7 * 12)
        attr_ls = list(df[attr].unique())
        num_days = int((max_date - min_date).days)
        df_dates = pd.DataFrame(range(num_days), columns=["days"])
        df_dates[attr] = [attr_ls] * len(df_dates)
        df_dates["anchor"] = min_date
        df_dates["delta"] = pd.to_timedelta(df_dates["days"], unit="days")
        df_dates["date"] = df_dates["anchor"] + df_dates["delta"]
        df_dates = df_dates.drop(columns=["anchor", "delta", "days"])
        df_dates = df_dates.explode(column=attr)

        tmp = pd.DataFrame(range(7), columns=["day_of_week"])
        df = pd.merge(df, tmp, how="cross")
        df = df.melt(id_vars=[attr, "release_date", "day_of_week"], value_vars=df.columns[:12])
        df["delta"] = (df["variable"].astype(int) * 7) + df["day_of_week"]
        df["delta"] = pd.to_timedelta(df["delta"], unit="days")
        df["date"] = df["release_date"] + df["delta"]
        df["value"] = df["value"] / 7
        df = df.rename(columns={
            "value": "sales"
        })
        df = df.drop(columns=["variable", "release_date", "delta", "day_of_week"])
        df = pd.merge(df_dates, df, on=[attr, "date"], how="left")
        df = df.fillna({"sales": 0})
        df = df.groupby(by=[attr, "date"]).agg(
            sales=pd.NamedAgg(column="sales", aggfunc="sum")
        ).reset_index()
        return df
    

    def __extract_trends(
            self, trends_df: pd.DataFrame, trend_len: int, release_date: datetime, 
            cat: str, col: str, fab: str, trend_type: str="weekly", scale: bool=True
        ) -> np.ndarray:
        
        trend_start = release_date - pd.DateOffset(weeks=trend_len)
        trends_df = trends_df.loc[trend_start:release_date]

        if trend_type == "daily":
            trend_len = trend_len * 7

        cat_trend = trends_df[cat][-trend_len:].values[:trend_len]
        col_trend = trends_df[col][-trend_len:].values[:trend_len]
        fab_trend = trends_df[fab][-trend_len:].values[:trend_len]

        if scale is True:
            cat_trend = MinMaxScaler().fit_transform(cat_trend.reshape(-1,1)).flatten()
            col_trend = MinMaxScaler().fit_transform(col_trend.reshape(-1,1)).flatten()
            fab_trend = MinMaxScaler().fit_transform(fab_trend.reshape(-1,1)).flatten()

        trends = np.vstack([cat_trend, col_trend, fab_trend])
        return trends
    

    def __prepare_dataset(
            self, train: bool = True, sample_frac: float = None
        ) -> TensorDataset:
        
        if train is True:
            data = self.train_df.copy()
        else:
            data = self.test_df.copy()

        if sample_frac is not None:
            data = data.sample(frac=sample_frac)

        data = data[data["release_date"] >= self.acceptance_date]

        gtrends = []
        vtrends = []
        for (_, row) in tqdm.tqdm(data.iterrows(), total=len(data), ascii=True):
            release_date = row['release_date']
            cat, col, fab = row['category'], row['color'], row['fabric']

            # print(release_date)
            tmp = self.__extract_trends(
                self.gtrends_df, self.gtrend_len, release_date, cat, col, fab
            )
            gtrends.append(tmp)

            tmp = self.__extract_trends(
                self.vtrends_df, self.vtrend_len, release_date, cat, col, fab, 
                trend_type="daily"
            )
            vtrends.append(tmp)

        gtrends = torch.tensor(np.array(gtrends), dtype=torch.float32)
        vtrends = torch.tensor(np.array(vtrends), dtype=torch.float32)
        data = data.drop(columns=["external_code", "season", "release_date", "image_path"])
        temporals = torch.tensor(data.iloc[:, 13:17].values, dtype=torch.float32)
        targets = torch.tensor(data.iloc[:, :12].values, dtype=torch.float32)
        categories = self.__str_to_tensor(self.cat_dict, data.iloc[:].category)
        colors = self.__str_to_tensor(self.col_dict, data.iloc[:].color)
        fabrics = self.__str_to_tensor(self.fab_dict, data.iloc[:].fabric)

        return TensorDataset(
            targets, 
            torch.vstack([colors, fabrics, categories]).T, 
            temporals, 
            gtrends,
            vtrends
        )


    def get_data_loader(
            self, batch_size: int = 32, train: bool = True, 
            sample_frac: float = None) -> DataLoader:
        
        dataset = self.__prepare_dataset(train, sample_frac)
        ### Only enable shuffle on training
        loader = DataLoader(
            dataset, batch_size, shuffle=train, persistent_workers=True, num_workers=8
        )
        return loader

