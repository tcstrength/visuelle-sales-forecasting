# -*- coding: utf-8 -*-
"""
    Author: tcstrength
    Date: 2024-01-22
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error


def get_pytorch_device() -> str:
    if getattr(torch, "has_mps", False):
        return "mps"
    
    if torch.cuda.is_available():
        return "gpu"
    
    return "cpu"


def force_numpy(gt, forecasts):
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    
    if torch.is_tensor(forecasts):
        forecasts = forecasts.detach().cpu().numpy()
    
    return gt, forecasts

def cal_mae(gt, forecasts, sales_scale):
    gt, forecasts = force_numpy(gt, forecasts)
    gt = gt * sales_scale
    forecasts = forecasts * sales_scale
    mae = mean_absolute_error(gt, forecasts)
    return round(mae, 3)


def cal_wape(gt, forecasts):
    gt, forecasts = force_numpy(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)
    return round(wape, 3)
    