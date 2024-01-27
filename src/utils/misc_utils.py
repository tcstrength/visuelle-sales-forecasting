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


def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)
    

def print_error_metrics(y_test, y_hat, rescaled_y_test, rescaled_y_hat):
    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, wape, rescaled_mae, rescaled_wape)