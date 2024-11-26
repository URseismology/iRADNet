import numpy as np
from scipy.io import loadmat
from typing import TypedDict


EXAMPLE = "/home/wmeng/crisprf/data/Sp_RF_syn1.mat"


class RFData(TypedDict):
    tshift: np.ndarray  # (1, 1)
    rayP: np.ndarray    # (1, 38)
    t: np.ndarray       # (1, 1000), time dimension
    x: np.ndarray       # (200, 1000), sparse codes
    y: np.ndarray       # (38, 1000), signal


def retrieve_single_xy(path: str = EXAMPLE) -> RFData:
    key_translation = {
        'tshift': 'tshift',
        'rayP': 'rayP',
        'taus': 't',    # time dimension
        'Min_2': 'x',   # sparse codes
        'tx': 'y',      # signal
    }
    data: RFData = loadmat(path)
    
    return {
        key_translation[k]: v
        for k, v in data.items()
        if k in key_translation
    }


def peek(**kwargs):
    return {
        k: v.shape
        for k, v in kwargs.items()
        if type(v) is np.ndarray
    }