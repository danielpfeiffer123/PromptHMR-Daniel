import pickle
import joblib
import numpy as np
from pathlib import Path

def pkl_to_npz(pkl_path, npz_path=None, compress=True):
    """转换 pkl 文件到 npz 格式"""
    if npz_path is None:
        npz_path = Path(pkl_path).with_suffix('.npz')
    
    with open(pkl_path, 'rb') as f:
        data = joblib.load(f)
    
    target_data = data['people']

    for key in target_data:
        npz_path = Path(npz_path).with_name(f"{Path(npz_path).stem}_{key}.npz")

        if isinstance(target_data[key], dict):
            if compress:
                np.savez_compressed(npz_path, **target_data[key])
            else:
                np.savez(npz_path, **target_data[key])
        else:
            if compress:
                np.savez_compressed(npz_path, data=target_data[key])
            else:
                np.savez(npz_path, data=target_data[key])


    # print(data.keys())
    # if isinstance(data, dict):
    #     if compress:
    #         np.savez_compressed(npz_path, **data)
    #     else:
    #         np.savez(npz_path, **data)
    # else:
    #     if compress:
    #         np.savez_compressed(npz_path, data=data)
    #     else:
    #         np.savez(npz_path, data=data)
    
    return npz_path

# 使用示例
pkl_to_npz('results/kungfu/results.pkl', 'results/kungfu/results', compress=True)
# npz_to_pkl('results/kungfu/results.npz', 'results/kungfu/results.pkl')