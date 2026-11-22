"""Exact submitted-M1 RGB metric protocol (border 4, Gaussian SSIM)."""
import numpy as np
from skimage.metrics import structural_similarity


def score(pred,ref):
    pred=np.clip(pred.astype(np.float64),0,1)[4:-4,4:-4]
    ref=np.clip(ref.astype(np.float64),0,1)[4:-4,4:-4]
    mse=np.mean((pred-ref)**2)
    psnr=float(-10*np.log10(mse)) if mse else float('inf')
    ssim=float(structural_similarity(ref,pred,data_range=1.,channel_axis=2,
              gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False))
    return psnr,ssim
