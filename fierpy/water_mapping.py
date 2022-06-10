import xarray as xr
import numpy as np

def simple_threshold(synthesized: xr.DataArray, threshold: float = -16):
    water = (synthesized <= -16).astype(np.float32)
    water = xr.where(np.isnan(synthesized), np.nan, water)
    return water.rename("water")