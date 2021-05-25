# Welcome to MkDocs

Python implementation of the Forecasting Inundation Extents using REOF method

Based off of the methods from [Chang et al., 2020](https://doi.org/10.1016/j.rse.2020.111732)

## Installation

```bash
$ conda create -n fier -c conda-forge python=3.8 numpy scipy xarray pandas sckit-learn eofs geoglows

$ pip install git:https://github.com/servir/fierpy.git
```

### Requirements
 * numpy
 * xarray
 * pandas
 * eofs
 * geoglows
 * scikit-learn
 * rasterio


## Example use

```python
import xarray as xr
import fierpy

# read sentinel1 time series imagery
ds = xr.open_dataset("sentine1.nc")

# apply rotated eof process
reof_ds = fierpy.reof(ds.VV,n_modes=4)

# get streamflow data from GeoGLOWS
# select the days we have observations
lat,lon = 11.7122,104.9653
q = fierpy.get_streamflow(lat,lon)
q_sel = fierpy.match_dates(q,ds.time)

# apply polynomial to different modes to find best stats
fit_test = fierpy.find_fits(reof_ds,q_sel,ds)
```