# fierpy

[![Python: 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SERVIR: Global](https://img.shields.io/badge/SERVIR-Global-green)](https://servirglobal.net)

Python implementation of the Forecasting Inundation Extents using REOF method

Based off of the methods from [Chang et al., 2020](https://doi.org/10.1016/j.rse.2020.111732)

## Installation

```bash
$ conda create -n fier -c conda-forge python=3.8 netcdf4 qt pyqt rioxarray numpy scipy xarray pandas scikit-learn eofs geoglows

$ conda activate fier

$ pip install git+https://github.com/servir/fierpy.git
```

To Install in OpenSARlab:

```bash
$ conda create --prefix /home/jovyan/.local/envs/fier python=3.8 netcdf4 qt pyqt rioxarray numpy scipy xarray pandas scikit-learn eofs geoglows jupyter kernda

$ conda activate fier

$ pip install git+https://github.com/servir/fierpy.git

$ /home/jovyan/.local/envs/fier/bin/python -m ipykernel install --user --name fier

$ conda run -n fier kernda /home/jovyan/.local/share/jupyter/kernels/fier/kernel.json --env-dir /home/jovyan/.local/envs/fier -o
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


## License and Distribution

fierpy is distributed by SERVIR under the terms of the MIT License. See
[LICENSE](https://github.com/SERVIR/fierpy/blob/master/LICENSE) in this directory for more information.

## Privacy & Terms of Use

fierpy abides to all of SERVIR's privacy and terms of use as described
at [https://servirglobal.net/Privacy-Terms-of-Use](https://servirglobal.net/Privacy-Terms-of-Use).