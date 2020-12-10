# fierpy
Python implementation of the Forecasting Inundation Extents using REOF

## Installation

### Requirements
 * numpy
 * xarray
 * pandas
 * eofs
 * geoglows


## Example use

```python
import xarray as xr
import fierpy

ds = xr.open_dataset("sentine1.nc")

reof_ds = fierpy.reof(ds,n_modes=4)

lat,lon = 11.7122,104.9653

q = fierpy.get_streamflow(lat,lon)
q_sel = fierpy.match_dates(q,ds.time)

```
