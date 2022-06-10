import xarray as xr
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path, PosixPath
from typing import Union, Iterable

def write_ds(ds, outfile, complevel=5):
    """
    Function that wraps `.to_netcdf()` but adds compression.
    If users want more options and control on the output other than compression, 
    then they should manually write the dataset using `.to_netcdf()`
    """
    # sepecify compression and create encoding dict based on variables
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in ds.data_vars}

    # TODO: add variable attributes to outputs

    ds.to_netcdf(outfile, encoding=encoding)

    return

def merged_q_tables(tbl_files: Iterable[pathlib.PosixPath]):
    reachids = []
    q_das = []
    for tbl in tbl_files:
        x = pd.read_csv(tbl,index_col="datetime")
        x.index = pd.to_datetime(x.index)
        x.index.name = "time"
        
        q_das.append(x.to_xarray())
        reachids.append(int(tbl.stem.split("_")[-1]))

    q_da = xr.concat(q_das,dim="reachid")
    q_da["reachid"] = reachids

    return q_da

def read_observations(path, datavar, timevar, yvar, xvar):
    datain = xr.open_dataset(path)[datavar]
    # check that the inputs have the correct name information for the fier process
    # TODO: add in automatic checks to handle different spatial info because not everything is lat,lon
    expectednames = ["time", "lat", "lon"]
    for i, name in enumerate([timevar, yvar, xvar]):
        if name != expectednames[i]:
            datain = datain.rename({name:expectednames[i]})

    return datain
        

def read_geoglows(path, rivids):
    ds = xr.open_dataset(path)

    ds = ds.sel(rivids=rivids)
    return 

def read_tables(tbls, datacol, timecol):
    reachids = []
    q_das = []
    for i,tbl in enumerate(tbls):
        # force the path to be a pathlib object for use
        tbl = Path(tbl)

        x = pd.read_csv(tbl,index_col=timecol)
        x.index = pd.to_datetime(x.index)
        x.index.name = "time"
        
        q_das.append(x.to_xarray()[datacol])

        # reachids.append(int(tbl.stem.split("_")[-1]))
        reachids.append(i)

    q_da = xr.concat(q_das,dim="reachid")
    q_da["reachid"] = reachids

    # rename variable to something used in fier process
    # set the original name as input data column to keep metadata
    q_da = q_da.rename("hydro_var")
    q_da.attrs = {"long_name": datacol}

    return q_da
