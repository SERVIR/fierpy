import os
import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from collections import namedtuple

from geoglows import streamflow
from sklearn import metrics
from sklearn.model_selection import train_test_split
import HydroErr as he

from typing import Union, Iterable, Tuple

from . import reof_ops

HydroModes = namedtuple("HydroModes", ("mode", "reach", "metric",))


# ----- Conventional EOF (unrotated) -----
def eof(stack: xr.DataArray, n_modes: Union[int, None] = None, n_iter:int = 10) -> xr.Dataset:
    """Function to perform unrotated empirical othogonal function (eof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        n_modes (int, optional): number of eof modes to use. default = -1

    returns:
        xr.Dataset: eof dataset with spatial modes, temporal modes, and mean values
            as variables
            
        list: explained variance (%)

    """
    if n_modes is None:
        n_modes = reof_ops.montecarlo_significance(stack, n_iter=n_iter)

    eof_out = reof_ops.eof(stack, n_modes=n_modes, center_input=True)
        
    # structure the spatial and temporal eof components in a Dataset
    eof_ds = reof_ops.eof_to_ds(stack, eof_out)

    return eof_ds

def reof(stack: xr.DataArray, n_modes: Union[int, None] = None, n_iter:int = 10) -> xr.Dataset:
    """Function to perform rotated empirical othogonal function (reof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        variance_threshold(float, optional): optional fall back value to select number of eof
            modes to use. Only used if n_modes is less than 1. default = 0.727
        n_modes (int, optional): number of eof modes to use. default = 4

    returns:
        xr.Dataset: rotated eof dataset with spatial modes, temporal modes, and mean values
            as variables

    """
    if n_modes is None:
        n_modes = reof_ops.montecarlo_significance(stack, n_iter=n_iter)

    reof_out = reof_ops.reof(stack, n_modes=n_modes, center_input=True)
        
    # structure the spatial and temporal eof components in a Dataset
    reof_ds = reof_ops.eof_to_ds(stack, reof_out)

    return reof_ds


def find_hydro_mode(eof_ds: xr.Dataset, hydro_da: xr.DataArray, metric: str='pearson_r', threshold: float=0.6, deoutlier: bool = False) -> HydroModes:   
    """
       Calculate correlation between temporal patterns and hydrological data.
       This helps determine water-related mode. By default,  >=0.6 is considered to be correlated.      

       args:
       1. eof_stack: Dataset with EOF or REOF results
       2. hydro_stack: DataArray with hydrological data (now GEOGloWS streamflow)
       3. r_type: 0: Pearson (default); 1: Spearman; 2: Nash-Sutcliffe
       4. r_threshold: Threshold (default: 0.6) of correlation coefficient to decide which modes are water-related
       5. deoutlier: Apply outlier removal (or not) to the hydrological data before calculating correlation
       
       output (site X modes):
       1. site_out: index for the in-situ data list
       2. mode_out: index for the eof/reof mode list
       3. r_out: corresponding r
       
    """   

    metrics_lookup = {
        "pearson_r":  he.pearson_r, 
        "spearman_r": he.spearman_r, 
        "nse":        he.nse, 
        "kge":        he.kge_2012
    }
    available_metrics = list(metrics_lookup.keys())
    if metric not in available_metrics:
        raise NotImplementedError(f"provided metric name, {metric}, is not implemented. Please provide one of the following {available_metrics}")
    else:
        metric_func = metrics_lookup[metric]

    # get number of mode     
    n_mode = eof_ds.sizes['mode']    
    # get number of hydrological data sites
    n_reach = hydro_da.sizes['reachid']    
        
    r = np.zeros((n_reach, n_mode))
    # p = np.zeros((n_reach, n_mode))    
    
    # ----- Instead of using FOR, see if it is possible to vectorize the process -----
    for ct_mode in range(n_mode):
        # get mode of tpc
        tpc = eof_ds.temporal_modes.isel(mode=ct_mode)    
      
        for ct_site in range(n_reach):
            # get hydrological data coincident with satellite images (TPC) 
            hydro_single = hydro_da.isel(reachid=ct_site).dropna(dim='time')
                                
            if deoutlier:
                hydro_zscore = stats.zscore(hydro_single)                
                indx_good_hydro = (np.abs(hydro_zscore) <= 3)
                hydro_site = hydro_single[indx_good_hydro]        
            else:
                hydro_site=hydro_single
        
            good_hydro = match_dates(hydro_site, tpc)      
            good_tpc = match_dates(tpc, good_hydro)
        
            # if streamflow is not constant all the time
            if not np.all(good_hydro==good_hydro[0]):    
                # calculate monotonic correlation between hydrological data and tpc
                # as a reference to judge their connection
                if metric in available_metrics[2:]:
                    good_hydro = stats.zscore(good_hydro)
                    good_tpc = stats.zscore(good_tpc)
                    r[ct_site, ct_mode] = metric_func(good_hydro, good_tpc)
                else:
                    r[ct_site, ct_mode] = np.abs(metric_func(good_tpc, good_hydro))
    
    indx_max_r_site = np.argmax(r, axis=0)
    r_temp = r[( indx_max_r_site, list(range(n_mode)) )]
    
    mode_out = (((r_temp >= threshold).astype(int)).nonzero())[0]
        
    site_out = indx_max_r_site[mode_out]
    r_out = r_temp[ (mode_out) ]
    
    return HydroModes(mode_out, site_out, r_out)

def combine_eof_hydro(eof_ds: xr.Dataset, hydro_da: xr.DataArray, hydro_modes: HydroModes) -> xr.Dataset:

    eof_sel = eof_ds.isel(mode=hydro_modes.mode)

    hydro_sel = match_dates(hydro_da.isel(reachid=hydro_modes.reach), eof_sel.temporal_modes)
    reaches = hydro_sel["reachid"]
    hydro_sel = hydro_sel.rename({"reachid":"mode"})
    hydro_sel["mode"] = eof_sel["mode"]

    eof_hydro = xr.merge([eof_sel,hydro_sel]).astype(np.float32)
    eof_hydro["reachid"] = (["mode"],reaches.data)
    eof_hydro["fit_metrics"] = (["mode"], hydro_modes.metric)

    n_sel_modes = eof_hydro.sizes["mode"]
    degrees = 3

    coef_mat = np.zeros([n_sel_modes, degrees])
    for i in range(n_sel_modes):
        mode_ds = eof_hydro.isel(mode = i)
        q = mode_ds["hydro_var"]
        tpc = mode_ds["temporal_modes"]


        p = np.polynomial.Polynomial.fit(q, tpc, 2)
        coef_mat[i,:] = p.convert().coef

    eof_hydro["degree"] = np.arange(degrees)
    eof_hydro["coefficients"] = (["mode","degree"],coef_mat.astype(np.float32))
    
    return eof_hydro

def _vec_poly1d(coeffs: xr.DataArray, vals: xr.DataArray) -> xr.DataArray:
    """Private function to apply vectorized polynomial on xarray dataarrays

    args:
        coeffs (Iterable[float]): array like sequence of polynomial coeffients
        vals (Iterable[float]): array kike values to apply polynomial function to
    """
    return np.polynomial.Polynomial(np.squeeze(coeffs))(vals)


def synthesize(reof_ds: xr.Dataset, q: Union[Iterable[float],xr.DataArray]) -> xr.DataArray:
    """Function to recontruct SAR imagery from streamflow and REOF analysis

    args:
        reof_ds (xr.Dataset)
        q (Iterable):

    returns:
        xr.DataArray

    """
    n_modes = reof_ds.mode.size

    coeff_da = reof_ds.coefficients

    if isinstance(q, Iterable) and not isinstance(q, xr.DataArray):
        q = xr.DataArray(q,dims=("time"))
    else:
        q = xr.DataArray(q)

    sim_pcs = xr.apply_ufunc(_vec_poly1d, coeff_da, q, vectorize="True",input_core_dims=[["degree"],[]])

    if "time" in sim_pcs.coords.keys():
        sim_pcs = sim_pcs.transpose("time","mode")

    reconstructed = sim_pcs.dot(reof_ds.spatial_modes) + reof_ds.center

    return reconstructed.astype(np.float32).rename("synthesized")

def correct_synthesis(stack, eof_ds, synthesis, apply_transform=False):

    stack_synth = synthesize(eof_ds, eof_ds["hydro_var"])

    # applies SAR dB to power transform for fitting distributions
    # should only be used if inputs are SAR data
    if apply_transform:
        stack = (10**(stack/10))
        stack_synth = (10**(stack_synth/10))
        synthesis = (10**(synthesis/10))

    stack_mean = stack.mean(dim="time")
    stack_std = stack.std(dim="time")

    stack_a = (stack_mean / stack_std) ** 2
    stack_b = (stack_std ** 2) / stack_mean

    stack_synth_mean = stack_synth.mean(dim="time")
    stack_synth_std = stack_synth.std(dim="time")

    stack_synth_a = (stack_synth_mean / stack_synth_std) ** 2
    stack_synth_b = (stack_synth_std ** 2) /  stack_synth_mean

    synth_percentile = stats.gamma.cdf(synthesis, a=stack_synth_a, scale=stack_synth_b)

    synth_corrected = synthesis.copy(deep=True)

    synth_corrected[:] = stats.gamma.ppf(synth_percentile, a=stack_a, scale=stack_b)

    # reverse SAR transform for power to dB
    if apply_transform:
        synth_corrected = 10 * np.log10(synth_corrected)

    return synth_corrected.astype(np.float32).rename("synthesized")


# ----- Streamflow-related function -----
def wrap_streamflow(lats: Iterable[float], lons: Iterable[float], forecast_opt:int) -> Tuple[xr.DataArray, list]:
    """Function to get and wrap up streamflow data from the GeoGLOWS server at different geographic coordinates 
    as one DataArray. Users now get to choose whether historical simulation, forecast stats, or forecast record 
    will be retrieved. For the forecast cases, output will be the daily average.
    
    May add bias correction in the future???

    args:
        lats (list): latitude values where to get streamflow data
        lons (list): longitude values where to get streamflow data
        forecast_opt (int): Option: 0: Historical simulation; 1: Forecast stats; 2: Forecast record

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    site_num = len(lats)
    reaches = []
    for ct_site in range(site_num):      
        q, reach_id = get_streamflow(lats[ct_site], lons[ct_site], forecast_opt)
        q["time"] = q["time"].dt.strftime("%Y-%m-%d")
        if ct_site==0:
            q_out = q.expand_dims(dim='site')
        else:       
            q_out = xr.concat( (q_out, q),dim='site' ) 
        reaches.append(reach_id)
      
    # return the series as a xr.DataArray
    return q_out, reaches


def get_streamflow(lat: float, lon: float, forecast_opt:int) -> Tuple[xr.DataArray, int]:
    """Function to get streamflow data from the GeoGLOWS server based on geographic coordinates.
    Users now get to choose whether historical simulation, forecast stats, or forecast record 
    will be retrieved. For the forecast cases, output will be the daily average.

    May add bias correction in the future???

    args:
        lat (float): latitude value where to get streamflow data
        lon (float): longitude value where to get streamflow data
        forecast_opt (int): Option: 0: Historical simulation; 1: Forecast stats; 2: Forecast record

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
        int: reach id
    """
    # ??? pass lat lon or do it by basin ???
    reach = streamflow.latlon_to_reach(lat,lon)
    # When I was using GEOGloWS over the Mississippi River, sometimes the output reachID is not neccessary 
    # the reach of the given lat and lon. May be due to the spatial resolution of the model???
    
    if forecast_opt == 0:
        # send request for the streamflow data
        q = streamflow.historic_simulation(reach['reach_id'])

        # rename column name to something not as verbose as 'streamflow_m^3/s'
        q.columns = ["discharge"]
        
        # rename index and drop the timezone value
        q.index.name = "time"
        q.index = q.index.tz_localize(None)

        # return the series as a xr.DataArray
        return q.discharge.to_xarray(), reach['reach_id']        
        
    else:
        # ----- Get the average of ensembles from GEOGloWS server and calculate their daily average -----
        # -- send request to get forecasted streamflow --
        # Forecast stats
        if forecast_opt == 1:
            init_q = streamflow.forecast_stats(reach['reach_id'])['flow_avg_m^3/s']
        # Forecast record
        elif forecast_opt == 2:
            init_q = streamflow.forecast_records(reach['reach_id'])['hydro_var']
        date_list = init_q.index.map(lambda t: t.date())
        uniq_date = date_list.unique()
        uniq_date.name = 'time'
        
        # Calculate the averages by dates
        q_temp = np.empty((len(uniq_date),))        
        for ct_uniq_date in range(len(uniq_date)):            
            q_temp[ct_uniq_date] = np.nanmean(init_q.to_numpy()[np.argwhere(date_list==uniq_date[ct_uniq_date])])

        q = xr.DataArray(
            data = q_temp,
            dims=['time'],
            coords=dict(
                time = pd.to_datetime(uniq_date)
            )                
        )
        q.name='discharge'        
        return q, reach['reach_id']


def match_dates(original: xr.DataArray, matching: xr.DataArray) -> xr.DataArray:
    """Helper function to filter a DataArray from that match the data values of another.
    Expects that each xarray object has a dimesion named 'time'

    args:
        original (xr.DataArray): original DataArray with time dimension to select from
        matching (xr.DataArray): DataArray with time dimension to compare against

    returns:
        xr.DataArray: DataArray with values that have been temporally matched
    """

    # return the DataArray with only rows that match dates
    return original.where(original.time.isin(matching.time),drop=True)


def fits_to_files(fit_dict: dict,out_dir: str) -> None:
    """Procedure to save coeffient arrays stored in dictionary output from `find_fits()` to npy files

    args:
        fit_dict (dict): output from function `find_fits()`
        out_dir (str): directory to save coeffients to
    """

    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for k,v in fit_dict.items():
        if k.endswith("coeffs"):
            components = k.split("_")
            name_stem = f"poly_{k.replace('_coeffs','')}"
            coeff_file = out_dir / f"{name_stem}.npy"
            np.save(str(coeff_file), v)

    return

def synthesize_indep(reof_ds: xr.Dataset, q_df: xr.DataArray, model_mode_order, model_path='.\\model_path\\'):
    """Function to synthesize data at time of interest and output as DataArray

    """
    mode_list=list(model_mode_order)
    for num_mode in mode_list:
        #for order in range(1,4):

        f = np.poly1d(np.load(model_path+'\poly'+'{num:0>2}'.format(num=str(num_mode))+'_deg'+'{num:0>2}'.format(num=model_mode_order[str(num_mode)])+'.npy'))

        y_vals = xr.apply_ufunc(f, q_df)

        synth = y_vals * reof_ds.spatial_modes.sel(mode=int(num_mode)) # + reof_ds.center

        synth = synth.astype(np.float32).drop("mode").sortby("time")

        return synth