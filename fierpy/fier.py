import numpy as np
import xarray as xr
import pandas as pd
from eofs.xarray import Eof
from geoglows import streamflow


def reof(stack: xr.DataArray, variance_threshold: float = 0.727, n_modes: int = -1) -> xr.Dataset:
    """
    """
    # extract out some dimension shape information
    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    # flatten the data from [t,y,x] to [t,...]
    da_flat = xr.DataArray(
        stack.values.reshape(shape2d),
        coords = [stack.time,np.arange(shape2d[1])],
        dims=['time','space']
    )

    # find the temporal mean for each pixel
    center = da_flat.mean(dim='time')

    centered = da_flat - center

    # get an eof solver object
    # explicitly set center to false since data is already
    solver = Eof(centered,center=False)

    # check if the n_modes keyword is set to a realistic value
    # if not get n_modes based on variance explained
    if n_modes < 0:
        n_modes = int((solver.varianceFraction().cumsum() < variance_threshold).sum())

    # calculate to spatial eof values
    eof_components = solver.eofs(neofs=n_modes).transpose()
    # get the indices where the eof is valid data
    nonMissingIndex = np.where(np.logical_not(np.isnan(eof_components[:,0])))[0]

    # create a "blank" array to set roated values to
    rotated = eof_components.values.copy()

    # # waiting for release of sklean version >= 0.24
    # # until then have a placeholder function to do the rotation
    # fa = FactorAnalysis(n_components=n_modes, rotation="varimax")
    # rotated[nonMissingIndex,:] = fa.fit_transform(eof_components[nonMissingIndex,:])

    # apply varimax rotation to eof components
    # placeholder function until sklearn version >= 0.24
    rotated[nonMissingIndex,:] = _ortho_rotation(eof_components[nonMissingIndex,:])

    # project the original time series data on the rotated eofs
    projected_pcs = np.dot(centered[:,nonMissingIndex], rotated[nonMissingIndex,:])

    # reshape the rotated eofs to a 3d array of [y,x,t]
    spatial_rotated = rotated.reshape(spatial_shape+(n_modes,))

    # structure the spatial and temporal reof components in a Dataset
    reof_ds = xr.Dataset(
        {
            "spatial_modes": (["y","x","mode"],spatial_rotated),
            "temporal_modes":(["time","mode"],projected_pcs),
            "center": (["y","x"],center.values.reshape(spatial_shape))
        },
        coords = {
            "lon":(["x"],stack.lon),
            "lat":(["y"],stack.lat),
            "time":stack.time,
            "mode": np.arange(n_modes)+1
        }
    )

    return reof_ds


def _ortho_rotation(components: np.array, method: str = 'varimax', tol: float = 1e-6, max_iter: int = 100) -> np.array:
    """Return rotated components."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix)


def get_streamflow(lat: float, lon: float) -> pd.DataFrame:
    """
    """
    # ??? pass lat lon or do it by basin ???
    reach = streamflow.latlon_to_reach(lat,lon)
    # send request for the streamflow data
    q = streamflow.historic_simulation(reach['reach_id'])
    # reset column name to something not as verbose as 'streamflow_m^3/s'
    q.columns = ["discharge"]
    return q


def match_dates(df: pd.DataFrame, dates: xr.DataArray) -> pd.DataFrame:
    """
    """
    # loop through all of the dates and see where they match the df index dates
    individual_masks = np.array([df.index == t for t in pd.to_datetime(dates.values,utc=True)])

    # reduce the individual masks to a single mask
    idx = individual_masks.sum(axis=0).astype(np.bool)

    # return the df with only rows that match dates
    return df.loc[idx]


def find_fits(reof_ds,q_df):
    x = q_df.discharge

    y = reof_ds.temporal_modes.sel(mode=2)

    c = np.polyfit(x,y,2)

    f = np.poly1d(c)

    return
