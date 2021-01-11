import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from eofs.xarray import Eof
from geoglows import streamflow
from sklearn import metrics
from sklearn.model_selection import train_test_split


def reof(stack: xr.DataArray, variance_threshold: float = 0.727, n_modes: int = 4) -> xr.Dataset:
    """Function to perform rotated empirical othogonal function (eof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        variance_threshold(float, optional): optional fall back value to select number of eof
            modes to use. Only used if n_modes is less than 1. default = 0.727
        n_modes (int, optional): number of eof modes to use. default = 4

    returns:
        xr.Dataset: rotated eof dataset with spatial modes, temporal modes, and mean values
            as variables

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
    non_masked_idx = np.where(np.logical_not(np.isnan(eof_components[:,0])))[0]

    # create a "blank" array to set roated values to
    rotated = eof_components.values.copy()

    # # waiting for release of sklean version >= 0.24
    # # until then have a placeholder function to do the rotation
    # fa = FactorAnalysis(n_components=n_modes, rotation="varimax")
    # rotated[non_masked_idx,:] = fa.fit_transform(eof_components[non_masked_idx,:])

    # apply varimax rotation to eof components
    # placeholder function until sklearn version >= 0.24
    rotated[non_masked_idx,:] = _ortho_rotation(eof_components[non_masked_idx,:])

    # project the original time series data on the rotated eofs
    projected_pcs = np.dot(centered[:,non_masked_idx], rotated[non_masked_idx,:])

    # reshape the rotated eofs to a 3d array of [y,x,c]
    spatial_rotated = rotated.reshape(spatial_shape+(n_modes,))

    # structure the spatial and temporal reof components in a Dataset
    reof_ds = xr.Dataset(
        {
            "spatial_modes": (["lat","lon","mode"],spatial_rotated),
            "temporal_modes":(["time","mode"],projected_pcs),
            "center": (["lat","lon"],center.values.reshape(spatial_shape))
        },
        coords = {
            "lon":(["lon"],stack.lon),
            "lat":(["lat"],stack.lat),
            "time":stack.time,
            "mode": np.arange(n_modes)+1
        }
    )

    return reof_ds


def _ortho_rotation(components: np.array, method: str = 'varimax', tol: float = 1e-6, max_iter: int = 100) -> np.array:
    """Return rotated components. Temp function"""
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


def get_streamflow(lat: float, lon: float) -> xr.DataArray:
    """Function to get histroical streamflow data from the GeoGLOWS server
    based on geographic coordinates

    args:
        lat (float): latitude value where to get streamflow data
        lon (float): longitude value where to get streamflow data

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    # ??? pass lat lon or do it by basin ???
    reach = streamflow.latlon_to_reach(lat,lon)
    # send request for the streamflow data
    q = streamflow.historic_simulation(reach['reach_id'])

    # rename column name to something not as verbose as 'streamflow_m^3/s'
    q.columns = ["discharge"]

    # rename index and drop the timezone value
    q.index.name = "time"
    q.index = q.index.tz_localize(None)

    # return the series as a xr.DataArray
    return q.discharge.to_xarray()


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

def find_fits(reof_ds: xr.Dataset, q_df: xr.DataArray, stack: xr.DataArray, train_size: float = 0.7, random_state: int = 0):
    """Function to fit multiple polynomial curves on different temporal modes and test results

    """

    X = q_df
    y = reof_ds.temporal_modes

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    spatial_test = stack.where(stack.time.isin(y_test.time),drop=True)

    shape3d = spatial_test.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    spatial_test_flat = xr.DataArray(
        spatial_test.values.reshape(shape2d),
        coords = [np.arange(shape2d[1]),spatial_test.time],
        dims=['space','time']
    )

    non_masked_idx= np.where(np.logical_not(np.isnan(spatial_test_flat[0,:])))[0]

    modes = reof_ds.mode.values

    fit_dict = dict()
    dict_keys = ['fit_r2','pred_correlation','pred_rmse']

    for mode in modes:

        y_train_mode = y_train.sel(mode=mode)
        y_test_mode = y_test.sel(mode=mode)

        for order in range(1,4):

            # apply polynomial fitting
            c = np.polyfit(X_train,y_train_mode,deg=order)
            f = np.poly1d(c)

            y_pred = f(X_test)

            synth_test = synthesize(reof_ds,X_test,f,mode=mode)

            synth_test_flat = xr.DataArray(
                synth_test.values.reshape(shape2d),
                coords = [np.arange(shape2d[1]),synth_test.time],
                dims=['space','time']
            )

            # calculate statistics
            # calculate the stats of fitting on a test subsample
            temporal_r2 = metrics.r2_score(y_pred,y_test_mode)
            temporal_r2 = -999 if temporal_r2 < 0 else np.sqrt(temporal_r2)

            # check the synthesis stats comapared to observed data
            space_correlation = metrics.r2_score(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
                multioutput="variance_weighted"
            )
            space_correlation = -999 if space_correlation < 0 else np.sqrt(space_correlation)

            space_rmse = metrics.mean_squared_error(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
                multioutput="variance_weighted",
                squared=False
            )

            # pack the resulting statistics in dictionary for the loop
            stats = [temporal_r2,space_correlation,space_rmse]
            loop_dict = {f"mode{mode}_order{order}_{k}":stats[i] for i,k in enumerate(dict_keys)}
            # merge the loop dictionary with the larger one
            fit_dict = {**fit_dict,**loop_dict}

    return fit_dict


def synthesize(reof_ds: xr.Dataset, q_df: xr.DataArray, polynomial: np.poly1d, mode: int = 1) -> xr.DataArray:
    """Function to synthesize data from reof data and regression coefficients.
    This will also format the result as a geospatially aware xr.DataArray

    args:
        reof_ds (xr.Dataset):
        q_df (xr.DataArray):
        polynomial (np.poly1d):
        mode (int, optional):

    returns:
        xr.DataArray: resulting synthesized data based on reof temporal regression
            and spatial modes
    """

    y_vals = xr.apply_ufunc(polynomial,q_df)

    synth = y_vals * reof_ds.spatial_modes.sel(mode=mode) + reof_ds.center

    # drop the unneeded mode dimension
    # force sorting by time in case the array is not already
    synth = synth.astype(np.float32).drop("mode").sortby("time")

    return synth


def reofs_to_geotiff(reof_ds: xr.Dataset, path: str):
    """Helper funtion to convert xr.Dataset spatial modes from REOF analysis to geotiff

    args:
        reof_ds (xr.Dataset):
        path (str):
    """

    reof_rio = reof_ds.rio.write_crs("epsg:4326").rename({"lon":"x","lat":"y"})

    reof_rio.spatial_modes.transpose('mode', 'y', 'x').rio.to_raster(path)

    return
