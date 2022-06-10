import numpy as np
import jax.numpy as jnp
import jax
import xarray as xr
from functools import partial
from collections import namedtuple
import numpy.typing as npt
from typing import Union

EofInputs = namedtuple("EofInputs", ("array", "shape3d", "shape2d", "valididx", "center",))
EofOutputs = namedtuple(
    "EofOutputs", ("spatial_modes", "temporal_modes", "variance", "center")
)


def _eof_prep(stack: Union[npt.ArrayLike, xr.DataArray], center_input: bool):
    """
    Private function to handle the preparation of arrays for the EOF process
    Users should not interface and it is called within other functions
    """

    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0], np.prod(spatial_shape))

    # reshape data from (time, y, x) to (time, n)
    if isinstance(stack, xr.DataArray):
        in_2d_array = stack.values.reshape(shape2d)
    elif isinstance(stack, np.ndarray):
        in_2d_array = stack.reshape(shape2d)
    else:
        raise TypeError(
            "Input array to EOF process must be of type xr.DataArray or np.ndarray"
        )

    if center_input:
        center_ = in_2d_array.mean(axis=0)
        centered = in_2d_array - center_
    else:
        center_ = np.zeros(shape2d[1])

    valid_idx = np.where(np.logical_not(np.isnan(centered[0])))[0]

    prepout = EofInputs(centered, shape3d, shape2d, valid_idx, center_)

    return prepout


@partial(jax.jit, static_argnums=(1,))
def _eof_process(*args):
    """
    Private function to apply EOF process in pre-processed inputs
    input arguments are a tuple of values that represent the EOF Inputs (see `_eof_prep()`)
    """
    in_array = args[0][:, args[3]]
    n_obs, n_pix = args[2]
    ddof = 1

    A, Lh, E = jnp.linalg.svd(in_array, full_matrices=False)

    normfactor = n_obs - ddof

    L = Lh * Lh / normfactor

    E_flat = args[0].copy() * np.nan
    # E_flat[:, args[3]] = E
    E_flat = E_flat.at[:, args[3]].set(E)

    P = A * Lh

    return (E_flat, P, L)


def eof(stack: Union[npt.ArrayLike, xr.DataArray], n_modes: Union[int, None] = None, center_input: bool = True) -> EofOutputs:
    inputs = _eof_prep(stack, center_input)

    E_flat, P, L = _eof_process(*inputs)

    if n_modes is not None:
        spatial_modes = E_flat[:n_modes, :].T
        temporal_modes = P[:, :n_modes]
        eof_var_frac = L[:n_modes].cumsum() / L[:n_modes].sum()

    outshape = inputs.shape3d[1:] + (n_modes,)

    spatial_modes = spatial_modes.reshape(outshape)
    center_out = inputs.center.reshape(inputs.shape3d[1:])

    return EofOutputs(spatial_modes, temporal_modes, eof_var_frac, center_out)


def reof(stack: Union[npt.ArrayLike, xr.DataArray], n_modes: Union[int, None] = None, center_input: bool = True) -> EofOutputs:
    inputs = _eof_prep(stack, center_input)

    E_flat, P, L = _eof_process(*inputs)

    if n_modes is not None:
        E_flat = E_flat[:n_modes, :].T
        P = P[:, :n_modes]

    # get total variance fractions of eof (up to the max retained mode)
    total_eof_var_frac = L[:n_modes].cumsum()

    # apply rotation on the valid observations
    rotation = orthorotation(E_flat[inputs.valididx, :])

    # create a "blank" array to set rotated values to
    rotated = np.array(E_flat)
    rotated[inputs.valididx, :] = rotation

    # project the original time series data on the rotated eofs
    projected_pcs = jnp.dot(
        inputs.array[:, inputs.valididx], rotated[inputs.valididx, :]
    )

    # get variance of each rotated mode
    rot_var = np.var(projected_pcs, axis=0)
    # get variance of all rotated modes
    total_rot_var = rot_var.cumsum()
    # get variance fraction of each rotated mode
    rot_var_frac = np.array((rot_var/total_rot_var)*total_eof_var_frac)

    outshape = inputs.shape3d[1:] + (n_modes,)

    center_out = inputs.center.reshape(inputs.shape3d[1:])

    # reshape the rotated eofs to a 3d array of [y,x,c]
    spatial_rotated = rotated.reshape(outshape)

    # sort modes based on variance fraction of REOF
    indx_rot_var_frac_sort = np.expand_dims(((np.argsort(-1 * rot_var_frac))), axis=0)
    projected_pcs = np.take_along_axis(projected_pcs, indx_rot_var_frac_sort, axis=1)

    indx_rot_var_frac_sort = np.expand_dims(indx_rot_var_frac_sort, axis=0)
    spatial_rotated = np.take_along_axis(
        spatial_rotated, indx_rot_var_frac_sort, axis=2
    )

    eof_var_frac = np.sort(rot_var_frac)[::-1].cumsum() / rot_var_frac.sum()

    return EofOutputs(spatial_rotated, projected_pcs, eof_var_frac, center_out)


def orthorotation(
    components: npt.ArrayLike,
    method: str = "varimax",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> npt.ArrayLike:
    """Return rotated components. Temp function"""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = jnp.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = jnp.linalg.svd(jnp.dot(components.T, comp_rot**3 - tmp))
        rotation_matrix = jnp.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return jnp.dot(components, rotation_matrix)


def montecarlo_significance(stack: Union[npt.ArrayLike, xr.DataArray], n_iter: int = 100, center: bool = True) -> int:
    """
    Significant test upon the EOF analysis results.
    """

    inputs = _eof_prep(stack, center)

    n_pix = inputs.shape2d[1]
    n_obs = inputs.shape2d[0]

    # Get eigenvalue from real data
    E_flat, P, L = _eof_process(*inputs)
    real_lamb = L

    sig_mode = 0

    rng = np.random.default_rng()

    mc_lamb = np.full((n_obs, n_iter), np.nan)

    obs_tmp = inputs.array.copy()

    for i in range(n_iter):
        np.random.seed(i)

        # ----- Some vectorization 1 -----
        obs_shuffle = rng.permuted(inputs.array[:, inputs.valididx], axis=1)
        obs_tmp[:, inputs.valididx] = obs_shuffle

        inputs_tmp = EofInputs(obs_tmp, *inputs[1:])

        E_flat_mc, P_mc, L_mc = _eof_process(*inputs_tmp)
        # print(solver_mc.eigenvalues(neigs = obs_num))
        mc_lamb[:, i] = L_mc

    mc_lamb = np.transpose(mc_lamb)
    mean_mc_lamb = np.mean(mc_lamb, axis=0)
    std_mc_lamb = np.std(mc_lamb, axis=0)

    gt = np.greater(real_lamb, mean_mc_lamb).astype(int)
    gt_rev = gt[::-1]
    sig_mode = len(gt_rev) - np.argmax(gt_rev) - 1

    return sig_mode


def eof_to_ds(stack: Union[npt.ArrayLike, xr.DataArray], eof_out: EofOutputs):
    # read the number of modes directly from the dataset
    n_modes = eof_out.spatial_modes.shape[-1]

    # structure the spatial and temporal eof components in a Dataset
    eof_ds = xr.Dataset(
        {
            "spatial_modes": (["lat", "lon", "mode"], eof_out.spatial_modes),
            "temporal_modes": (["time", "mode"], eof_out.temporal_modes),
            "center": (["lat", "lon"], eof_out.center),
            "variance": (["mode"], eof_out.variance),
        },
        coords={
            "lon": (["lon"], stack.lon.values),
            "lat": (["lat"], stack.lat.values),
            "time": stack.time.values,
            "mode": np.arange(n_modes) + 1,
        },
    )
    return eof_ds
