import xarray as xr
import logging

from .fier import *
from .io import *
from .water_mapping import *

def build_regressions(config: dict):

    # extract out the information on reading observation
    obspath = config["observations"]["path"]
    datavar = config["observations"]["datavar"]
    timevar = config["observations"]["timevar"]
    yvar = config["observations"]["yvar"]
    xvar = config["observations"]["xvar"]

    configkeys = list(config.keys())

    if "geoglows" in configkeys:
        ingeoglows = config["geoglows"]["path"]
        rivids = config["geoglows"]["rivids"]
        datacol = "Q"
        timecol = "Date"
        hydro_da = read_geoglows(ingeoglows, rivids)

    elif "tables" in configkeys:
        # extract out the information on hydrologic info for regressions
        intables = config["tables"]["path"]
        datacol = config["tables"]["datacol"]
        timecol = config["tables"]["timecol"]
        hydro_da = read_tables(intables, timecol = timecol, datacol = datacol)

    else:
        raise RuntimeError("table inputs or geoglows inputs must be specified")
        return

    # extract out output location for regression
    outpath = config["regressions"]["path"]

    # extract processing options if any
    if "options" in configkeys:
        optkeys = list(config["options"].keys())

        nmodes = config["options"]["nmodes"] if "nmodes" in optkeys else None

        removeoutliers = config["options"]["removeoutliers"] if "removeoutliers" in optkeys else False

        nsims = config["options"]["nsimulations"] if "nsimulations" in optkeys else 10

        rotate = config["options"]["apply_rotation"] if "apply_rotation" in optkeys else True

    datain = read_observations(obspath, datavar, timevar, yvar ,xvar)


    if rotate:
        eof_ds = reof(datain, n_modes=nmodes, n_iter=nsims)
    else:
        eof_ds = eof(datain, n_modes=nmodes, n_iter=nsims)

    hydro_modes = find_hydro_mode(eof_ds, hydro_da, deoutlier=removeoutliers)

    eof_hydro = combine_eof_hydro(eof_ds, hydro_da, hydro_modes)

    write_ds(eof_hydro, outpath)

    return


def run_synthesize(config: dict):

    # extract out the information on reading observation
    fitpath = config["regressions"]["path"]
    timevar = config["observations"]["timevar"]
    yvar = config["observations"]["yvar"]
    xvar = config["observations"]["xvar"]

    configkeys = list(config.keys())

    # extract out output location for regression
    outpath = config["synthesis"]["path"]

    if "geoglows" in configkeys:
        ingeoglows = config["geoglows"]["path"]
        rivids = config["geoglows"]["rivids"]
        datacol = "Q"
        timecol = "Date"
        hydro_da = read_geoglows(ingeoglows, rivids)        

    elif "tables" in configkeys:
        # extract out the information on hydrologic info for regressions
        intables = config["tables"]["path"]
        datacol = config["tables"]["datacol"]
        timecol = config["tables"]["timecol"]
        hydro_da = read_tables(intables, timecol = timecol, datacol = datacol)

    else:
        raise RuntimeError("table inputs or geoglows inputs must be specified")
        return

    if "dates" in config["synthesis"].keys():
        dates = pd.to_datetime(config["synthesis"]["dates"])

    elif "daterange" in config["synthesis"].keys():
        drange = config["synthesis"]["daterange"]
        starttime = drange[0]
        endtime = drange[1]
        # create date information
        dates = pd.date_range(start=starttime, end=endtime, freq="1D")

    else:
        raise RuntimeError("'daterange' or 'dates' values could not be parsed")
        return

    # extract processing options if any
    if "options" in configkeys:
        optkeys = config["options"].keys()

        correct = config["options"]["apply_correction"] if "apply_correction" in optkeys else False
        transform = config["options"]["apply_transform"] if "apply_transform" in optkeys else False
        procwater = config["options"]["water"] if "water" in optkeys else True


    eof_hydro = xr.open_dataset(fitpath)
    hydro_da = hydro_da.sel(reachid = eof_hydro["reachid"], time = dates)

    synthesis = synthesize(eof_hydro, hydro_da)
    print(correct)

    if correct:
        if "observations" in configkeys:
            # extract out the information on reading observation
            obspath = config["observations"]["path"]
            datavar = config["observations"]["datavar"]
            timevar = config["observations"]["timevar"]
            yvar = config["observations"]["yvar"]
            xvar = config["observations"]["xvar"]

            datain = read_observations(obspath, datavar, timevar, yvar ,xvar)
        else:
            raise RuntimeError("observation inputs must be specified if applying correction to synthesis")

        synthesis = correct_synthesis(datain, eof_hydro, synthesis, apply_transform=transform)

    if procwater:
        water = simple_threshold(synthesis)
        out_ds = xr.merge([synthesis,water])

    else:
        out_ds = synthesis.to_dataset()

    write_ds(out_ds, outpath)

    return


    