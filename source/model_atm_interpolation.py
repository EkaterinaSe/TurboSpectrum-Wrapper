# external
import os
from sys import argv
import shutil
import subprocess
import datetime
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay
import pickle
import glob
import time
import warnings
# local
from .atmos_package import model_atmosphere


def get_all_ma_parameters(models_path, depthScaleNew, format='m1d', debug = False):
    """
    Gets a list of all available model atmopsheres and their parameters
    for interpolation later on.
    If no list is available, creates one by scanning through all available
    models in the specified input directory.

    Parameters
    ----------
    models_path : str
        input directory contatining all available model atmospheres
    depthScaleNew : array
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    format : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    debug : boolean
        switch detailed print out

    Returns
    -------
    MAgrid : dict
        dictionary containing grid of model atmospheres including both
        the parameters (like Teff, log(g), etc)
        and structure (density as a function of depth, etc)
    """

    save_file = f"{models_path}/all_models_save.pkl"

    if os.path.isfile(save_file) and os.path.getsize(save_file) > 0:
        if debug:
            print(f"reading pickled grid of model atmospheres from {save_file}")
        with open(save_file, 'rb') as f:
            MAgrid = pickle.load(f)
        depthScaleNew = MAgrid['structure'][:, np.where(MAgrid['structure_keys'][0] == 'tau500')[0][0] ]
        if np.shape(depthScaleNew) != np.shape(np.unique(depthScaleNew, axis=1)):
            print(f"depth scale is not uniform in the model atmosphere grid read from {save_file}")
            print(f"try removing file {save_file} and run the code again")
            exit()
        else:
            depthScaleNew = np.array(depthScaleNew[0])
    else:
        print(f"Checking all model atmospheres under {models_path}")

        MAgrid = {
        'teff':[], 'logg':[], 'feh':[], 'vturb':[], 'file':[], 'structure':[], 'structure_keys':[], 'mass':[]\
        }

        with os.scandir(models_path) as all_files:
            for entry in all_files:
                if not entry.name.startswith('.') and entry.is_file():
                    # try:
                    file_path = models_path + entry.name
                    ma = model_atmosphere()

                    ma.read(file_path, format=format)

                    if ma.mass <= 1.0:

                        MAgrid['teff'].append(ma.teff)
                        MAgrid['logg'].append(ma.logg)
                        MAgrid['feh'].append(ma.feh)
                        MAgrid['vturb'].append(ma.vturb[0])
                        MAgrid['mass'].append(ma.mass)

                        MAgrid['file'].append(entry.name)

                        ma.temp = np.log10(ma.temp)
                        ma.ne = np.log10(ma.ne)

                        # bring all values to the same depth_scale (tau500)
                        for par in ['temp', 'ne', 'vturb']:
                            f_int = interp1d(ma.depth_scale, ma.__dict__[par], fill_value='extrapolate')
                            ma.__dict__[par] = f_int(depthScaleNew)
                        ma.depth_scale = depthScaleNew

                        MAgrid['structure'].append( np.vstack( (ma.depth_scale, ma.temp, ma.ne, ma.vturb )  ) )
                        MAgrid['structure_keys'].append( ['tau500', 'temp', 'ne', 'vturb'])

                    # except: # if it's not a model atmosphere file, or format is wrong
                    #         if debug:
                    #             print(f"Cound not read model file {entry.name} for model atmosphere")

        for k in MAgrid:
            MAgrid[k] = np.array(MAgrid[k])

        " Check if any model atmosphere was successfully read "
        if len(MAgrid['file']) == 0:
            raise Exception(f"no model atmosphere parameters were retrived from files under {models_path}.\
Try setting debug = 1 in config file. Check that expected format of model atmosphere is set correctly.")
        else:
            print(f"{len(MAgrid['file'])} model atmospheres in the grid")

        "Print UserWarnings about any NaN in parameters"
        for k in MAgrid:
            try: # check for NaNs in numeric values:
                if np.isnan(MAgrid[k]).any():
                    pos = np.where(np.isnan(MAgrid[k]))
                    for p in pos:
                        message = f"NaN in parameter {k} from model atmosphere {MAgrid['path'][p]}"
                        warnings.warn(message, UserWarning)
            except TypeError: # ignore other [non-numerical] keys, such as path, name, etc
                pass
        "Dump all in one file (only done once)"
        with open(save_file, 'wb') as f:
            pickle.dump(MAgrid, f)
    return MAgrid

def preInterpolationTests(data, interpol_coords, valueKey, dataLabel = 'default'):
    """
    Run multiple tests to catch possible exceptions
    that could affect the performance of the underlying
    Qnull math engine during Delaunay triangulation
    Parameters
    ----------
    data : str
        input directory contatining all available model atmospheres
    interpol_coords : array
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    valueKey : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    dataLabel : boolean
        switch detailed print out

    Returns
    -------
    boolean
    """

    " Check for degenerate parameters (aka the same for all grid points) "
    for k in interpol_coords:
        if max(data[k]) == min(data[k]):
            print(f"Grid {dataLabel} is degenerate in parameter {k}")
            print(F"Values: {np.unique(data[k])}")
            return False

    " Check for repetitive points within the requested coordinates "
    test = [ data[k] for k in interpol_coords]
    if len(np.unique(test, axis=1)) != len(test):
        print(f"Grid {dataLabel} with coordinates {interpol_coords} \
has repetitive points")
        return False

    "Any coordinates correspond to the same value? e.g. [Fe/H] and A(Fe) "
    for k in interpol_coords:
        for k1 in interpol_coords:
            if k != k1:
                diff = 100 * ( np.abs( data[k] - data[k1]) ) / np.mean(np.abs( data[k] - data[k1]))
                if np.max(diff) < 5:
                    print(f"Grid {dataLabel} is only {np.max(diff)} % different \
in parameters {k} and {k1}")
                    return False

    for k in interpol_coords:
        if np.isnan(data[k]).any():
                print(f"Warning: found NaN in coordinate {k} in grid '{dataLabel}'")
    if np.isnan(data[valueKey]).any():
        print(f"Found NaN in {valueKey} array of {dataLabel} grid")
    return True


def NDinterpolateGrid(all_par, interpol_par, valueKey = 'structure', dataLabel='model_atm'):

    " Normalise the coordinates of the grid "
    points = []
    norm_coord = {}
    for k in interpol_par:
            points.append(all_par[k] / max(all_par[k]) )
            norm_coord.update( { k :  max(all_par[k])} )
    points = np.array(points).T

    "Create the function that interpolates model atmospheres structure"
    values = np.array(all_par[valueKey])
    interp_f = LinearNDInterpolator(points, values)

    return interp_f, norm_coord


if __name__ == '__main__':
    exit(0)
