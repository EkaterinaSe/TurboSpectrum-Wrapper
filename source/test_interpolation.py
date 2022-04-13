# external
import os
from sys import argv
import shutil
import subprocess
import datetime
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pickle
import glob
import time
import warnings
# local
import convolve
import read_config
from atmos_package import model_atmosphere
from model_atm_interpolation import *

def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)

def random_interpol_test(all_parameters, request_coords, ind_excl):
    """
    Test the performance of the interpolation
    Remove a random model from the grid and interpolate to its parameters

    Input:
    (dict) all_parameters -- MA grid
                    read by get_all_ma_parameters from model_atm_interpolation
    (int) ind_excl -- which point to exclude?
    (str, tuple) request_coords -- over which coordinates to interpolate?
                                        e.g. Teff, log(g), and [Fe/H]
                                        not Vturb or mass
    """

    " Copy the whole grid of models omitting one random point [at ind_excl]"
    all_shortened = {}

    for k in all_parameters:
        all_shortened.update({ k : np.delete(all_parameters[k], ind_excl, axis=0) } )

    """
    Create an interpolator function using Delanay triangulation
    Coordinates are normalised to the max value
    which is returned in norm_coords
    """
    interp_f, norm_coord = NDinterpolate_MA(all_shortened, request_coords )

    "Apply to the coordinates of the excluded model"
    point = [ all_parameters[k][ind_excl] / norm_coord[k] for k in request_coords]
    # it returns array of interpolated models, so for one take 0th element
    interpolated_structure = interp_f(point)[0]

    orig_model = all_parameters['structure'][ind_excl]

    for i in range(len(all_parameters['structure_keys'][ind_excl])):
        name = all_parameters['structure_keys'][ind_excl][i]
        max_diff = np.max(interpolated_structure[i] - orig_model[i])
        print(f"max abs diff({name}) = { max_diff }")
    print()

    return orig_model, interpolated_structure




if __name__ == '__main__':
    if len(argv) < 2:
        print(f"Specify a number of counts (how many times to perform the test?)")
        exit()
    else:
        count = int(argv[1])

    atmos_path = '/Users/semenova/phd/projects/ts-wrapper/input/atmos/MARCS/all/'

    ma_grid= get_all_ma_parameters(atmos_path, \
                                            format = 'marcs', debug=True)
    interpol_coords = ['teff', 'logg', 'feh', 'vturb']

    "Generate random indexes to exclude models from the grid"
    random_ind = ( np.random.uniform(low=0.0, \
        high=len(ma_grid[interpol_coords[0]])-1.0, size=count) ).astype(int)

    for i in random_ind:
        orig_structure, interpol_structure = \
                            random_interpol_test(ma_grid, interpol_coords, i)

    exit(0)
