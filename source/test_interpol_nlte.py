# external
import os
from sys import argv
import shutil
import datetime
import numpy as np
import pickle
import glob
from matplotlib import pyplot as plt
# local
import read_config
from model_atm_interpolation import *
from read_nlte import *


if __name__ == '__main__':
    " Read NLTE grid of departure coefficients "
    grid_file_path = '/Users/semenova/phd/projects/ts-wrapper/input/nlte_grid/'
    bin_file = grid_file_path + 'NLTEgrid_H_MARCS_May-10-2021.bin'
    aux_file = grid_file_path + 'auxData_H_MARCS_May-10-2021.txt'

    # read every record and it's parameters
    nlte_data = read_full_grid( bin_file, aux_file )


    # get the grid of model atmospheres
    # and bring NLTE departure grid to the same depth scale
    atmos_path = '/Users/semenova/phd/projects/ts-wrapper/input/atmos/MARCS/all/'
    "Read all model atmospheres"
    all_parameters = get_all_ma_parameters(atmos_path, \
                                            format = 'marcs', debug=True)



    interpol_parameters = { 'teff':None, 'logg':None, 'feh':None, 'vturb':None}
    for k in interpol_parameters:
        interpol_parameters[k] = nlte_data[k]

    interp_f, params_to_interpolate = NDinterpolate_NLTE_grid(interpol_parameters, nlte_data)
    print(params_to_interpolate)

    i = 100
    point = np.array([ nlte_data[k][i] / params_to_interpolate[k] for k in params_to_interpolate])
    dep_int = interp_f(point)[0]

    print(dep_int)
    x = dep_int[0]
    nk = 2
    y =  dep_int[nk]

    x0 = nlte_data['depart'][i][0]
    y0 = nlte_data['depart'][i][nk]
    plt.plot(x, y, 'k--')
    plt.plot(x0, y0, 'r-')

    plt.show()
