# external
import os
import sys
from sys import argv
import shutil
import datetime
import numpy as np
import pickle
import glob
from matplotlib import pyplot as plt
sys.path.append('/Users/semenova/phd/projects/TurboSpectrum-Wrapper/')
# local
from source.model_atm_interpolation import get_all_ma_parameters, NDinterpolateGrid, preInterpolationTests
from source.read_nlte import *

def testInterpolNLTE(nlte_data, interpol_parameters,  i):
    if 'depthScale' in nlte_data:
        nlte_data['departNew'] = np.full((np.shape(nlte_data['depart'])[0], np.shape(nlte_data['depart'])[1]+1, np.shape(nlte_data['depart'])[2]), np.nan)
        for i in range(len(nlte_data['pointer'])):
            nlte_data['departNew'][i] = np.vstack([nlte_data['depthScale'][i], nlte_data['depart'][i]])
        nlte_data['depart'] = nlte_data['departNew'].copy()
        del nlte_data['departNew']
        del nlte_data['depthScale']


    if 'comment' in nlte_data:
        if len(nlte_data['comment'].strip()) > 0:
            print( nlte_data['comment'])
        del nlte_data['comment']

    passed = preInterpolationTests(nlte_data, \
                                        interpol_parameters, \
                                        valueKey='depart', \
                                        dataLabel=f"")
    if passed:

        for k in interpol_parameters:
            interpol_parameters[k] = nlte_data[k]

        mask = np.full(len(nlte_data['pointer']), True)
        mask[i] = False

        nlte_dataCopy = {}
        for k in nlte_data:
            nlte_dataCopy[k] = nlte_data[k][mask]

        interp_f, params_to_interpolate = NDinterpolateGrid(nlte_dataCopy, interpol_parameters, valueKey = 'depart')

        point = np.array([ nlte_data[k][i] / params_to_interpolate[k] for k in params_to_interpolate])
        return interp_f(point)[0]

if __name__ == '__main__':
    bin_file = './NLTEgrid_H_MARCS_May-10-2021.bin'
    aux_file = './auxData_H_MARCS_May-10-2021.txt'
    nlte_data = read_fullNLTE_grid( bin_file, aux_file )

    atmos_path = '/Users/semenova/phd/projects/ts-wrapper/input/atmos/MARCS/all/'
    interpol_parameters = { 'teff':None, 'logg':None, 'feh':None, 'vturb':None}
    i = 100

    testInterpolNLTE(nlte_data, interpol_parameters, i)
