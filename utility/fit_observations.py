import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
from observations import readSpectrumTSwrapper, spectrum, read_observations
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
import sys

def restoreFromNormLabels(wavelen, NNet, labelsNorm):
    """ Labels  are already normalised before the call according to the NN used"""
    labels = np.array(labelsNorm)

    """ layers ??? """
    l1 = np.dot(NNet['w_array_0'], labels) + NNet['b_array_0']

    l1[l1<0.0]=0.0

    l2 = np.dot( NNet['w_array_1'], l1 ) + NNet['b_array_1']
    l2 = 1.0 / (1.0 + np.exp(-l2) )

    predictFlux = np.dot( NNet['w_array_2'], l2 ) + NNet['b_array_2']
    """ Interpolate flux to the new requested wavelength scale """
    flux = np.interp(wavelen, NNet['wvl'], predictFlux)

    """ return wavelength and flux """
    return flux


def readNN(NNpath):
    NNet = np.load(NNpath)
    print(f"min(lambda) = {min(NNet['wvl'])}")
    print(f"max(lambda) = {max(NNet['wvl'])}")
    if 'labelsKeys' in NNet:
        print("Labels NN is trained on:")
        for i in range(len(NNet['labelsKeys'])):
            print(f"{NNet['labelsKeys'][i]}: max = {max(NNet['labelsInput'][i])}  \
min = {min(NNet['labelsInput'][i])}")

    return NNet

def callNN(wavelength, obsSpec, NN, labels):
    #fileOut = "./iterFlux.dat"
    #if os.path.isfile(fileOut):
    #    data = np.loadtxt(fileOut)
    #else:
    #    data = None

    spec = spectrum(
                    wavelength,
                    restoreFromNormLabels(wavelength, NN, labels[:-2]), res = np.inf
                    )
    spec.convolve_resolution(obsSpec.R)
    spec.convolve_macroturbulence(labels[-2]*100)
    spec.convolve_rotation(labels[-1]*100)
    #if not isinstance(data, type(None)):
    #    data = np.vstack( ( data, spec.flux) )
    #else: data = spec.flux.copy()
    #np.savetxt(fileOut, data)
    return spec.flux

def fitToNeuralNetwork(obsSpec, NN, fitForLabels = None, quite = True):

    if isinstance(fitForLabels, type(None)):
        fitForLabels = NN['labelsKeys']

    """
    Initialise the labels if not provided
    Extra dimension for macro-turbulence
    """

    initLabels = np.zeros(len(fitForLabels)+2)

    fitFunc = lambda wavelength, *labels : callNN(
                                                wavelength, obsSpec,
                                                NN, labels
                                                )
    popt,_ = curve_fit(
                    fitFunc, obsSpec.lam, \
                    obsSpec.flux, p0=initLabels,\
                    )
    " restore normalised labels "
    popt[:-2] = (popt[:-2] + 0.5)*( NN['x_max'] - NN['x_min'] ) + NN['x_min']
    popt[-2:] =  popt[-2:]*100
    if not quite:
        for i in range(len(fitForLabels)):
            print(f" {fitForLabels[i]} = {popt[i]:.2f} ")
        print(f"Vmac = {popt[-2]:.3f}")
        print(f"Vrot = {popt[-1]:.3f}")
    spec = spectrum(
                    obsSpec.lam,
                    restore(obsSpec.lam, NN, popt[:-2]), res = np.inf
                    )
    spec.convolve_resolution(obsSpec.R)
    spec.convolve_macroturbulence(popt[-1])
    np.savetxt(f"./{obsSpec.ID}_modelFlux.dat", spec.flux)
    return popt

if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra>")
        exit()
    "Fit using Payne neural network"
    nnPath = argv[1]
    obsPath = argv[2]

    NN = readNN(nnPath)

    with open(f"./fittingResults.dat", 'w') as LogResults:
        for obsSpecPath in glob.glob(obsPath):
            print(obsSpecPath)
            w, f = read_observations(obsSpecPath, format = 'ascii')
            # mask = np.logical_and(f> 0, f<1.5)
            # w, f = w[mask], f[mask]
            obsSpec = spectrum(
                            w, f,\
                            res = 15e3
                            )
            obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')

            labelsFit = fitToNeuralNetwork(obsSpec, NN)

            LogResults.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + '\n')
