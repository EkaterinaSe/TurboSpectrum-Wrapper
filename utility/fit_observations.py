import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
from observations import readSpectrumTSwrapper, spectrum, read_observations
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
from PayneModule import restore, restoreFromNormLabels, readNN
import sys

def callNN(wavelength, obsSpec, NN, p0, freeLabels, setLabels, quite=True):
    """
     To ensure the best convergence this function needs to be called on normalised labels (in p0)
     maybe it would work withput normalisation? it would make the code so much nicer
    """
    labels = setLabels.copy()
    labels[freeLabels] = p0

    spec = spectrum(
                    wavelength,
                    restoreFromNormLabels(wavelength, NN, labels[:-2]), res = np.inf
                    )
    spec.convolve_resolution(obsSpec.R)
    spec.convolve_macroturbulence( (labels[-2] + 0.5)*50, quite=quite)
    spec.convolve_rotation(( labels[-1]+0.5)*50, quite=quite)


#    fileOut = "./iterFlux.dat"
#    if os.path.isfile(fileOut):
#        data = np.loadtxt(fileOut)
#    else:
#        data = None
#
#    if not isinstance(data, type(None)):
#        data = np.vstack( ( data, spec.flux) )
#    else: data = spec.flux.copy()
#    np.savetxt(fileOut, data)
    return spec.flux

def fitToNeuralNetwork(obsSpec, NN, prior = None, quite = True):

    freeLabels = np.full(len(NN['labelsKeys'])+2, True)
    setLabels = np.full(len(NN['labelsKeys'])+2, 0.0)
    if isinstance(prior, type(None)):
        pass
    else:
        if len(prior)  < len(NN['labelsKeys']) + 2:
            for i, l in enumerate( np.hstack( (NN['labelsKeys'], ['vmac', 'vrot']))):
                l = l.lower()
                if l in prior:
                    freeLabels[i] = False
                    setLabels[i] = prior[l]
            
        elif prior.keys() != NN['labelsKeys']:
            print(f"Provided prior on the labels {prior} does not match labels ANN was trained on: {NN['labelsKeys']}")
            exit()

    """
    Initialise the labels if not provided
    Extra dimension is for macro-turbulence and rotation
    """

    initLabels = np.zeros( np.sum(freeLabels) )
    norm = {'min' : np.hstack( [NN['x_min'], [0, 0]] ), 'max': np.hstack( [NN['x_max'], [50, 50]] ) }
    for i, p in enumerate(setLabels):
        if np.isnan(p):
            setLabels[i] = 0
        else:
            setLabels[i] = (setLabels[i] - norm['min'][i] ) / ( norm['max'][i] - norm['min'][i] ) - 0.5

    fitFunc = lambda wavelength, *labels : callNN(
                                                wavelength, obsSpec,
                                                NN, labels, freeLabels, setLabels, quite = quite
                                                )
    popt,_ = curve_fit(
                    fitFunc, obsSpec.lam, \
                    obsSpec.flux, p0=initLabels,\
                    bounds = (-0.5, 0.5)
                    )
    " restore normalised labels "
    setLabels[freeLabels] = popt
    setLabels = (setLabels+ 0.5)*( norm['max'] - norm['min'] ) + norm['min']

    spec = spectrum(
                    obsSpec.lam,
                    restore(obsSpec.lam, NN, setLabels[:-2]), res = np.inf
                    )
    spec.convolve_macroturbulence(setLabels[-2])
    spec.convolve_rotation(setLabels[-1])
    chi2 = np.sqrt(np.sum(obsSpec.flux - spec.flux)**2)
    np.savetxt(f"./{obsSpec.ID}_modelFlux.dat", spec.flux)
    return setLabels, chi2

if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra>")
        exit()
    "Fit using Payne neural network"
    nnPath = argv[1]
    NNs = glob.glob(nnPath)
    print(f"found {len(NNs)} ANNs")
    obsPath = argv[2]
    specList = glob.glob(obsPath)
    print(f"found {len(specList)} observed spectra")
    for nnPath in NNs:
        NN = readNN(nnPath)
        NNid = nnPath.split('/')[-1].replace('.npz', '').strip() 

        out = {'file':[], 'chi2':[], 'vmac':[], 'vrot':[], 'diffMg':[], 'diffLogg':[]}
        with open(f"./fittingResults_{NNid}.dat", 'w') as LogResults:
            LogResults.write( "#" + '\t'.join(NN['labelsKeys']) + ' Vmac    Vrot  chi2\n' )
            for obsSpecPath in specList:
                print(obsSpecPath)
                out['file'].append(obsSpecPath)
                obsSpec = readSpectrumTSwrapper(obsSpecPath)
                obsSpec.convolve_resolution(NN['res'])
                # resolution is considered constant, therefore FWHM will be bigger for smaller wavelngth range
                # be careful with the resolution convolution for both observations and ANN restored fluxes
                obsSpec.cut([min(NN['wvl']), max(NN['wvl'])] )
                #obsSpec.cut([5182, 5185] )
                obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')
    
                prior = {}
                for l in NN['labelsKeys']:
                    if l.lower() != 'mg':
                    #if l.lower() != 'logg':
                        prior[l.lower()] = obsSpec.__dict__[l]
                prior['vmac'] = 0
                prior['vrot'] = 0
    
                labelsFit, bestFitChi2 = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True)
                for i, l in enumerate(NN['labelsKeys']):
                    if l not in out:
                        out.update({l:[]})
                    out[l].append(labelsFit[i])
                out['diffMg'].append( obsSpec.Mg - out['Mg'][-1] )
                out['diffLogg'].append( obsSpec.logg - out['logg'][-1] )
                out['vmac'].append(labelsFit[-2])
                out['vrot'].append(labelsFit[-1])
                out['chi2'].append(bestFitChi2)
                   
                LogResults.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + f"{bestFitChi2 : .3f}\n")
        with open(f'./fittingResults_{NNid}.pkl', 'wb') as f:
            pickle.dump(out, f)
