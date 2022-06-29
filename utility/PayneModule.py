from sys import argv
import numpy as np
import sys
sys.path.append('/home/semenova/codes/ts-wrapper/TurboSpectrum-Wrapper/')
from utility.observations import spectrum, readSpectrumTSwrapper, read_observations
from multiprocessing import Pool

def compareANN2validation(NNpath, spectraOriginal, res=20e3, ncpu=1):
    args = [ [NNpath, spectraOriginal[i::ncpu], res, True] for i in range(ncpu)]
    args[0][-1] = False
    with Pool(processes=ncpu) as pool:
         out = pool.map( compareSpectra, args )
    comparison = {}
    for i in range(len(out)):
        for k in out[i]:
            if k not in comparison:
                comparison.update( { k : [] } )
            comparison[k].extend( out[i][k] )
    for k in comparison:
        comparison[k] = np.array(comparison[k])
    return comparison

def compareSpectra(arg):
    NNpath, spectraOriginal, res, debug = arg
    NN = readNN(NNpath, quite=True)
    comparison = {
        'name':[],
        'diff':[],
        'teff':[],
        'logg':[],
        'feh' : [],
        'vturb' : []
    }
    for i in range(len(spectraOriginal)):
        if not debug:
            print(f"{i}/{len(spectraOriginal)}")
        specOrig = readSpectrumTSwrapper( spectraOriginal[i] )
        specOrig.cut( [min(NN['wvl']), max(NN['wvl'])] )
        if len(specOrig.lam) != len(NN['wvl']):
            specOrig = None

        if not isinstance(specOrig, type(None)):
            specOrig.convolve_resolution(res)
            labels = [ specOrig.__dict__[k] for k in NN['labelsKeys']]
            fRe = restore(specOrig.lam, NN, labels)
            comparison['name'].append(spectraOriginal[i])
            diff = (fRe-specOrig.flux)/specOrig.flux
            comparison['diff'].append( diff)
            comparison['teff'].append( specOrig.teff )
            comparison['logg'].append( specOrig.logg )
            comparison['feh'].append( specOrig.feh )
            comparison['vturb'].append( specOrig.vturb )
    return comparison

def restore(wavelen, NNet, labels):
    """ Normalised labels """
    labels = np.array(labels)
    labels = (labels - NNet['x_min'] ) / ( NNet['x_max'] - NNet['x_min'] ) - 0.5

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


def readNN(NNpath, quite=False):
    NNet = np.load(NNpath)
    if not quite:
        print(f"min(lambda) = {min(NNet['wvl'])}")
        print(f"max(lambda) = {max(NNet['wvl'])}")
        if 'labelsKeys' in NNet:
            print("Labels NN is trained on:")
            for i in range(len(NNet['labelsKeys'])):
                print(f"{NNet['labelsKeys'][i]}: min = {min(NNet['labelsInput'][i])}  \
max = {max(NNet['labelsInput'][i])}")

    return NNet

if __name__ == '__main__':
     """
     """

     NNet = readNN(argv[1])

     x_new = np.arange(5000, 8000, 100)
     labels = np.array([5500, 4.0, -1])

     """
     Normalise in the same way it was done for training the NN
     """
     f = restore(x_new, NNet, labels)
