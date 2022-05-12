import numpy as np
from sys import argv, exit
import glob
import pickle
from scipy import interpolate
from observations import readSpectrumTSwrapper, spectrum

def gatherModelData(Path):
    pickledGridPath = Path + 'gridCoordinates.pkl'
    if os.path.isfile(pickledGridPath) and os.path.getsize(pickledGridPath) > 0:
        with open(pickledGridPath, 'rb') as f:
            modelGrid = pickle.load(f)
    else:
        print(f"reading model spectra under {Path} ...")
        # TODO: parallelise
        ncpu = 1
        specList = glob.glob(Path)
        args = [ specList[i::ncpu] for i in range(ncpu)]
        modelGrid = { }

        with Pool(processes=ncpu) as pool:
            out = pool.map(readModelSpectralGrid, args )
        " Gather output from parallel processes "
        for grid in out:
            for k in grid:
                if not k in spectralModelGrid:
                    modelGrid.update( { k : [] } )
                modelGrid[k].extend(grid[k])
        for k in modelGrid:
            modelGrid[k] = np.array( modelGrid[k] )
        with open(pickledGridPath, 'wb') as f:
            pickle.dump(modelGrid, f)

    return modelGrid

def findDistanceSpectralCoords(spectraGrid, labels):
    """
    find the closest spectrum in the grid based on direct quadratic distance
    """
    dist = 0
    for k in labels:
        dist += ( spectraGrid[k] - labels[k] )**2
    pos = np.where(dist == min(dist))[0]
    if len(pos) > 1:
        print(f"found {len(pos)} 'closest' spectra:")
        for p in pos:
            print(spectraGrid['path'][p].split('/')[-1])
    return pos[0]


def getClosestModel(lamObs, modelGrid, *labels):
    pos = findDistanceSpectralCoords(modelGrid, labels)
    modSpec = readSpectrumTSwrapper( modelGrid['path'][pos] )
    flux = np.interp(lamObs, spec.lam, spec.flux )
    return flux


def fitObservedSpectrum(obsSpec, modelGrid, initLabels = None):

    " Initialise the labels if not provided "
    if isinstance(initLabels, type(None)):
        initLabels = []
        for k in modelGrid:
            initLabels.append( np.mean( modelGrid[k] ) )

    fitFunc = lambda lam, label : getClosestModel(lam, modelGrid, label)

    popt, pcov = curve_fit(
                            fitFunc,
                            obsSpec.lam,
                            p0 = initLabels,
                            )
    bestFitFlux = getClosestModel(obsSpec.lam, modelGrid, popt)
    return popt, bestFitFlux


if __name__ == '__main__':
    modelPath = argv[1]
    obsPath = argv[2]
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra> <path to observed spectra> ")
        exit()

    modelGrid = gatherModelData(modelPath)

    for obsSpecPath in glob.glob(obsPath):
        obsSpec = spectrum(
                        read_observations(obsSpecPath, format = 'ascii'), \
                        res = np.inf
                        )
        fitObservedSpectrum(obsSpec, modelGrid, initLabels=None )









    exit()
