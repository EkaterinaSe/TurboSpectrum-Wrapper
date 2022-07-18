from sys import argv, exit
import numpy as np
from multiprocessing import Pool
import glob
from observations import readSpectrumTSwrapper
import h5py
import cProfile
import pstats
from multiprocessing import Pool

def gather_data( arg ):
    specList, wvl, labels = arg
    # spectraList, wave_new, Rnew, quite, limits = arg
    spectraIncl = []
    # get the labels
    totSize = len(specList)
    labels = { k:np.full( totSize, np.nan ) for k in labels}
    flxl = np.full( shape=( totSize, len(wvl) ), fill_value = np.nan )
    names = []
    ## TODO:
    comments = []

    for i, specFile in enumerate(specList):
        spec = readSpectrumTSwrapper(specFile)
        names.append(specFile)
        if not isinstance(spec, type(None)):
            flxl[i] = np.interp(wvl, spec.lam, spec.flux)
            for k in labels:
                if k in spec.__dict__:
                    labels[k][i] = spec.__dict__[k]
        # fluxes and labels are initialised to NaNs
        else: pass
                

    return (flxl, labels, wvl, names)

if __name__ == '__main__':

    path = argv[1]
    if len(argv) > 2:
        ncpu = int(argv[2])
    else:
        ncpu = 1
    specList = glob.glob(path)
    print(f"found {len(specList)} files...")

    # profiler = cProfile.Profile()
    # profiler.enable()
    spec = readSpectrumTSwrapper(specList[0])
    wvl = spec.lam
    labels = spec.labels

    args = [ [specList[i::ncpu], wvl, labels] for i in range(ncpu)]
    with Pool(processes=ncpu) as pool:
        out = pool.map(gather_data, args )

    flxl = np.vstack( list(out[i][0] for i in range(len(out))) )
    labels = { k : np.hstack( list(out[i][1][k] for i in range(len(out))) ) for k in out[0][1]  }
    names = []
    for i in range(len(out)):
        names.extend( out[i][3])
    names = [a.encode('utf8') for a in names]

    with h5py.File('./test.h5', 'w') as hf:
        hf.create_dataset( 'flux', data=flxl, shape=np.shape(flxl), dtype='float64')
        hf.create_dataset('labelKeys', data = [a.encode('utf8') for a in labels] )
        for k in labels:
            hf.create_dataset( f"{k}", data=labels[k] )
        hf.create_dataset( 'wave', data=out[0][2], dtype='float64')
        hf.create_dataset( 'ID', data=names)

    with h5py.File('./test.h5', 'r') as hf:
        print( list(hf.keys()) )
        #print( [a.decode('utf8') for a in hf['ID']] )
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()



    exit()
