from sys import argv, exit
import numpy as np
from multiprocessing import Pool
import glob
from observations import readSpectrumTSwrapper
import h5py
import cProfile
import pstats
from multiprocessing import Pool

def gather_data(specList):
    # spectraList, wave_new, Rnew, quite, limits = arg
    spectraIncl = []
    # get the labels
    spec = readSpectrumTSwrapper(specList[0])
    wvl = spec.lam
    labels = { k:[] for k in spec.labels}
    flxl = np.zeros( shape=(len(specList), len(spec.lam) ) )

    for i, specFile in enumerate(specList):
        spec = readSpectrumTSwrapper(specFile)
        flxl[i] = spec.flux
        for k in labels:
            labels[k].append(spec.__dict__[k])
    return (flxl, labels, wvl)

if __name__ == '__main__':

    path = argv[1]
    if len(argv) > 2:
        ncpu = int(argv[2])
    else:
        ncpu = 1
    specList = glob.glob(path)[:100]

    # profiler = cProfile.Profile()
    # profiler.enable()

    args = [ spectraList[i::ncpu] for i in range(ncpu)]
    with Pool(processes=ncpu) as pool:
        out = pool.map(gather_data, args )
    # for i in range(len(out)):
    #     fl = out[i][0]
    #     lab   = out[i][1]
        # w   = out[i][1]
    flxl = np.hstack( (out[i][0] for i in range(len(out))) )
    print( np.shape(flxl) )
    # flxl, labels, wvl = gather_data(specList)

    with h5py.File('./test.h5', 'w') as hf:
        hf.create_dataset( 'fluxes', data=flxl, shape=np.shape(flxl), dtype='float64')
        for k in labels:
            hf.create_dataset( f"{k}", data=labels[k] )
        hf.create_dataset( 'wave', data=wvl, shape=np.shape(wvl), dtype='float64')



    # with h5py.File('./test.h5', 'r') as hf:
    #     print( list(hf.keys()) )
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()



    exit()
