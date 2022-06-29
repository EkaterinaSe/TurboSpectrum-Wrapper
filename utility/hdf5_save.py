from sys import argv, exit
import numpy as np
from multiprocessing import Pool
import glob
from observations import readSpectrumTSwrapper
import h5py

def gather_data(specList):
    # spectraList, wave_new, Rnew, quite, limits = arg
    flxl = []
    spectraIncl = []
    # get the labels
    spec = readSpectrumTSwrapper(specList[0])
    labels = { k:[] for k in spec.labels}

    for specFile in specList:
        spec = readSpectrumTSwrapper(specFile)
        flxl.append(spec.flux)
        for k in labels:
            labels[k].append(spec.__dict__[k])
    print(flxl)
    print(labels)
    return flxl, labels

if __name__ == '__main__':

    path = argv[1]
    specList = glob.glob(path)[:10]
    gather_data(specList)

    with h5py.File('./test.h5', 'w') as hf:
        hf.creat_dataset( 'fluxes', flxl )
        hf.creat_dataset( 'labels', labels )




    with open('./test.h5', 'r') as hf:
        print(hf.keys())
        print(hf['labels'])






    exit()
