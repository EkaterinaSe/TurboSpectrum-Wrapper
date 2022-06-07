from sys import argv
import numpy as np

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
