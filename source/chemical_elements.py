import numpy as np
import os
import shutil
from sys import argv, exit
import datetime
import glob

def atomicZ(el):
    """
    Finds atomic number (Z) of the chemical element by comparing its name
    to the list stored in ./atomic_numbers.dat

    Parameters
    ----------
    el : str
        element name e.g. 'Mg'
    """
    if os.path.isfile('./atomic_numbers.dat'):
        el_z = np.loadtxt('./atomic_numbers.dat', usecols=(0), dtype=int)
        el_id = np.loadtxt('./atomic_numbers.dat', usecols=(1), dtype=str)
    else:
        print("Can not find './atomic_numbers.dat' file. Stopped.")
        exit(1)
    for i in range(len(el_id)):
        if el.lower() == el_id[i].lower():
            return el_z[i]


class ChemElement(object):
    """
    Class for handling individual chemical elements. Gets atomic number
    and checks whether element is Fe or H when initialised

    Parameters
    ----------
    ID : str
        element name e.g. 'Fe'
    """
    def __init__(self, ID = ''):

        self.ID = ID.strip().capitalize()
        self.Z = atomicZ(self.ID)
        self.nlte = False
        self.comment = ""

        # TODO: If you find a nicer way to figure this out
        # TODO: please change here and the rest of the code will manage
        if ID.strip().lower() == 'fe' and self.Z == 26:
            self.isFe = True
        else: self.isFe = False

        if ID.strip().lower() == 'h' and self.Z == 1:
            self.isH = True
        else: self.isH = False
