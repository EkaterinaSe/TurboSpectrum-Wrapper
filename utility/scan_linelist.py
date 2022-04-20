import numpy as np
from sys import argv, exit
import glob

class linelist(object):
    def __init__(self, file = './linelist.txt'):
        """
        Read the linelist in TS format
        """
        with open(file, 'r') as f:
            self.data = f.readlines()
        self.elements = []

        for line in self.data:
            line = line.replace('\n','')

            if line.startswith("'") and line.endswith("'"):
                elID = line.replace("'","").split()[0]
                el = element(elID)
                self.elements.append(el)

                ion = line.replace("'","").split()[1]
            elif line.startswith("'"):
                el.Z = int(line.replace("'","").split()[0])
                el.nlines = int(line.split()[-1])
            elif line.endswith("'"):
                wave  = float(line.split()[0])
                ep = float(line.split()[1])
                loggf = float(line.split()[2])
                com = line.split()[-1]

                el.lines['ion'].append(ion)
                el.lines['wavelength'].append(float(wave))
                el.lines['loggf'].append(loggf)
                el.lines['Ei'].append(ep)
                el.lines['comment'].append(com)



class element(object):
    def __init__(self, ID):
        self.ID = ID
        self.Z = 0
        self.lines = {
                        'ion' : [],\
                        'wavelength' : [], \
                        'loggf' : [],\
                        'Ei' :[],\
                        'comment' :[]
                    }

def identify_line(linelist, wave, element, dist_min = 0.01):
    """
    Get all information about the line from the linelist
    providing the central wavelength and element only
    Identification will most probably be off if more than 1 isotop is present in the linelist!
    """

    out = []
    for el in linelist.keys():
        if el.lower() == element.lower():
            for ion, data in linelist[el].items():
                pos = np.where( abs( data['wave'] - wave ) == min(abs( data['wave'] - wave )) )[0]
                if min(abs( data['wave'] - wave )) < dist_min:
                    print(f"{el} {ion}")
                    for p in pos:
                        # print(f"{wave}:\n {data['rec'][p]}\n")
                        out.append(data['rec'][p])
            return out
            break


def lines_around(linelist, wave_c, dist):
    """
    Find in the linelist and print all lines (of diff. elements)
    within dist Å from the central wavelength
    """
    for el in linelist.keys():
        for ion, data in linelist[el].items():
            pos = np.where( abs( data['wave'] - wave_c ) < dist )[0]
            if len(pos)>0:
                print(f"{el} {ion}")
                for p in pos:
                    print(f"{data['rec'][p]}\n")
                print()

if __name__ == '__main__':

    if len(argv) < 1:
        raise Warning("<Usage>: python3.7 scan_linelist.py linelist.txt. Only working with TS formatted linelist for now.")
    ll_file = argv[1]
    linelist_ges = linelist(file = ll_file).data
