import numpy as np
from sys import argv, exit
import glob

class linelist(object):
    def __init__(self, file = './linelist.txt'):
        """
        Read the linelist in TS format
        """
        with open(file, 'r') as f:
            data = f.readlines()

        self.data = {}
        for line in data:
            l_tmp = line.replace('\n','')

            if l_tmp.startswith("'") and l_tmp.endswith("'"):
            # identification of the element
                element = line.replace("'","").split()[0]
                ion = line.replace("'","").split()[1]
                if not element in self.data.keys():
                    self.data.update({element: {} })
                self.data[element].update({ion: { 'wave':[], 'rec':[] }})
            elif l_tmp.startswith("'"):
            # line before the element line, atomic number, length, sequential number
                pass
            elif l_tmp.endswith("'"):
            # record for a line of the element ^
                wave = line.split()[0]
                self.data[element][ion]['wave'].append(float(wave))
                self.data[element][ion]['rec'].append(line.replace('\n',''))
        for el in self.data.keys():
            for ion in self.data[el].keys():
                for k in self.data[el][ion].keys():
                    self.data[el][ion][k] = np.array(self.data[el][ion][k])



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
