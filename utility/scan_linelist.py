import numpy as np
from sys import argv, exit
import glob

class linelist(object):
    def __init__(self, file = './linelist.txt', fmt='GES'):
        if fmt.lower() == 'ges':
            """
            Read the linelist in GES format
            """
            with open(file, 'r') as f:
                self.data = f.readlines()
            self.elements = []
            included = []
            for rec in self.data:
                rec = rec.replace('\n','')

                if rec.startswith("'") and rec.endswith("'"):
                    elID = rec.replace("'","").split()[0]
                    if elID not in included:
                        included.append(elID)
                        el = element(elID)
                        el.Z = z
                        el.nlines = nlines
                        self.elements.append(el)

                    ion = rec.replace("'","").split()[1]
                elif rec.startswith("'"):
                    z =  int(float(rec.replace("'","").split()[0]))
                    nlines = int(rec.split()[-1])
                elif rec.endswith("'"):
                    wave  = float(rec.split()[0])
                    ep = float(rec.split()[1])
                    loggf = float(rec.split()[2])
                    com = rec.split()[-1]

                    ll = line(wave, el)
                    ll.ion = ion
                    ll.ID += f"{ll.ion}"
                    el.lines.append(ll)
                    ll.Ei = ep
                    ll.loggf = loggf
                    ll.comment = com
        if fmt.lower() == 'h':
            with open(file, 'r') as f:
                self.data = [l for l in  f.readlines()  if not l.startswith("'")]
            el = element('H')
            el.Z = 1
            el.nlines = len(self.data)

            for rec in self.data:
                wave = float(rec.split()[0])
                ep = float(rec.split()[3])
                loggf = float(rec.split()[5])
                com = rec.split()[6]

                ll = line(wave, el)
                ll.ion = 'I'
                ll.ID += f"{ll.ion}"
                ll.Ei = ep
                ll.loggf = loggf
                ll.comment = com
                el.lines.append(ll)

            self.elements = [el]


class element(object):
    def __init__(self, ID):
        self.ID = ID
        self.Z = 0
        self.lines = []

class line(element):
    def __init__(self, lam, parent):
        self.Z = parent.Z
        self.ID = parent.ID
        self.lam = lam
        self.ion = 0
        self.loggf = np.nan
        self.Ei = np.nan

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
    for el in linelist.elements:
        for l in el.lines:
            if np.abs(l.lam - wave_c) < dist:
                print(f"{l.ID} {l.lam:.3f} ep={l.Ei} log(gf)={l.loggf}")
                print(l.comment)

if __name__ == '__main__':

    if len(argv) < 1:
        raise Warning("<Usage>: python3.7 scan_linelist.py linelist.txt. Only working with TS formatted linelist for now.")
    ll_file = argv[1]
    linelist_ges = linelist(file = ll_file).data
