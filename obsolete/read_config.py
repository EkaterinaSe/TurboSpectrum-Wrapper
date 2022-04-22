import numpy as np
import os
# local
from observations import *
"""
Setup that will be used to run TS
"""
class setup(object):
    def __init__(self, file='config.txt'):

        self.cwd = os.getcwd()

        """
        Reads specifications for the TS run from a config file
        """
        "Some default keys:"
        self.debug = 0
        self.interpolate = 0

        print('Reading configuration file %s' %(file ) )
        for line in open(file, 'r'):
            line=line.strip()
            if not line.startswith("#") and line != '':

                    key, val = line.split('=')
                    key, val = key.strip(), val.strip()
                    if val.startswith("'") or val.startswith('"'):
                        self.__dict__[key] = val[1:-1]
                    elif val.startswith("["):
                        self.__dict__[key] = eval('np.array(' + val + ')')
                    elif '.' in val:
                        self.__dict__[key] = float(val)
                    else:
                        self.__dict__[key] = int(val)

        if self.use_abund == 1:
            self.abund_list = [self.new_abund]
        elif self.use_abund == 2:
            self.abund_list = np.arange(self.start_abund, self.end_abund, self.step_abund)
            # [start, end) --> [start, end]
            self.abund_list = np.hstack((self.abund_list,  self.end_abund ))

        if self.interpolate:
            print(f"Model atmosphere(s) will be created using input parameters")
            print(f"Reading parameters from {self.use_atmos}")
            input = np.loadtxt(self.use_atmos, dtype=float, ndmin=1)
            self.atmos_input_params = {  }
            i = 0
            for k in['teff', 'logg', 'feh', 'vturb']:
                self.atmos_input_params.update({
                    k : input[:, i]
                })
                i +=1
            "Prepare model atmospheres"
            interpolate_ma_grid(self.atmos_path, self.atmos_format, self.debug)

            exit(1)
        else:
            print(f"Reading a list of model atmospheres from {self.use_atmos}")
            self.atmos_list = np.loadtxt(self.use_atmos, ndmin=1, dtype=str, usecols=(0))
            self.atmos_list = [ self.atmos_path + F"/{x}" for x in self.atmos_list ]

        print('Element: %s' %self.element)
        print(F"Use {len(self.abund_list)} abundances")
        print(F"Use {len(self.atmos_list)} model atmospheres")

        if self.nlte == 1:
            self.nlte = True
            self.lte = False

        elif self.nlte == 0:
            self.lte = True
            self.nlte = False

        else:
            print("'nlte' flag unrecognised")
            exit(1)

        if self.nlte:
            if not os.path.isfile(self.depart_grid):
                print(F"Specified departure grid not found: {self.depart_grid}")
            if not os.path.isfile(self.aux_file):
                print(F"Specified auxliarly file not found: {self.aux_file}")


        self.ts_input = { 'PURE-LTE':'', 'NLTE':'', 'MARCS-FILE':'.true.', 'NLTEINFOFILE':'', 'LAMBDA_MIN':0, 'LAMBDA_MAX':0, 'LAMBDA_STEP':0, 'MODELOPAC':'OPAC', 'RESULTFILE':'' }
        if self.lte:
            self.ts_input['PURE-LTE'] = '.true.'
            self.ts_input['NLTE'] = '.false.'

        if self.nlte:
            self.ts_input['PURE-LTE'] = '.false.'
            self.ts_input['NLTE'] = '.true.'

        if self.atmos_format.lower() != 'marcs':
            self.ts_input['MARCS-FILE'] = '.false.'
            print("NOTE: specific commenting in the model atmosphere is required")


        """ At what wavelenght range to compute a spectrum? """
        if  self.lam_end < self.lam_start:
            tmp = self.lam_start
            self.lam_start = self.lam_end
            self.lam_end = tmp


        """ Compute wavelenght step from resolution """
        self.wave_step = np.mean([self.lam_start, self.lam_end]) / self.resolution

        self.ts_input['LAMBDA_MIN'] = self.lam_start
        self.ts_input['LAMBDA_MAX'] = self.lam_end
        self.ts_input['LAMBDA_STEP'] = self.wave_step

        """ Linelists """
        if type(self.linelist) == np.ndarray:
            pass
        elif type(self.linelist) == str:
            self.linelist = np.array([self.linelist])
        else:
            print("Do not understand linelist argument. Stopped.")
            exit(1)
        print("Linelist(s) will be read from:", self.linelist)

        self.ts_input['NFILES'] = len(self.linelist)
        self.ts_input['LINELIST'] = '\n'.join(self.linelist)

        """
        Find Z (atomic number of the element)
        and check the agreenment with the config file
        """
        if os.path.isfile('./atomic_numbers.dat'):
            el_z = np.loadtxt('./atomic_numbers.dat', usecols=(0))
            el_id = np.loadtxt('./atomic_numbers.dat', usecols=(1), dtype=str)
        else:
            print("Can not find './atomic_numbers.dat' file. Stopped.")
            exit(1)
        for i in range(len(el_id)):
            if self.element.lower() == el_id[i].lower():
                z = el_z[i]
        if not 'element_z' in self.__dict__.keys():
            self.element_z = z
        elif z != self.element_z :
            print(F"config file states Z={self.element_z} for element {self.element}, but according to ./atomic_numbers.dat Z={z}. Stopped.")
            exit(1)
