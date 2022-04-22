import numpy as np
import os
from sys import argv, exit
from astropy import convolution
import glob
import pickle
from scipy import interpolate
from functools import reduce
from astropy import constants as const
from astropy.modeling import Fittable1DModel, Parameter
from copy import deepcopy



def read_observations(path, format):
    """ Read observed spectrum """
    if format.lower() == 'kpno':
        w_obs, f_obs = [], []
        for f in glob.glob(path + '/flux*.txt'):
            w, f = np.loadtxt(f, usecols=(0,1), unpack=True)
            w_obs.extend(w)
            f_obs.extend(f)
        w_obs, f_obs = np.array(w_obs), np.array(f_obs)
        # nm to AA
        w_obs = w_obs * 10
    if format.lower() == 'ascii':
        w_obs, f_obs = np.loadtxt(path, unpack=True)

    # EXCLUDE REPEATING WAVELENGTH POINTS
    _, un_ind = np.unique(w_obs, return_index=True)
    print(F"Excluded {(len(w_obs) - len(w_obs[un_ind]))} (out of {len(w_obs)}) repeating points")
    w_obs, f_obs = w_obs[un_ind], f_obs[un_ind]

    return w_obs, f_obs

def select_within_linemask(line_masks, element, w_obs, w_model, debug=False):
    """
    Select wavelength points within a given linemask for a chemical element.
    Always work on synthetic and model spectra at the same time. For safety reasons.
    """
    element = element.lower()

    # check whether wavelenght fall in the segment

    if element not in line_masks.keys():
        print(F"No linemask for {element.capitalize()}. Stopped")
        exit(1)
    else:
        lm = line_masks[element]
        # check whether wavelenght fall in the segment
        check = np.full(len(lm['w_center']), False)
        ind_model = []
        for j in range(len(lm['w_center'])):
            mask = np.where(np.logical_and.reduce([ w_model >= lm['w_start'][j], w_model <= lm['w_end'][j] ]))[0]
            if len(mask) > 0:
                ind_model.append(mask)
            else:
                check[j] = True
                if debug:
                    print(F"No model points fall within {lm['w_start'][j]} -- {lm['w_end'][j]} AA")

        ind_obs = []
        for j in range(len(lm['w_center'])):
            # condition on observed spectrum required for interpolation
            mask = np.where(np.logical_and.reduce([ w_obs >= lm['w_start'][j], w_obs <= lm['w_end'][j] ]))[0]
            if len(mask) > 0:
                ind_obs.append(mask)
            else:
                check[j] = True
                if debug:
                    print(F"No observed points fall within {lm['w_start'][j]} -- {lm['w_end'][j]} AA")



        return lm['w_center'], ind_obs, ind_model, check


class spectrum(object):
    def __init__(self, w, f, res):
        self.lam = np.array(w)
        # by default normalised flux
        self.flux = np.array(f)

        self.R = res
        # determine step of the datapoints
        self.lam_step = np.median(self.lam[1:] - self.lam[:-1])


    def convolve_resolution(self, R_new):
        print(F"Convolving spectrum from R={self.R} to R={R_new}...")

        d_lam = (np.mean(self.lam)/R_new)
        sigma = d_lam / (2.0 * np.sqrt(2. * np.log(2.)))
        kernel = convolution.Gaussian1DKernel(sigma/self.lam_step)
        self.flux =  convolution.convolve(self.flux, kernel)
        self.R = R_new

    def convolve_rotation(self, Vrot, debug):
        """
        Convolve with rotational profile
        Identical to faltbon (tested on O triplet)
        Following the recipe from Gray's 'The observation and analysis
        of Stellar Photospheres'
        input:
        Vrot (float) V*sin(i) in km/s
        """
        self.Vrot = Vrot
        if self.Vrot < 0:
            print(F"Rotational velocity <0: {self.Vrot}. Can only be positive (km/s). Stopped.")
            exit(1)
        if self.Vrot == 0: # do nothing
            pass
        elif not np.isnan(self.Vrot):
            spec_deltaV = self.lam_step/np.mean(self.lam) * const.c.to('km/s').value
            if (spec_deltaV) > self.Vrot:
                if debug:
                    print(F"WARNING: resolution of model spectra {spec_deltaV} is less than Vrot={self.Vrot}. No convolution will be done, Vrot=0.")
                self.Vrot = 0
            else:
                # FWHM: km/s --> A --> step
                # assumes constant step along the whole wavelength range
                fwhm = self.Vrot * np.mean(self.lam) / const.c.to('km/s').value / self.lam_step

                # kernel should always have odd size  along all axis
                w, f = self.lam, self.flux
                odd = False
                if len(w) % 2 == 0:
                    w = np.hstack((w, w[-1]))
                    f = np.hstack((f, f[-1]))
                    odd = True

                rot_kernel = convolution.Model1DKernel(rotation(fwhm), x_size=len(w) )
                f = convolution.convolve(f, rot_kernel, fill_value=1)

                if odd:
                    self.flux = f[:-1]
                else:
                    self.flux = f
        else:
            print(F"Unexpected Vrot={self.Vrot} [km/s]. Stopped.")
            exit(1)

    def convolve_macroturbulence(self, Vmac, debug=False):
        """
        Convolve with macro-turbulence (radial-tangential profile)
        Identical to faltbon (tested on O triplet)
        Following the recipe from Gray's 'The observation and analysis
        of Stellar Photospheres' and Gray, 1978
        input:
        Vmac (float) in km/s
        """
        self.Vmac = Vmac

        if self.Vmac == 0: # do nothing
            pass
        elif self.Vmac < 0:
            print(F"Macroturbulence <0: {self.Vmac}. Can only be positive (km/s). Stopped.")
            exit(1)
        elif not np.isnan(self.Vmac):
            spec_deltaV = self.lam_step/np.mean(self.lam) * const.c.to('km/s').value
            if (spec_deltaV) > self.Vmac:
                if debug:
                    print(F"WARNING: resolution of model spectra {spec_deltaV} is less than Vmac={self.Vmac}. No convolution will be done, Vmac = 0.")
                self.Vmac = 0
            else:
                # FWHM: km/s --> A --> step
                # assumes constant step along the whole wavelength range
                fwhm = self.Vmac * np.mean(self.lam) / const.c.to('km/s').value / self.lam_step

                # kernel should always have odd size along all axis
                w, f = self.lam, self.flux
                odd = False
                if len(w) % 2 == 0:
                    w = np.hstack((w, w[-1]))
                    f = np.hstack((f, f[-1]))
                    odd = True
                # # TODO: size of the kernel needs to be estimated from the FWHM
                macro_kernel = convolution.Model1DKernel( rad_tang(fwhm), x_size=len(w) )

                f = convolution.convolve(f, macro_kernel, fill_value=1)

                if odd:
                    self.flux = f[:-1]
                else:
                    self.flux = f
        else:
            print(F"Unexpected Vmac={self.Vmac} [km/s]. Stopped.")
            exit(1)


    def copy(self):
        return deepcopy(self)




class rotation(Fittable1DModel):
    fwhm = Parameter(default=0)
    # FWHM is v*sin i in wavelength units
    @staticmethod
    def evaluate(x, fwhm):
        f = np.zeros(len(x))
        mask = np.where( 1-(x/fwhm)**2  >=0 )
        eps=1.-0.3*x[mask]/5000.
        f[mask] =  2*(1-eps)*np.sqrt(1-(x[mask]/fwhm)**2)+np.pi/2 * eps * (1.-(x[mask]/fwhm)**2) / (np.pi*fwhm*(1-eps/3))
        return f

class rad_tang(Fittable1DModel):
    fwhm = Parameter(default=0)

    @staticmethod
    def evaluate(x, fwhm):
        # Gray, 'Turbulence in stellar atmospheres', 1978
        rtf = [1.128,.939,.773,.628,.504,.399,.312,.240,.182,.133, \
       .101,.070,.052,.037,.024,.017,.012,.010,.009,.007,.006, \
       .005,.004,.004,.003,.003,.002,.002,.002,.002,.001,.001, \
       .001,.001,.001,.001,.000,.000,.000,.000 ]
        rtf_x = np.arange(len(rtf))
        # step of macroturbulence function (rtf)
        delta=fwhm*1.433/10.

        x = np.abs(x /delta)
        rtf_inter = interpolate.interp1d(rtf_x, rtf, fill_value='extrapolate')

        return rtf_inter(x)
