# external
import os
from sys import argv
import shutil
import subprocess
import datetime
import numpy as np
import pickle
from astropy import constants as const
import glob
from scipy import interpolate
# local
import read_config
from atmos_package import read_atmos_marcs, model_atmosphere
from read_nlte import grid_to_ts
from run_ts import *
from observations import *


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)

def stats(O, E):
    # input:
    # O -- array of observed values
    # E -- array of expected values
    chi2 = np.sum( (O-E)**2 / E ) / (len(O)-1)
    return chi2

## TODO: only compute babsma for a model atmosphere, not abundance

"""
Routine to fit observations with TS
For now (May 2020)
only fitting abundance of one element (in 1D, <3D>, LTE, NLTE) is possible
"""
if __name__ == '__main__':
    precomputed_spectra = False
    if len(argv) > 1:
        conf_file = argv[1]
        if len(argv) > 2:
            output_spectra = argv[2]
            precomputed_spectra = True
    else:
        conf_file = './config.txt'
    set = read_config.setup(file = conf_file)
    if not set.fitting:
        print("Set fitting = 1 in the configuration file. Stopped.")
        exit(1)
    spec_obs = set.observed_spectrum
    """ Make directory to save output spectra """
    today = datetime.date.today().strftime("%b-%d-%Y")
    set.output_dir = set.cwd + F"/spectra-{set.element}-{today}/"
    mkdir(set.output_dir)

    results_fit = {}
    if set.nlte:
        nlte_str ='nonLTE'
    else:
        nlte_str ='LTE'
    output_res = open(set.cwd + F"/fit_results_{set.element.capitalize()}_{nlte_str}.dat", 'w')
    output_res.write('# model atmosphere, central wavelenght (if nan, all lines in the linemask were used), best fit abundance, Vmac km/s, Vrot km/s, chi2 of the best fit, Resolution, SNR \n')
    snr = 0.

    """
    Make a segment file to speed up TS
    NOTE! Very important: segments can not overlap
    overlapping segments lead to over-estimated abundances!
    """
    with open(set.cwd + '/segments.dat', 'w') as f:
        set.ts_input.update({'SEGMENTSFILE':set.cwd + '/segments.dat'})

        lm = set.line_masks[set.element.lower()]
        end_seg = 0.0
        for i in range(len( lm['w_center'] )):
            if lm['w_center'][i] - set.line_offset > end_seg:
                start_seg = lm['w_center'][i] - set.line_offset
                end_seg = lm['w_center'][i] + set.line_offset
                for j in range(len( lm['w_center'][i:] )):
                    if lm['w_center'][i+j] - set.line_offset < end_seg:
                        end_seg = lm['w_center'][i+j] + set.line_offset

                if (start_seg) > lm['w_start'][i] or (end_seg) < lm['w_end'][i]:
                    if set.debug:
                        print(F"WARNING: you requested offset of {set.line_offset}, so current segment is:")
                        print(F"{start_seg:.3f} -- {end_seg:.3f} AA")
                        print("but the linemask states: ")
                        print(F"center: {lm['w_center'][i]}, start: {lm['w_start'][i]}, end: {lm['w_end'][i]} AA")
                        print("Setting segment to match the linemask.")
                        print()
                    start_seg = lm['w_start'][i]
                    end_seg = lm['w_end'][i]

                f.write(F"{start_seg:.3f}   {end_seg:.3f} \n")




    """ Loop over requested model atmospheres """
    for atm_file in set.atmos_list:
        print()

        """ First read model atmosphere for parameteres """
        atmos = model_atmosphere(file = atm_file, format=set.atmos_format)
        atmos.path = atm_file


        results_fit.update( { atmos.id : { 'abund':[], 'line':[], 'chi2':[], \
                            'vmac':[], 'vrot':[], 'resolution':[], 'snr':[] } } )
        print(F"{set.element}, {atmos.id}")

        # compute opac file with babsma
        compute_with_ts(set, atmos, np.nan, routine='babsma')
        for abund in set.abund_list:
#            if set.debug:
            print(F"running TS with A({set.element})={abund:4.2f}")


            """ Prepare NLTE input if requested """
            if set.nlte:
                if set.debug:
                    print("using NTLE grid")
                set = prep_nlte_input(set, atmos, abund)

            """ Run TS """
            set, w, f, abs_f = compute_with_ts(set, atmos, abund, routine='bsyn')
            spec_mod =  spectrum(w, f, res=np.nan)
            # convolve model spectrum with resolution of an observed one
            spec_mod.convolve_resolution(spec_obs.R)

            """ Select observed and model spectrum within line masks """
            w_centers, masks_obs, masks_model, empty = select_within_linemask(set.line_masks, set.element.lower(),   spec_obs.lam, spec_mod.lam, debug = set.debug)
            if empty.all():
                print(F"No data points fall within linemask. Stopped")
                exit(1)
            else:
                if not set.line_by_line:
                    # if not line by line, merge all segments together:
                    w_centers = [np.nan]
                    masks_obs_new = []
                    masks_model_new = []
                    for m in masks_obs:
                        masks_obs_new.extend(m)
                    masks_obs = [masks_obs_new]
                    for m in masks_model:
                        masks_model_new.extend(m)
                    masks_model = [masks_model_new]


                """
                Convolve the whole (chunck of) the spectrum computed with TS

                Loop over Vmac values
                If no fitting requested, set.vmac == [np.nan]
                Same with Vrot
                """

                # loop over Vmac
                for vel_mac in set.vmac:
                    # to avoid duplicate convolution
                    spec_mod_vmac = spec_mod.copy()
                    spec_mod_vmac.convolve_macroturbulence(vel_mac, set.debug)
                    vel_mac  = spec_mod_vmac.Vmac

                    # loop over Vrot
                    for vel_rot in set.vrot:
                        # to avoid duplicate convolution
                        spec_mod_vmac_vrot = spec_mod_vmac.copy()
                        spec_mod_vmac_vrot.convolve_rotation(vel_rot, set.debug)
                        vel_rot = spec_mod_vmac_vrot.Vrot


                        """ Compare observed and model spectra """
                        for j in range(len(w_centers)):
                            if not empty[j]:
                                if set.debug:
                                    print(w_centers[j], F"empty={empty[j]}")
                                ### Compute chi2 only within the linemask
                                spec_obs_seg = spectrum(spec_obs.lam[masks_obs[j]], spec_obs.flux[masks_obs[j]], res=spec_obs.R)
                                spec_model_seg = spectrum(spec_mod_vmac_vrot.lam[masks_model[j]], spec_mod_vmac_vrot.flux[masks_model[j]], res=spec_mod_vmac_vrot.R)
                                if len(spec_model_seg.lam) < 10:
                                    print(F"Model spectrum has only {len(spec_model_seg.lam)} points around the line {w_centers[j]}.")
                                    print(F"Set higher resolution (currently R={set.resolution} or extend the segment. Stopped.")
                                    exit(1)

                                ### INTERPOLATE MODEL SPECTRA TO OBSERVED WAVELENGTHS
                                f_interp = interpolate.interp1d(spec_model_seg.lam, spec_model_seg.flux, kind='linear', fill_value = 'extrapolate')

                                chi2 = stats( spec_obs_seg.flux, f_interp(spec_obs_seg.lam) )

                                if set.debug:
                                    # with open(set.cwd + F"/spec_{set.element}_{atmos.id}_A_{abund:.2f}_Vmac_{vel_mac:.2f}_Vrot_{vel_rot:.2f}.txt", 'w') as f:
                                    #     for ii in range(len(spec_model_vmac_vrot.lam)):
                                    #         f.write(F"{spec_model_vmac_vrot.lam[ii]} {spec_model_vmac_vrot.flux[ii]}\n")
                                    print(F"{atmos.id}, nonLTE={set.nlte}, line at {w_centers[j]:.2f} AA, Vmac = {vel_mac:.2f} km/s, Vrot = {vel_rot:.2f} km/s, chi2 = {chi2:.5E}")

                                results_fit[atmos.id]['abund'].append(abund)
                                results_fit[atmos.id]['line'].append(w_centers[j])
                                results_fit[atmos.id]['chi2'].append(chi2)
                                results_fit[atmos.id]['snr'].append(snr)
                                results_fit[atmos.id]['resolution'].append(spec_obs_seg.R)
                                results_fit[atmos.id]['vmac'].append(vel_mac)
                                results_fit[atmos.id]['vrot'].append(vel_rot)

                            else:
                                pass

        compute_with_ts(set, atmos, np.nan, routine='clean')

        """ Print results of fitting """
        for k in results_fit[atmos.id].keys():
            results_fit[atmos.id][k] = np.array(results_fit[atmos.id][k])

        print()

        best_fit_abund = []
        # line by line
        if not np.array([np.isnan(x) for x in results_fit[atmos.id]['line']]).all():
            for w in np.sort(np.unique(results_fit[atmos.id]['line'])):
                for r in np.sort(np.unique( results_fit[atmos.id]['resolution'] )):
                    for sn in np.sort(np.unique( results_fit[atmos.id]['snr'] )):
                        mask = np.logical_and.reduce( [results_fit[atmos.id]['line'] == w, \
                                results_fit[atmos.id]['snr'] == sn, results_fit[atmos.id]['resolution'] == r] )
                        c2 = results_fit[atmos.id]['chi2'][mask]
                        ab = results_fit[atmos.id]['abund'][mask]
                        vm = results_fit[atmos.id]['vmac'][mask]
                        vr = results_fit[atmos.id]['vrot'][mask]

                        pos = np.where( c2 == min(c2))[0][0]
                        best_fit_abund.append(ab[pos])
                        if set.debug:
                            print(F"line at {w:10.3f} AA: A({set.element.capitalize()})_best={ab[pos]:4.2f}, chi2={c2[pos]:4.3E}, \
R={r:6.0f}, SNR={sn:3.0E}, Vmac={vm[pos]:.2f}, Vrot={vr[pos]:.2f}")
                        output_res.write(F"{atmos.id} {w:10.3f} {ab[pos]:6.3f} {vm[pos]:.2f} {vr[pos]:.2f} {c2[pos]:4.3E} {r:6.0f} {sn:3.0E} \n")
        # all lines at the same time
        else:
            print("From all lines in the linemask:")
            for r in np.sort(np.unique( results_fit[atmos.id]['resolution'] )):
                for sn in np.sort(np.unique( results_fit[atmos.id]['snr'] )):
                    mask = np.logical_and.reduce( [results_fit[atmos.id]['snr'] == sn, results_fit[atmos.id]['resolution'] == r] )
                    c2 = results_fit[atmos.id]['chi2'][mask]
                    ab = results_fit[atmos.id]['abund'][mask]
                    vm = results_fit[atmos.id]['vmac'][mask]
                    vr = results_fit[atmos.id]['vrot'][mask]

                    pos = np.where( c2 == min(c2))[0][0]
                    best_fit_abund.append(ab[pos])
                    if set.debug:
                        print(F"A({set.element.capitalize()})_best={ab[pos]:4.2f}, chi2={c2[pos]:4.3E}, \
R={r:6.0f}, SNR={sn:3.0E}, Vmac={vm[pos]:.2f}, Vrot={vr[pos]:.2f}")
                    output_res.write(F"{atmos.id}  nan {ab[pos]:6.3f} {vm[pos]:.2f} {vr[pos]:.2f} {c2[pos]:4.3E} {r:6.0f} {sn:3.0E}\n")

        #""" Delete all the spectra except the one(s) corresponding to the best fit """
        #if not set.save_all_spec:
        #    for spec_file in glob.glob(set.output_dir + '/spec_*'):
        #        ab  = float(spec_file.split('_')[-1])
        #        if atmos.id in spec_file and not ab  in best_fit_abund:
        #            os.remove(spec_file)
        #        else:
        #            print(ab)

    output_res.close()
    exit(0)
