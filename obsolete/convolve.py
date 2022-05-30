# adapted from SAPP-ss by Jeffrey Gerber, Apr. 2021
import numpy as np
import os
import time

def conv_res(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	with open('original.txt', 'w') as f:
		for j in range(len(wave)):
			#faltbon needs three columns of data, but only cares about the first two
			f.write(F"{wave[j]:f} {flux[j]:f} -9999 \n")
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 2; } | ./faltbon" % fwhm)
	if not os.path.isfile('./convolve.txt'):
		return np.nan, np.nan
	wave_conv, flux_conv, _ = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	os.remove('./convolve.txt')
	return wave_conv, flux_conv

def conv_macroturbulence(wave, flux, velocity):
	if velocity == 0.0:
		return wave, flux
	else:
		with open('original.txt', 'w') as f:
			for j in range(len(wave)):
				f.write(F"{wave[j]:f} {flux[j]:f} -9999 \n")
		os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 3; } | ./faltbon > log_conv_vmac.txt" % -velocity)
		wave_conv, flux_conv, _ = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
		os.remove('./convolve.txt')
		return wave_conv, flux_conv

def conv_rotation(wave, flux, velocity):
	# faltbon doesn't handle rotational velocity == 0
	# if gives nans as an output
	if np.abs(velocity) < 0.005:
		return wave, flux
	else:
		#create a file to feed to ./faltbon of spectrum that needs convolving
		with open('original.txt', 'w') as f:
			for j in range(len(wave)):
				f.write(F"{wave[j]:f} {flux[j]:f} {-9999} \n")
		os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 4; } | ./faltbon > log_conv_vrot.txt" % -velocity)
		wave_conv, flux_conv, _ = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
		os.remove('./convolve.txt')
		return wave_conv, flux_conv
