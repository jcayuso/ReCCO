import numpy as np
import healpy as hp
from scipy import interpolate

def model_selection(tag):
	if 'SO' in tag:
		return SOlike_noise_spec
	elif 'Planck' in tag:
		return Plancklike_noise_spec
	else:  # If unrecognized, zero noise at all frequencies
		print('Instrumental noise specified but no noise model found. Bypassing instrumental noise.')
		return lambda F, ell_sparse : np.zeros((len(F), ell_sparse.max()+1))

def SOlike_noise_spec(freq, ell_sparse):
	lmax = ell_sparse.max()
	if not np.iterable(freq):
		freq = np.array([freq])
	# Linear interpolation function to extrapolate SO baseline noise parameters
	# to arbitrary frequencies. Most reasonable curves come when
	# interpolation is done in log space
	SO_FIT_FREQS = np.array([27, 39, 93, 145, 225, 280])
	SO_FIT_FWHMS = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
	SO_FIT_NOISE_BASELINE = np.array([71, 36, 8.0, 10, 22, 54])
	SO_FIT_NOISE_RED = np.array([100, 39, 230, 1500, 17000, 31000])
	baseline_func = interpolate.interp1d(np.log(SO_FIT_FREQS), np.log(SO_FIT_NOISE_BASELINE), kind='linear', fill_value="extrapolate")  # Also possible to use cubic
	red_func = interpolate.interp1d(np.log(SO_FIT_FREQS), np.log(SO_FIT_NOISE_RED), fill_value="extrapolate")
	fwhm_func = interpolate.interp1d(np.log(SO_FIT_FREQS), np.log(SO_FIT_FWHMS), fill_value="extrapolate")
	# SHORTHAND FUNCTIONS
	arcmin_to_rad = lambda arcmin : arcmin * (np.pi/(180*60))  # arcminute to radians shortcut
	makelin = lambda interp_func, input_freq : np.exp(interp_func(np.log(input_freq)))  # Log to linear space shortcut
	# FIXED NOISE PARAMETERS AS IN https://arxiv.org/abs/1808.07445
	# Noise spectra constants
	a_knee = -3.5
	l_knee = 1000.
	# Integration time to scale Nred
	FSKY = 0.4
	LAT_deg_tot = ((4*np.pi)*FSKY) * (180 / np.pi)
	LAT_deg_fov = 7.8
	mission_duration = 5 * 365.25 * 86400
	integration_time = (LAT_deg_fov/LAT_deg_tot) * mission_duration
	# Compute noise spectrum at frequency
	white_noise = (makelin(baseline_func, freq) * arcmin_to_rad(1.)) ** 2  # uK-arcmin to uK-rad (a unit equiv to uK) then square to become to uK^2
	pink_noise = makelin(red_func, freq) / integration_time
	smoothing_beam = [hp.gauss_beam(arcmin_to_rad(makelin(fwhm_func, f)), lmax=lmax) for f in freq]
	N_white = white_noise[:,np.newaxis] / np.array(smoothing_beam)[:,ell_sparse]**2
	with np.errstate(divide='ignore'):  # Suppress divide by zero warning
		N_red = pink_noise[:,np.newaxis] * (ell_sparse / l_knee) ** a_knee
	Nl = N_red + N_white
	#Nl[:, 0] = 0  # Monpole removal -- No longer needed since ell_sparse won't include l=0?
	return Nl

def Plancklike_noise_spec(freq,ell_sparse):
	lmax = ell_sparse.max()
	if not np.iterable(freq):
		freq = np.array([freq])
	# Values taken from https://arxiv.org/abs/1911.05717
	# which did not include 857 GHz. 857 GHz obtained from
	# power spectrum of Planck 857 GHz noise map
	BASELINE = np.array([195.1, 226.1, 199.1, 77.4, 33.0, 46.8, 153.6, 818.2, 40090.7])
	# Fit for final fwhm
	FIT_FREQS = np.array([30, 44, 70, 100, 143, 217, 353, 545])
	FIT_FWHM = np.array([32.408, 27.100, 13.315, 9.69, 7.30, 5.02, 4.94, 4.83])
	fwhm857 = np.exp(interpolate.interp1d(np.log(FIT_FREQS),np.log(FIT_FWHM), kind='linear', fill_value='extrapolate')(857))
	FREQS = np.append(FIT_FREQS, 857)
	FWHM = np.append(FIT_FWHM, fwhm857)
	arcmin_to_rad = lambda arcmin : arcmin * (np.pi/(180*60))  # arcminute to radians shortcut
	white_noise = (BASELINE * arcmin_to_rad(1.)) ** 2  # These values should match the Planck maps
	N_white = {}
	for f in np.arange(FREQS.size):
		smoothing_beam = hp.gauss_beam(arcmin_to_rad(FWHM[f]), lmax=lmax)
		smoothing_beam[np.where(smoothing_beam<1e-100)] = smoothing_beam[np.where(smoothing_beam>=1e-100)][-1]  # avoid division by zero
		N_white[FREQS[f]] = white_noise[f,np.newaxis] / smoothing_beam[ell_sparse]**2
	Nl = np.zeros((freq.size, ell_sparse.size))
	for f, user_freq in enumerate(freq):
		Nl[f,:] = N_white[int(user_freq)]
	
	return Nl