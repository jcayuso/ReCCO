# ReCCO

Code for calculation of CMB secondaries (kSZ,tSZ,lensing,moving lens) and CIB, cross-correlations
with galaxy surveys and reconstruction of radial velocity and transverse velocity
on the lightcone.

spectra.py generates Cls
estim.py  processes the Cls to calculate biases and noise to reconstructed 
velocity fields and provides pipeline for Gaussian simulations.

Future update:
-user guide
-fully integrate remote dipole and remote quadrupole reconstructions
(basic functions in remote_spectra.py)