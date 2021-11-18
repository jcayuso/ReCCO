#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:56:42 2020

@author: jcayuso
"""

import numpy as np
import power 
import config as conf
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import time
import redshifts
from math import lgamma
import pyfftlog
import camb
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
import argparse, sys
import common as c
import loginterp
import noisegen as ng
import gc
#########################################################
####################  SOME BASIC VALUES AND CONFIGURATION
#########################################################

#Some parallelization is used at some points. If you want to keep some cores of your computer free, set below

free_cores = 1


basic_conf = c.get_basic_conf(conf)


G_SI = 6.674e-11
mProton_SI = 1.673e-27
H100_SI = 3.241e-18
thompson_SI = 6.6524e-29
meterToMegaparsec = 3.241e-23


zb = redshifts.binning(conf_module = conf) 
csm = zb.csm


#-----Average electron density today but with Helium II reionization at z<3. Units: 1/meter**3

#from https://github.com/msyriac/orphics/blob/master/orphics/cosmology.py
def ne0z(z):
    chi = 0.86
    me = 1.14
    gasfrac = 0.9
    omgh2 = gasfrac* conf.ombh2
    ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me                   
    return ne0_SI


# frequencies

def freqs():   
    return conf.cleaning_frequencies[conf.cleaning_mode]
       
# changes Jy to microK
def f_tSZ(nu):
    #returns spectral function of tSZ at nu
    h =  6.626e-34 #plancks constant in SI
    k =   1.38e-23 #boltzmann constant in SI
    T = conf.T_CMB*1e-6 #temperature of BM
    x = h * nu*1e9/(k *T)

    return x/np.tanh(x/2)-4

def jy_to_uK(nu):
    #jy to microK
    
    frequencies=[100,143,217,353,545,857]
    
    if nu in frequencies:
        conversions=[244.1,371.74,483.69,287.45,58.04,2.27]
        return 1/conversions[frequencies.index(nu)]
    
    else:
        x=nu/56.8
        conversionfactor=1.05e3*(np.exp(x)-1)**2*np.exp(-x)*(nu/100)**-4
        return 1/(1/conversionfactor*1e6)
    
def get_tau():
    
    tau = np.zeros(conf.N_bins)
    for i in np.arange(conf.N_bins):
        chis  = np.linspace(zb.zbins_chi[i], zb.zbins_chi[i+1], 200)
        zs = csm.z_from_chi(chis)
        a = 1.0/(1.0+zs)
        tau[i] = np.trapz(thompson_SI*ne0z(zs)*a**(-2.)/meterToMegaparsec,chis)
        
    return tau

####################################################
####################  SOME UTILITY FUNCTIONS
####################################################

       
def get_n_cores() :
    cpu_count = mp.cpu_count()
    if cpu_count > 8 :
        # probably on a cluster node, use the whole thing
        return cpu_count-free_cores
    else :
        # probably something local, save one core
        return max(1, int( cpu_count - free_cores ))
    
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

    
####################################################
####################  FAST FOURIER TRANSFORM
####################################################



def growth(tag,fq):
    
    pk = c.load(basic_conf, 'p_linear_'+c.retag(tag)+c.retag(tag)+'_f1='+str(fq)+'_f2 ='+str(fq), dir_base = 'pks')
    p = pk(conf.ks_hm,conf.zs_hm)
    G = np.sqrt(p/((p[0,:])[np.newaxis,:]))
    return interp2d(conf.ks_hm,conf.zs_hm,G , kind = 'linear',bounds_error=False,fill_value=0.0)

def eff_growth(tag, fq, b):
     
    zc = zb.zbins_zcentral[b]
    geff =  growth(tag, fq)(conf.ks_hm[0],conf.zs_hm)[:,0]/(growth(tag, fq)(conf.ks_hm[0],zc)[0])
    return interp1d(conf.zs_hm, geff, kind = 'linear',bounds_error=False,fill_value="extrapolate")

# def eff_growth(tag, fq, b):
    
#     geff =  growth(tag, fq)(conf.ks_hm[0],conf.zs_hm)[:,0]
#     return interp1d(conf.zs_hm, geff, kind = 'linear',bounds_error=False,fill_value=0.0)


def fftlog_weights(tag, fq, b,chis):
                          
    chi_1 = zb.zbins_chi[b]
    chi_2 = zb.zbins_chi[b+1]
                        
    if tag in ['e','m']:
        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        return  eff_growth(tag, fq, b)(zs)*cut/zb.deltachi/np.sqrt(chis)

    if tag == 'tSZ':
        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < csm.chi_from_z(0.01) , 0,1)
        return  cut*eff_growth(tag, fq, b)(zs)/conf.T_CMB/np.sqrt(chis)
        

    if tag == 'CIB':
        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < csm.chi_from_z(0.01) , 0,1)
        return  cut*eff_growth(tag, fq, b)(zs)*jy_to_uK(fq)/conf.T_CMB/np.sqrt(chis)       
    
    if tag == 'g':
        zs = csm.z_from_chi(chis)
        w  = zb.get_galaxy_window(b)(zs)
        wmax = np.max(w)
        cut  = np.where(w< np.max(wmax)*1e-4, 0,1)
        return eff_growth(tag, fq, b)(zs)*w*cut/np.sqrt(chis)
    
    if tag == 'taud':
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        zs = csm.z_from_chi(chis)
        aas = 1.0/(1.0 + zs)
        kappa = thompson_SI*ne0z(zs)*aas**(-2.)/meterToMegaparsec
        return eff_growth('e',fq, b)(zs)*kappa*cut/zb.deltachi/np.sqrt(chis)
             
    if tag == 'vr' :
        
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        zs = csm.z_from_chi(chis)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)
        f = csm.f_growth(zs)
        
        return eff_growth('m', fq, b)(zs)*a*H*f*cut/zb.deltachi/np.sqrt(chis)
    
    if tag == 'vt' :
        
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        zs = csm.z_from_chi(chis)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)
        f = csm.f_growth(zs)
        
        return eff_growth('m', fq, b)(zs)*a*H*f*cut/chis/zb.deltachi/np.sqrt(chis)
    
    if tag == 'isw_lin' :
        
        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < csm.chi_from_z(0.15) , 0,1)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)
        z_grad = np.logspace(-2,np.log10(conf.z_max+1),150)
        grad = np.gradient(eff_growth('m', fq, b)(z_grad),z_grad)
        dgeff = interp1d(z_grad, grad, kind = 'linear',bounds_error=False,fill_value=0.0)(zs)
        H0 = csm.H_z(0.001)
        Omega = conf.Omega_b+conf.Omega_c
        
        return -3.0*Omega*(H0**2)*H*(dgeff/a+eff_growth('m', fq, b)(zs))*cut/np.sqrt(chis)

    if tag == 'ml' :
        
        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)
                
        Omega = conf.Omega_b+conf.Omega_c
        H0 = csm.H_z(0.001)
        Dpsi_Tk = 3.0 * cut * Omega * (H0**2) * eff_growth('m', fq, b)(zs)/a

        return Dpsi_Tk*cut/np.sqrt(chis)/chis/zb.deltachi
    
    if tag == 'lensing' :

        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < csm.chi_from_z(0.02) , 0,1)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)

        chi_s = csm.chi_from_z(1090)

        Omega = conf.Omega_b+conf.Omega_c
        H0 = csm.H_z(0.001)
        Dpsi_Tk = 3.0 * cut * Omega * (H0**2) * eff_growth('m', fq, b)(zs)/a

        return Dpsi_Tk*cut/np.sqrt(chis)*((chi_s-chis)/(chi_s*chis))
    
def fftlog_integral(tag, fq, b, ell):
    
    N = int(30000)
    chi_0 = (1.0/conf.ks_hm[-1])*1.05
    #chi_e = csm.chi_from_z(1000)
    chi_e = csm.chi_from_z(conf.z_max+2.99)
    chis = np.logspace(np.log(chi_0), np.log(chi_e), num = N, base = np.exp(1))
    dlnr = np.log(chi_e/chi_0)/(N-1)
    jc = (N+1)/2
    rc = chi_0*np.exp(jc*dlnr)
    k = chis/rc**2
    kr = 1
    kr, xsave = pyfftlog.pyfftlog.fhti(N, ell+1/2, dlnr, q=0, kr=kr, kropt=0)
    FW = c.load(basic_conf, 'fweight_'+tag+'_fq='+str(fq)+'_b='+str(b), dir_base = 'ffts')

    return k, pyfftlog.pyfftlog.fht(FW, xsave, tdir=1)


def ffts_tracer(tag,fq, b,ell):
    
    k = fftlog_integral(tag, fq, 0, ell)[0]

    if tag == 'vr': 
        A = fftlog_integral(tag, fq, b,ell-1)
        B = fftlog_integral(tag, fq, b,ell+1)
        return (A[1]*ell/(2*ell+1)-B[1]*(ell+1)/(2*ell+1))/k*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.05, 0, 1)
    
    elif tag == 'vt':
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.5, 0, 1)/k**2.
 
    elif tag =='tSZ' or tag == 'CIB':
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.05, 0, 1)
    
    elif tag == 'isw_lin':
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.05, 0, 1)/k**2
       
    elif tag == 'ml':
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.5, 0, 1)/k**2
    
    elif tag == 'lensing':
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.5, 0, 1)/k**2
    else:
        A = fftlog_integral(tag, fq, b, ell)
        return A[1]*np.where(k < ell/zb.zbins_chi[conf.N_bins]*0.05, 0, 1)
  
    
def save_fft_weights(tag,fq):
    
    N = int(30000)
    chi_0 = (1.0/conf.ks_hm[-1])*1.05
    #chi_e = csm.chi_from_z(1000)
    chi_e = csm.chi_from_z(conf.z_max+2.99)
    chis = np.logspace(np.log(chi_0), np.log(chi_e), num = N, base = np.exp(1))
    
    for i in np.arange(zb.nbin_tag(tag)):
        FW = fftlog_weights(tag, fq, i, chis)
        c.dump(basic_conf, FW, 'fweight_'+tag+'_fq='+str(fq)+'_b='+str(i), dir_base = 'ffts')
            
    pass
    
def save_fft(tag,fq,b,ell):
    
    F = ffts_tracer(tag, fq, b, ell)
    #we will rewrite for different ells
    c.dump(basic_conf, F, 'f_'+tag+'_fq='+str(fq)+'_b='+str(b), dir_base = 'ffts')
    pass

    
def get_ffts(tag,fq,ell):
        
    if zb.nbin_tag(tag) <= get_n_cores()/2:
        for i in np.arange(zb.nbin_tag(tag)):
            save_fft(tag, fq, i, ell)   
    else:   
        pool = mp.Pool(get_n_cores())
        pool.starmap(save_fft, [(tag, fq, i, ell) for i in np.arange(zb.nbin_tag(tag))])
        pool.close()       
            
    pass
    

####################################################
####################  INTEGRATIONS FOR CLS
####################################################

##########  Limber approximation for function F(k,z)

def limber(F,chis,ell):
    return np.diagonal(np.flip(F((ell+1.0/2.0)/chis,csm.z_from_chi(chis)),axis=1))  #need to flip because of how pks are computed
    
##########  Weights for Limber approximation

def limber_weight(tag, fq, b, ell ,chis):
    
    chi_1 = zb.zbins_chi[b]
    chi_2 = zb.zbins_chi[b+1]
                        
    if tag in ['e','m']:
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        return  cut/zb.deltachi
    
    if tag in ['tSZ']:
        return  np.ones(len(chis))/conf.T_CMB
    
    if tag in ['CIB']:
        return  np.ones(len(chis))*jy_to_uK(fq)/conf.T_CMB
    
    if tag == 'g':
        
        zs = csm.z_from_chi(chis)
        w  = zb.get_galaxy_window(b)(zs)
        wmax = np.max(w)
        cut  = np.where(w< np.max(wmax)*1e-4, 0,1)
        return w*cut
        
    
    if tag == 'taud':
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        zs = csm.z_from_chi(chis)
        aas = 1.0/(1.0 + zs)
        kappa = thompson_SI*ne0z(zs)*aas**(-2.)/meterToMegaparsec
        return kappa*cut/zb.deltachi
             
    if tag == 'vr' :
        
        lf_a = (ell+1.0/2.0)/(ell-1.0/2.0)
        cut_a  = np.where( chis < chi_1*lf_a , 0,1)*np.where( chis > chi_2*lf_a , 0,1)
        zs_a = csm.z_from_chi(chis/lf_a)        
        a_a = 1.0/(1.0 + zs_a)
        H_a = csm.H_z(zs_a)
        f_a = csm.f_growth(zs_a)
        
        term_a = a_a*H_a*f_a*cut_a*ell/(2.0*ell+1.0)/np.sqrt(2.0)/zb.deltachi
        
        lf_b = (ell+1.0/2.0)/(ell+3.0/2.0)
        cut_b  = np.where( chis < chi_1*lf_b , 0,1)*np.where( chis > chi_2*lf_b , 0,1)
        zs_b = csm.z_from_chi(chis/lf_b)        
        a_b = 1.0/(1.0 + zs_b)
        H_b = csm.H_z(zs_b)
        f_b = csm.f_growth(zs_b)
        
        term_b = a_b*H_b*f_b*cut_b*(ell+1)/(2.0*ell+1.0)/np.sqrt(2.0)/zb.deltachi
        
        return (term_a-term_b)*chis/(ell+0.5)
    
    if tag == 'vt' :
    
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        z_b = csm.z_from_chi(chis)

        a_b  = 1.0/(1.0 + z_b)
        H_b  = csm.H_z(z_b)
        H0 = csm.H_z(0.001)
        Omega = conf.Omega_b+conf.Omega_c
        k = (ell+1.0/2.0)/chis
        f_b = csm.f_growth(z_b)

        return a_b*H_b*f_b*cut/zb.deltachi/k**2./chis
    
    if tag == 'isw_lin':
        
        chis_interp = np.linspace(csm.chi_from_z(0.05),csm.chi_from_z(conf.z_max+0.9),400)
        z_interp = csm.z_from_chi(chis_interp)  
  
        z_grad = np.logspace(-2,np.log10(conf.z_max+1),2000)
        pk = c.load(basic_conf, 'p_'+c.retag(tag)+c.retag(tag)+'_f1='+str(fq)+'_f2 ='+str(fq), dir_base = 'pks')
        p = pk(conf.ks_hm,z_grad)
        p_grad = interp2d(conf.ks_hm,z_grad,np.gradient(p,z_grad,axis=0), kind = 'quintic',bounds_error=False,fill_value=0.0)
        
        p_grad_limb = limber(p_grad,chis_interp,ell)
        pmm_limb = limber(pk,chis_interp,ell) 
        
        a  = 1.0/(1.0 + z_interp)
        H  = csm.H_z(z_interp)
        H0 = csm.H_z(0.001)
        Omega = conf.Omega_b+conf.Omega_c
        k = (ell+1.0/2.0)/chis_interp
        
        I = interp1d(chis_interp,-3.0*Omega*(H0/k)**2*H*(p_grad_limb/2.0/pmm_limb/a+1.0), kind = 'linear',bounds_error=False,fill_value=0.0)
        
        return I(chis)
    

    if tag == 'ml':
        
        cut  = np.where( chis < chi_1 , 0,1)*np.where( chis >= chi_2 , 0,1)
        zs = csm.z_from_chi(chis)
        a = 1.0/(1.0 + zs)                
        Omega = conf.Omega_b+conf.Omega_c
        H0 = csm.H_z(0.001)
        k = (ell+1.0/2.0)/chis

        Dpsi_Tk = 3.0*cut*Omega*(H0**2)/k**2./a/zb.deltachi

        return Dpsi_Tk*cut/chis
    
    if tag == 'lensing':

        zs = csm.z_from_chi(chis)
        cut  = np.where( chis < csm.chi_from_z(0.02) , 0,1)
        a = 1.0/(1.0 + zs)
        H = csm.H_z(zs)

        Omega = conf.Omega_b+conf.Omega_c
        H0 = csm.H_z(0.001)
        k = (ell+1.0/2.0)/chis

        Dpsi_Tk = 3.0*Omega*(H0**2)/k**2./a

        chi_s = csm.chi_from_z(1090)

        return Dpsi_Tk*cut*((chi_s-chis)/(chi_s*chis))

        
##########  Limber integration for the non-linear contribution to the angular power spectrum




def limber_integration(tag1, tag2, fq1, fq2, b1, b2, ell, pk_limb):
    
    #Let's first determine an optimal grid for integration. This is necessary
    #because having a common grid for all observables would require a very
    #high number of points to be precise everywhere. The scheme below opttimizes
    #the grid for each specific tag1/tag2 combination.
    
    if tag1 in ['isw_lin','g', 'tSZ', 'CIB','lensing'] and tag2 in ['isw_lin','g','tSZ', 'CIB','lensing']: 
        #This observables have wide window functions regardless of the redshift binning.
        chis = np.linspace(csm.chi_from_z(1e-2), csm.chi_from_z(conf.z_max+0.99), 6000)
        #chis = np.linspace(csm.chi_from_z(1e-2), csm.chi_from_z(8), 10000)
        
    else:
        
        
        if 'vr' in [tag1,tag2]:
            num = int((50000))
        else:
            num = 3000
        
        if tag1 in ['isw_lin','g', 'tSZ', 'CIB','lensing']:
            min_chi = np.max([csm.chi_from_z(1e-2)*1.001,zb.zbins_chi[b2]*(ell+0.5)/(ell+1.5)])
            max_chi = np.min([csm.chi_from_z(conf.z_max+0.9)*0.999,zb.zbins_chi[b2+1]*(ell+0.5)/(ell-0.5)])
            chis = np.linspace(min_chi, max_chi, num)     
        elif tag2 in ['isw_lin','g', 'tSZ', 'CIB','lensing']:
            min_chi = np.max([csm.chi_from_z(1e-2)*1.001,zb.zbins_chi[b1]*(ell+0.5)/(ell+1.5)])
            max_chi = np.min([csm.chi_from_z(conf.z_max+0.9)*0.999,zb.zbins_chi[b1+1]*(ell+0.5)/(ell-0.5)])
            chis = np.linspace(min_chi, max_chi, num)        
        else:
            if np.abs(b1-b2)>4:
                #Top hats and v window functions are at most 1 bin away in this limit, so we can drop the rest.
                return 0.0
            else:   
                min_chi = np.max([csm.chi_from_z(1e-2)*1.001,zb.zbins_chi[np.min([b1,b2])]*(ell+0.5)/(ell+1.5)])
                max_chi = np.min([csm.chi_from_z(conf.z_max+0.9)*0.999,zb.zbins_chi[np.max([b1+1,b2+1])]*(ell+0.5)/(ell-0.5)])
                chis = np.linspace(min_chi, max_chi, num)
                
    
    
    lw1 = limber_weight(tag1, fq1, b1, ell, chis)
    lw2 = limber_weight(tag2, fq2, b2, ell, chis)
    
    I = lw1*lw2*pk_limb(chis)/(chis**2)
    return np.trapz(I,chis)
    

#########  Bayond limber integration method. Combines limber with fftlog stuff.
    
def beyond_limber(tag1, tag2, fq1, fq2, b1, b2, k, ell, pk_limb):
    
    
    if pk_limb == None:
        term1 = 0
    else:               
        term1  = limber_integration(tag1, tag2, fq1, fq2, b1, b2, ell, pk_limb)
    
    pk = c.load(basic_conf, 'p_linear_'+c.retag(tag1)+c.retag(tag2)+'_f1='+str(fq1)+'_f2 ='+str(fq2), dir_base = 'pks')
    fft1 = c.load(basic_conf, 'f_'+tag1+'_fq='+str(fq1)+'_b='+str(b1), dir_base = 'ffts')
    fft2 = c.load(basic_conf, 'f_'+tag2+'_fq='+str(fq2)+'_b='+str(b2), dir_base = 'ffts')  
    
    #int_term2 = fft1*fft2*np.sqrt(pk(k,zc_1)*pk(k,zc_2))/k
    int_term2 = fft1*fft2*pk(k,conf.zs_hm[0])/k
       
    term2 = np.trapz(int_term2,k)   
    
    return term1+term2
    


####################################################
####################  Building the CLS sample
####################################################

def Cl_row(tag1, tag2, fq1, fq2, ell, pk_limb,i):
        
    N2 = zb.nbin_tag(tag2)
    
    Crow = np.zeros(N2)

    k = fftlog_integral(tag1,fq1,0,ell)[0]
    
    for j in np.arange(N2):
        
        Crow[j] = beyond_limber(tag1, tag2, fq1, fq2, i, j, k, ell, pk_limb)
    
    return Crow   

def Cl(tag1, tag2, fq1, fq2, ell):
    
    if ell <30:
        pk_limb = None
    else:   
        chis_interp = np.linspace(csm.chi_from_z(1e-2), csm.chi_from_z(conf.z_max+1.1), 1000)            
        pf_interp = limber(c.load(basic_conf, 'p_'+c.retag(tag2)+c.retag(tag2)+'_f1='+str(fq2)+'_f2 ='+str(fq2), dir_base = 'pks'),chis_interp,ell)  
        pl_interp = limber(c.load(basic_conf, 'p_linear_'+c.retag(tag2)+c.retag(tag2)+'_f1='+str(fq2)+'_f2 ='+str(fq2), dir_base = 'pks'),chis_interp,ell)          
        pk_limb = interp1d(chis_interp,pf_interp-pl_interp, kind = 'linear',bounds_error=False,fill_value=0.0)
        
    N1,N2 = zb.nbin_tag(tag1),zb.nbin_tag(tag2)
    
    C = np.zeros((N1,N2))
      
    if N1 <= get_n_cores():
        for i in np.arange(N1):
            C[i,:] = Cl_row(tag1, tag2, fq1, fq2, ell, pk_limb,i) 
    else:   
        pool = mp.Pool(get_n_cores())
        pool.starmap(Cl_row, [(tag1, tag2, fq1, fq2, ell, pk_limb, i)  for i in np.arange(N1)])
        pool.close()       
                
    return C
    

def wigner_symbol(ell, ell_1,ell_2):
     
    if not ((np.abs(ell_1-ell_2) <= ell) and (ell <= ell_1+ell_2)):  
        return 0 
 
    J = ell +ell_1 +ell_2
    if J % 2 != 0:
        return 0
    else:
        g = int(J/2)
        w = (-1)**(g)*np.exp((lgamma(2*g-2*ell+1)+lgamma(2*g-2*ell_1+1)+lgamma(2*g-2*ell_2+1)-lgamma(2*g+1+1))/2 +lgamma(g+1)-lgamma(g-ell+1)-lgamma(g-ell_1+1)-lgamma(g-ell_2+1))
        
        return w


def Cl_ksz_row(Cf, ell, ell_1,lmax):
    
    terms = 0.0
    for ell_2 in np.arange(np.abs(ell_1-ell),ell_1+ell+1):
        if ell_2 > lmax+200:   #triangle rule
            continue
        wigner = wigner_symbol(ell, ell_1,ell_2)
        
        factor = (2*ell_1+1)*(2*ell_2+1)/4/np.pi
                                          
        terms += factor*wigner*wigner*Cf[ell_1,ell_2]
     
    return terms


def Cl_ksz(Cf, ell, lmax):
    
    Ls = np.unique(np.append(np.geomspace(1,lmax+200,500).astype(int), lmax+200))   
    a = []
    for ell_1 in Ls:
        a.append(Cl_ksz_row(Cf,ell,ell_1,lmax))    
    I = interp1d(Ls ,np.asarray(a), kind = 'linear',bounds_error=False,fill_value=0.0)(np.arange(lmax+201))
    
    return np.sum(I)


def Cl_ml_row(Cf, ell, ell_1, lmax):
    terms = 0.0
    for ell_2 in np.arange(np.abs(ell_1 - ell), ell_1 + ell + 1):
        if ell_2 > lmax+200:
            continue
        wigner = wigner_symbol(ell, ell_1, ell_2)
        factor = (2 * ell_1 + 1) * (2 * ell_2 + 1) / 4 / np.pi
        factor_ml = (ell_1 * (ell_1 * 1.0) + ell_2 * (ell_2 + 1.0) - ell * (ell + 1))/2.0
        terms += factor * wigner * wigner * Cf[ell_1, ell_2] * factor_ml * factor_ml

    return terms


def Cl_ml(Cf, ell, lmax):
    
    L1 = np.unique(np.append(np.geomspace(1, 1000, 100).astype(int), np.arange(1000,lmax+200,20))) 
    Ls = np.unique(np.append(L1, lmax+200))
    #Ls = np.unique(np.append(np.geomspace(1, lmax+200, 3000).astype(int), lmax+200))
    a = []
    for ell_1 in Ls:
        a.append(Cl_ml_row(Cf, ell, ell_1, lmax))

    I = interp1d(Ls, (np.asarray(a)), kind='linear', bounds_error=False, fill_value=0.0)(np.arange(lmax + 201))
    return np.sum(I)


###################################### Some auxiliary functions for multifrequency cleaning
    

def get_CTT(freq_i, freq_j,  lmax, primary=True, inoise=True, cib=True, tsz=True):
    
    ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax)) 
    TT_spec = np.zeros(len(ell_sparse))
    TT_noise = np.zeros(len(ell_sparse))
    if primary:
        TT_spec += loginterp.log_interpolate_matrix(c.load(basic_conf,'Cl_pCMB_pCMB_lmax='+str(lmax), dir_base = 'Cls'), c.load(basic_conf,'L_pCMB_lmax='+str(lmax), dir_base = 'Cls'))[ell_sparse,0,0]
    if inoise:
        noisemodel = ng.model_selection(conf.cleaning_mode)
        noise_spec = noisemodel(conf.cleaning_frequencies[conf.cleaning_mode], ell_sparse)
        if freq_i == freq_j:  # Assume uncorrelated freqxfreq noise
            f_i = np.where(freq_i==conf.cleaning_frequencies[conf.cleaning_mode])[0][0]
            TT_noise = noise_spec[f_i,:] / conf.T_CMB**2
        TT_spec += TT_noise
    if ((cib) or (tsz)):
        if cib:
            TT_spec += c.load(basic_conf,'Cl_CIB'+'('+str(min(freq_i,freq_j))+')'+'_CIB'+'('+str(max(freq_i,freq_j))+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0]
        if tsz:
            TT_spec += c.load(basic_conf,'Cl_tSZ'+'('+str(min(freq_i,freq_j))+')'+'_tSZ'+'('+str(max(freq_i,freq_j))+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0]
        if (tsz and cib):
            try:
                TT_spec += c.load(basic_conf,'Cl_CIB'+'('+str(freq_i)+')'+'_tSZ'+'('+str(freq_j)+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0] 
            except:
                TT_spec += c.load(basic_conf,'Cl_tSZ'+'('+str(freq_j)+')'+'_CIB'+'('+str(freq_i)+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0] 
            try:
                TT_spec += c.load(basic_conf,'Cl_CIB'+'('+str(freq_j)+')'+'_tSZ'+'('+str(freq_i)+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0] 
            except:
                TT_spec += c.load(basic_conf,'Cl_tSZ'+'('+str(freq_i)+')'+'_CIB'+'('+str(freq_j)+')'+'_lmax='+str(lmax), dir_base = 'Cls')[:,0,0] 
            
        
    return TT_spec*np.where(ell_sparse>1,1,0)


def get_CTX( X,freq_i, lmax, isw = False, cib=False, tsz=False):
    
    ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax))
    clTX = np.ones((len(ell_sparse),conf.N_bins))*1e-35
    
    if isw:   
        try:
            clTX += c.load(basic_conf,'Cl_'+X+'_isw_lin_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(X,'isw_lin',conf))[:,:,0]
        except:
            clTX += c.load(basic_conf,'Cl_isw_lin_'+X+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('isw_lin',X,conf))[:,0,:]
        
    if ((cib) or (tsz)):  # If specified as true add extragalactics
        if cib:
            try:
                clTX += c.load(basic_conf,'Cl_'+X+'_CIB'+'('+str(freq_i)+')'+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(X,'CIB',conf))[:,:,0]
            except:
                clTX += c.load(basic_conf,'Cl_CIB'+'('+str(freq_i)+')'+'_'+X+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('CIB','X',conf))[:,0,:]
        if tsz:
            try:
                clTX += c.load(basic_conf,'Cl_'+X+'_tSZ'+'('+str(freq_i)+')'+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(X,'tSZ',conf))[:,:,0]
            except:
                clTX += c.load(basic_conf,'Cl_tSZ'+'('+str(freq_i)+')'+'_'+X+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('tSZ',X,conf))[:,0,:]
    return clTX

###################################### halo model spectra

def power_spectra(tag1,tag2,Fq1,Fq2,ell_sparse):
    
    print("Getting all necessary power spectra")
    
    Pks = power.pks(conf_module = conf)
    
    for fq1 in Fq1:
        for fq2 in Fq2:
                       
            print('f1='+str(fq1)+' f2='+str(fq2))
            Pks.get_pks(tag1,tag2,fq1,fq2)  #Check if underlying power spectra is already computed
            if tag1 != tag2:
                Pks.get_pks(tag1,tag1,fq1,fq1)
                Pks.get_pks(tag2,tag2,fq2,fq2)
                
            if tag1 == 'g' and tag2 == 'g':
                
                if not c.exists(basic_conf,'Nlshot_g_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf)):
                
                    print('Calculating shot noise for gg spectra')
                    
                    if conf.LSSexperiment == 'LSST':
                        
                        if Pks.hmod is None:
                            Pks.start_halo() 
                        
                        ngalMpc3z = Pks.hmod.nbar_galaxy(zb.zbins_zcentral,Pks.hmod.logmmin,Pks.hmod.logmmax,mthreshHOD=Pks.mthreshHODstellar) 
                        dz = np.diff(zb.zbins_z) #get the density over the red shift bin (assuming it is constant within the bin).
                        ngalSteradBinned = Pks.hmod.convert_n_mpc3_arcmin2(ngalMpc3z,zb.zbins_zcentral) * power.halomodel.allsky_arcmin2 / power.halomodel.allsky_sterad * dz
                        
                        Ns_ell = np.zeros((len(ell_sparse),conf.N_bins,conf.N_bins))
                        for i_1 in np.arange(conf.N_bins):
                            for i_2 in np.arange(conf.N_bins):
                                if i_1 ==i_2:
                                    Ns_ell[:,i_1,i_2] = 1.0/ngalSteradBinned[i_1]
                                    #Ns_ell[:,i_1,i_2] = 9.2e-8  # This is for unwise blue bin only
                                    
                        c.dump(basic_conf, Ns_ell,'Nlshot_g_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf))
                    
                    elif conf.LSSexperiment == 'unwise_blue':     
                        
                        Ns_ell = np.zeros((len(ell_sparse),1,1))
                        for i_1 in np.arange(1):
                            for i_2 in np.arange(1):
                                if i_1 ==i_2:
                                    Ns_ell[:,i_1,i_2] = 9.2*10**(-8)
    
                                    
                        c.dump(basic_conf, Ns_ell,'Nlshot_g_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf))
    
               
                    elif conf.LSSexperiment == 'custom':  
                        
                        Ns_ell = np.zeros((len(ell_sparse),1,1))
                        for i_1 in np.arange(conf.N_bins):
                            for i_2 in np.arange(conf.N_bins):
                                if i_1 ==i_2:
                                    Ns_ell[:,i_1,i_2] = 9.2*10**(-8) 
                                    
                                    
                        c.dump(basic_conf, Ns_ell,'Nlshot_g_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf))
     
                    else:   
                        raise Exception ("LSS experiment not supported.")
    
    del(Pks)
    gc.collect()
    pass

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Script to offload various calculations.')
    parser.add_argument('-t1', '--tracer1', help='Tracer to compute a transfer function for, (vr|g|taud).', action='store')
    parser.add_argument('-t2', '--tracer2', help='Tracer to compute a transfer function for, (vr|g|taud).', action='store')
    parser.add_argument('-lmax', '--lmax', help='Highest multipole ell. Default is conf.estim_smallscale_lmax', action='store')
    parser.add_argument('-postcomp', '--postcomp', help='Option to indicate we are computing new spectra from precomputed ones',action='store_true')
    parser.add_argument('-kSZ', '--kSZ', help='Compute kSZ power spectrum',action='store_true')
    parser.add_argument('-ML', '--ML', help='Compute ML power spectrum', action='store_true')
    parser.add_argument('-Nfine', '--Nfine', help='Number of fine bins. Used when calling ksz spectrum',action='store')
    parser.add_argument('-gettau', '--gettau', help='Compute the contribution to the optical depth coming from each redshift bin',action='store_true')
    parser.add_argument('-cleanTX', '--cleanTX', help='Compute the cleaned TT spectra and the cleaned TX spectra',action='store')
    parser.add_argument('-Tfreq', '--Tfreq', help='Compute the TT,Tg,Tvr and Ttaud for a given frequency',action='store')
    parser.add_argument('-coarse', '--coarse', help='Coarse grain from Nfine to Nbin',action='store_true')
    
    
    if not len(sys.argv) > 1 :
        parser.print_help()
        
    else:    
        args = parser.parse_args()
        
        
        
        #################################################################################################
        ###########################################  SPACE FOR USEFUL CALLS 
        #################################################################################################
        
        if args.gettau:   
            c.dump(basic_conf, get_tau(),'tau_binned', dir_base = '')
            
        #################################################################################################
        ###########################################  QUANTITIES DERIVED FROM PRECOMPUTED SPECTRA
        #################################################################################################
        
        elif args.postcomp:
            
            if args.lmax is None:
                parser.print_help()
                raise Exception("lmax has to be specified ")
            lmax = int(args.lmax)
            
            
            #####################################################################
            ####################### COARSE CL
            ####################################################################
            
            if args.coarse:
                if args.lmax is None:
                    parser.print_help()
                    raise Exception("lmax has to be specified ")
                lmax = int(args.lmax)
                
                ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax)) 
                
                if args.Nfine is None:
                    parser.print_help()
                    raise Exception("Nfine for ksz contribution has to be specified. Use -Nfine flag")
                Nfine = int(args.Nfine)
                            
                if args.tracer1 is None or args.tracer2 is None:
                    parser.print_help()
                    raise Exception("Both tracers t1 t2 have to be specified ")
        
                tag1 = args.tracer1
                tag2 = args.tracer2
                
                killflag1 = killflag2 = True  # Avoid double processing for frequency-less indices. Kind of messy can probably be done nicer
                coarsefreqs1 = coarsefreqs2 = ['']
                if zb.nbin_tag(tag1) == 1:
                    if tag1 in ['CIB', 'tSZ']:
                        coarsefreqs1 = ['(%d)'% f for f in freqs()]
                    killflag1 = False
                if zb.nbin_tag(tag2) == 1:
                    if tag2 in ['CIB', 'tSZ']:
                        coarsefreqs2 = ['(%d)'% f for f in freqs()]
                    killflag2 = False
                
                killflag = killflag1 * killflag2  # Avoid double processing for frequency-less indices. Kind of messy can probably be done nicer

                for f1, cf1 in enumerate(coarsefreqs1):
                    for f2, cf2 in enumerate(coarsefreqs2):
                        if ((args.tracer1 == args.tracer2) and (f1 > f2)):  # If CIBCIB etc we don't have f1 <= f2
                            continue
                        tag1 = args.tracer1+cf1
                        tag2 = args.tracer2+cf2
                        C = c.load(basic_conf,'Cl_'+tag1+'_'+tag2+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf,N_bins = Nfine))  
                        print('Cl_'+tag1+'_'+tag2+'_lmax='+str(lmax))

                        nbin1 = nbin2 = conf.N_bins
                        if not killflag1:
                            nbin1 = 1
                        if not killflag2:
                            nbin2 = 1

                        C_coarse = np.zeros((len(ell_sparse),nbin1,nbin2))

                        for i in progressbar(np.arange(len(ell_sparse)), "Computing: ", 40):
                            ell = ell_sparse[i]

                            if nbin1 == nbin2 == conf.N_bins:
                                C_coarse[i,:,:] = zb.coarse_matrix(Nfine,conf.N_bins, C[i,:,:])
                            elif nbin2 > nbin1:  # Order matters for dot product in following function, i.e. CIBg vs gCIB should always have shape (Nbin, 1)
                                C_coarse[i,:,:] = zb.coarse_vector(Nfine,conf.N_bins, C[i,:,:].T).T
                            elif nbin1 < nbin2:
                                C_coarse[i,:,:] = zb.coarse_vector(Nfine,conf.N_bins, C[i,:,:])
                            else:  # For 1x1 temperature spectra, we can use this coarse args to make copies to the coarse Cls output
                                C_coarse = C

                        c.dump(basic_conf, C_coarse,'Cl_'+tag1+'_'+tag2+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf))
                        if killflag:
                            break
                    if killflag:
                        break
                    # Make copy of L sample
                    c.dump(basic_conf, c.load(basic_conf,'L_sample_lmax='+str(lmax), dir_base = 'Cls'), dir_base = 'Cls')
                
                
                
                
            #####################################################################
            ####################### KSZ POWER SPECTRUM
            #####################################################################
            
            
            elif args.kSZ:
                
                if args.Nfine is None:
                    parser.print_help()
                    raise Exception("Nfine for ksz contribution has to be specified. Use -Nfine flag")
                Nfine = int(args.Nfine)
          
                ell_sparse =np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax))   
                Cls = np.zeros((len(ell_sparse),1,1))
                
                Cvv = c.load(basic_conf,'Cl_vr_vr_lmax='+str(lmax), dir_base ='Cls/'+c.direc('vr','vr',conf,N_bins = Nfine))
                Ctdtd = c.load(basic_conf,'Cl_taud_taud_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('taud','taud',conf,N_bins = Nfine))
                try:
                    Cvtd = c.load(basic_conf, 'Cl_vr_taud_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('vr','taud',conf,N_bins = Nfine))
                except:
                    Cvtd = np.transpose(c.load(basic_conf, 'Cl_taud_vr_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('taud','vr',conf,N_bins = Nfine)), axes =[0,2,1])
                    
                Cl1l2 = np.zeros((len(ell_sparse),len(ell_sparse)))  
                for lid_1 , ell_1 in enumerate(ell_sparse):
                    for lid_2 , ell_2 in enumerate(ell_sparse):   
                            Cl1l2[lid_1,lid_2] = np.trace(np.dot(Cvv[lid_1,:,:],Ctdtd[lid_2,:,:])+np.dot(Cvtd[lid_1,:,:],Cvtd[lid_2,:,:])) 
                    
                Cf = interp2d(ell_sparse,ell_sparse,Cl1l2 ,kind = 'linear',bounds_error=False)(np.arange(lmax+201),np.arange(lmax+201))
                
                chis_bounds = zb.Chi_bin_boundaries(conf.z_min, conf.z_max, Nfine)
                deltachi = chis_bounds[1]-chis_bounds[0]
    
                starttime = time.time()
                
                for i in progressbar(np.arange(len(ell_sparse)), "Computing: ", 40):
                #for lid, ell in enumerate(ell_sparse):   
                    ell = ell_sparse[i]
                    Cls[i,0,0] = Cl_ksz(Cf, ell, lmax)*deltachi**2
               
                print('That took {} seconds'.format(time.time() - starttime))
                
                
                
                c.dump(basic_conf, Cls,'Cl_kSZ_Nfine_'+str(Nfine)+'_lmax='+str(lmax), dir_base = 'Cls')
                
                
            #####################################################################
            ####################### MOVING LENS POWER SPECTRUM
            #####################################################################                

            elif args.ML:
                
                if args.Nfine is None:
                    parser.print_help()
                    raise Exception("Nfine for ksz contribution has to be specified. Use -Nfine flag")
                Nfine = int(args.Nfine)
                
                ell_sparse = np.unique(np.append(np.geomspace(1, lmax, 120).astype(int), lmax))
                Cls = np.zeros((len(ell_sparse),1,1))
                Cvtvt = c.load(basic_conf, ('Cl_vt_vt_lmax=' + str(lmax)), dir_base='Cls/'+c.direc('vt','vt',conf,N_bins = Nfine))
                Cmlml = c.load(basic_conf, ('Cl_ml_ml_lmax=' + str(lmax)), dir_base='Cls/'+c.direc('ml','ml',conf,N_bins = Nfine))
                try:
                    Cvtml = c.load(basic_conf, ('Cl_vt_ml_lmax=' + str(lmax)), dir_base='Cls/'+c.direc('vt','ml',conf,N_bins = Nfine))
                except:
                    Cvtml = np.transpose(c.load(basic_conf, 'Cl_ml_vt_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('ml','vt',conf,N_bins = Nfine)), axes =[0,2,1])
                Cl1l2 = np.zeros((len(ell_sparse), len(ell_sparse)))
                for lid_1, ell_1 in enumerate(ell_sparse):
                    for lid_2, ell_2 in enumerate(ell_sparse):
                        Cl1l2[(lid_1, lid_2)] = np.trace(np.dot(Cvtvt[lid_1, :, :], Cmlml[lid_2, :, :]) + np.dot(Cvtml[lid_1, :, :], Cvtml[lid_2, :, :]))

                Cf = interp2d(ell_sparse, ell_sparse, Cl1l2, kind='linear', bounds_error=False)(np.arange(lmax + 201), np.arange(lmax + 201))
                chis_bounds = zb.Chi_bin_boundaries(conf.z_min, conf.z_max, Nfine)
                deltachi = chis_bounds[1] - chis_bounds[0]
                starttime = time.time()
                for i in progressbar(np.arange(len(ell_sparse)), 'Computing: ', 40):
                    ell = ell_sparse[i]
                    Cls[i,0,0] = Cl_ml(Cf, ell, lmax)*deltachi**2

                print('That took {} seconds'.format(time.time() - starttime))
                c.dump(basic_conf, Cls, ('Cl_ML_Nfine_' + str(Nfine) + '_lmax=' + str(lmax)), dir_base='Cls')
                c.dump(basic_conf, ell_sparse, ('L_sample_lmax=' + str(lmax)), dir_base='Cls')              
   
            #####################################################################
            ####################### CLEANING OF TT  AND TG POWER SPECTRUM
            #####################################################################
             
            elif args.cleanTX is not None:
                
                X = args.cleanTX
                
                ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax)) 
                
                CTT_clean = np.zeros((len(ell_sparse),1,1))
                CTX_clean = np.zeros((len(ell_sparse),1,conf.N_bins))
                                
                F = freqs()
                NMAPS = len(F)
                C = np.zeros((NMAPS, NMAPS, len(ell_sparse)))
                
                for i in np.arange(NMAPS):
                    for j in np.arange(NMAPS):
                        C[i,j] = get_CTT(F[i], F[j], lmax, primary=True, inoise=True, cib=True, tsz=True)

                        
                print('Computing ILC weights')
                weights_l = np.zeros((NMAPS, len(ell_sparse)))
                es = np.ones(NMAPS)
                for lid, l  in enumerate(ell_sparse):
                    try:
                        Cl_inv = np.linalg.inv(C[:,:,lid])
                    except np.linalg.LinAlgError as e:
                        print('%s at l = %d' % (e, l))
                        weights_l[:,lid] = np.ones(NMAPS) / np.linalg.norm(np.ones(NMAPS))
                    else:
                        weights_l[:,lid] = (np.dot(Cl_inv, es) / np.dot(np.dot(es,Cl_inv),es))
                
                ell_facs = np.zeros(len(ell_sparse))
                cross_facs = np.zeros((len(ell_sparse),conf.N_bins))
                extragal_cross = np.zeros((NMAPS, len(ell_sparse), conf.N_bins))
                extragal_specs = np.zeros((NMAPS, NMAPS, len(ell_sparse)))
                noise_specs = np.zeros((NMAPS, NMAPS, len(ell_sparse)))  
                
                print('Adding dirty components')
                for i in range(NMAPS):
                    extragal_cross[i] += get_CTX( X,F[i], lmax, isw = False, cib=True, tsz=True)
                    noise_specs[i,i] += get_CTT(F[i], F[i], lmax,primary=False, inoise=True,cib=False,tsz=False)
                    for j in range(NMAPS):          
                        extragal_specs[i,j] += get_CTT(F[i], F[j], lmax, primary=False,inoise=False,cib=True,tsz=True)    
                
                for lid, l in enumerate(ell_sparse):
                    
                    if l< 2:
                        continue
                    else:
                                            
                        w = weights_l[:,lid]
                        c_extragal = extragal_specs[:,:,lid]
                        N_l = noise_specs[:,:,lid]
                        ell_facs[lid] = np.dot(np.dot(w, c_extragal+N_l),w)  # This is the cleaning residual for the auto spectrum
                        for n in range(conf.N_bins):
                            cross_facs[lid,n] = np.dot(w, extragal_cross[:,lid,n])  # The cleaned cross spectra is just the weighted CgT as shot/inoise are uncorrelated
            
                CTT_clean[:,0,0] = get_CTT(F[0], F[0], lmax,primary=True,inoise=False,cib=False,tsz=False) + ell_facs  # TT clean = TT real + ILC residuals
                if X in ['vr','taud']:
                    CTX_clean[:,0,:] = get_CTX( X,F[0], lmax, isw = True, cib=False, tsz=False)
                else:
                    CTX_clean[:,0,:] = get_CTX( X,F[0], lmax, isw = True, cib=False, tsz=False)+cross_facs 
                
                c.dump(basic_conf, CTT_clean,'Cl_Tc_Tc_lmax='+str(lmax), dir_base = 'Cls')
                c.dump(basic_conf, CTX_clean,'Cl_Tc_'+X+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('Tc',X,conf)) 

            
                
            elif args.Tfreq is not None:
                
                ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax)) 
                
                CTT_freq = np.zeros((len(ell_sparse),1,1))
                CTg_freq = np.zeros((len(ell_sparse),1,conf.N_bins))
                CTv_freq = np.zeros((len(ell_sparse),1,conf.N_bins))
                CTtaud_freq = np.zeros((len(ell_sparse),1,conf.N_bins))
                
                freq = int(args.Tfreq)
                CTT_freq[:,0,0] = get_CTT(freq, freq,  lmax, primary=True, inoise=False, cib=True, tsz=True) # we dont add instrumental noise here
                CTg_freq[:,0,:] = get_CTX( 'g',freq, lmax, isw = True, cib=True, tsz=True)
                CTv_freq[:,0,:] = get_CTX( 'vr',freq, lmax, isw = True, cib=False, tsz=False)
                CTtaud_freq[:,0,:] = get_CTX( 'taud',freq, lmax, isw = True, cib=False, tsz=False)
                
                c.dump(basic_conf, CTT_freq,'Cl_'+'T('+str(freq)+')'+'_'+'T('+str(freq)+')'+'_lmax='+str(lmax), dir_base = 'Cls')
                c.dump(basic_conf, CTg_freq,'Cl_'+'T('+str(freq)+')'+'_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('T','g',conf)) 
                c.dump(basic_conf, CTv_freq,'Cl_'+'T('+str(freq)+')'+'_vr_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('T','vr',conf))
                c.dump(basic_conf, CTtaud_freq,'Cl_'+'T('+str(freq)+')'+'_taud_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('T','taud',conf))
            else:
                
                raise Exception("No post computation operation was specified.")
  
        #################################################################################################
        ###########################################  COMPUTATION OF POWER SPECTRA FOR A PAIR OF TRACERS
        #################################################################################################
          
        else:
            
            if args.lmax is None:
                parser.print_help()
                raise Exception("lmax has to be specified ")
            lmax = int(args.lmax)
        
            tag1 = args.tracer1
            tag2 = args.tracer2
            

            if args.tracer1 is None or args.tracer2 is None:
                parser.print_help()
                raise Exception("Both tracers t1 t2 have to be specified ")
                
            # If we want the primary CMB power spectrum we use camb.
                
            if tag1 in ['pCMB'] and tag2 in ['pCMB']:
                
                print("Computing primary CMB using camb (both lensed and unlensed).")
                
                ell_sparse =np.unique(np.append(np.geomspace(2,lmax,800).astype(int), lmax)) 
                Cls = np.zeros((len(ell_sparse),1,1))
                Cls_unlensed = np.zeros((len(ell_sparse),1,1))
                csm.cambpars.set_for_lmax(lmax, lens_potential_accuracy=1)
                results = camb.get_results(csm.cambpars)
                powers =results.get_cmb_power_spectra(csm.cambpars, CMB_unit='muK')
                totCL=powers['total']
                CL_unlensed=powers['unlensed_scalar']
                ls = np.arange(totCL.shape[0])
                Cls[:,0,0]          = totCL[ell_sparse,0]*2*np.pi/ls[ell_sparse]/(ls[ell_sparse]+1)/conf.T_CMB**2
                Cls_unlensed[:,0,0] = CL_unlensed[ell_sparse,0]*2*np.pi/ls[ell_sparse]/(ls[ell_sparse]+1)/conf.T_CMB**2
                
                c.dump(basic_conf, Cls,'Cl_'+tag1+'_'+tag2+'_lmax='+str(lmax), dir_base = 'Cls')
                c.dump(basic_conf, Cls_unlensed,'Cl_'+tag1+'_'+tag2+'_unlensed_lmax='+str(lmax), dir_base = 'Cls')
                c.dump(basic_conf, ell_sparse,'L_pCMB_lmax='+str(lmax), dir_base = 'Cls')
                 
                
                
            else:
                
                ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax))
                                        
                #Set up frequencies
                
                if tag1 == 'CIB':
                    Fq1 = freqs()
                else:
                    Fq1 = [None]
                if tag2 == 'CIB':
                    Fq2 = freqs()
                else:
                    Fq2 = [None]
                    
                #Run for all relevant frequencies
                
                #First lets do all the halo model calculations 
                
                power_spectra(tag1,tag2,Fq1,Fq2,ell_sparse)


                for fq1 in Fq1:
                    for fq2 in Fq2:                            

                        # if ((fq1 and fq2) is not None and (fq1 > fq2)):  # Save computation time. Only foregrounds are frequency dependent, and will produce symmetric freqxfreq matrices
                        #     continue
                        
                        print('Computing for frequencies fq1 = '+str(fq1)+' fq2 = '+str(fq2))
                                                       
                        Cls = []
                        
                        starttime = time.time()
                        
                        
                        if tag1 == tag2 and fq1 ==fq2:        
                            save_fft_weights(tag1,fq1)
                        else:
                            save_fft_weights(tag1,fq1)
                            save_fft_weights(tag2,fq2)
                            
                        
                        for i in progressbar(np.arange(len(ell_sparse)), "Computing: ", 40):
                        #for lid, ell in enumerate(ell_sparse):
                            
                            ell = ell_sparse[i]           

                            if tag1 == tag2 and fq1 == fq2:
                                st = time.time()
                                get_ffts(tag1,fq1,ell)
                            else:
                                st = time.time()
                                get_ffts(tag1,fq1,ell)
                                get_ffts(tag2,fq2,ell)
                                
                                                                            
                            Cls.append(Cl(tag1, tag2, fq1, fq2, ell))
                            
                        CLS = np.asarray(Cls)
                               
                        print('That took {} seconds'.format(time.time() - starttime))
                        
                        
                        #add shot noise for gg spectra:
                            
                        if tag1 == 'g' and tag2 == 'g':
                            
                            print('Adding shot noise for gg spectra')
                        
                            Ns_ell = c.load(basic_conf,'Nlshot_g_g_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf))
                            c.dump(basic_conf, CLS,'Cl_g_g_noshot_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('g','g',conf))
                            CLS += Ns_ell
                                                        
                        #Build labels
                        label1 = ''
                        label2 = ''
                        if tag1 =='CIB':      
                            label1 += '('+str(fq1)+')'
                        if tag2 =='CIB':  
                            label2 += '('+str(fq2)+')'        
                            
                        if tag1 == 'tSZ' and tag2 != 'tSZ':
                            for f in freqs():
                                c.dump(basic_conf, CLS*f_tSZ(f),'Cl_tSZ'+'('+str(f)+')'+'_'+tag2+label2+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf))
                        elif tag1 != 'tSZ' and tag2 == 'tSZ':
                            for f in freqs():
                                c.dump(basic_conf, CLS*f_tSZ(f),'Cl_'+tag1+label1+'_tSZ'+'('+str(f)+')'+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf))     
                        elif tag1 == 'tSZ' and tag2 == 'tSZ':
                            for f1 in freqs():
                                for f2 in freqs():
                                    c.dump(basic_conf, CLS*f_tSZ(f1)*f_tSZ(f2),'Cl'+'_tSZ'+'('+str(f1)+')'+'_tSZ'+'('+str(f2)+')'+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf))    
                        else:
                            c.dump(basic_conf, CLS,'Cl_'+tag1+label1+'_'+tag2+label2+'_lmax='+str(lmax), dir_base = 'Cls/'+c.direc(tag1,tag2,conf))
                                                 
                            
                        c.dump(basic_conf, ell_sparse,'L_sample_lmax='+str(lmax), dir_base = 'Cls')
                          
                                
            
                    

        

            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
