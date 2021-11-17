#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:34:29 2019

@author: fionamccarthy

This module precomputes the yk profiles for the tSZ power spectrum

It takes a couple of minutes to compute yk at the default resolution

"""


import numpy as np
import common as c
import halomodel as hm

#define some physical constants

sigmaT= 6.6524587158e-29          # Thompson scattering cross section, in m^2
mElect= 4.579881126194068e-61     # mass of the electron, in solar masses
c=299792458.0                     # speed of light, in m/s
Gnewt=4.518294724674995e-48       # Newton's constant, in megaparsecs^3/solarmass/second^2

class tSZ(object):
    
    def __init__(self, conf_module = None, halomodel = None):
        
        self.G_mpc=4.30091e-09

        if conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = c.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a hash string or configuration module to locate precomputed data") 
        
        self.fb=self.conf.Omega_b/self.conf.Omega_m   
        
        self.halomodel=halomodel
        
        
        self.z=halomodel.z
        
        self.mhalos=np.exp(halomodel.lnms)
        
        self.rvir=self.halomodel.rvir
        self.mvir=self.halomodel.mvir

        self.m200_c = halomodel.m200_c
        self.r200_c = halomodel.r200_c
        
        self.m200_d = halomodel.mhalos
        self.r200_d = halomodel.r200_d
        
        
        self.k=self.halomodel.k
        
        
        self.ukm=np.zeros((self.k.shape[0],self.mhalos.shape[0],self.z.shape[0]))
        
        self.rhocritz = halomodel.rhocritz
        
        self.pdelta = self.pdelta()
    
      
    def gasprofile_battaglia_AGN_pressure_x(self,x,z,m200):
        #equation 10 of 1109.3711
            P00 = 18.1
            P0am = 0.154
            P0az = -0.758
            
            beta0 = 4.35
            betaam = 0.0393
            betaaz = 0.415
            
            xc0 = 0.497
            xcam = -0.00865
            xcaz = 0.731
            
            P0 = P00*(m200/1e14)**(P0am)*(1+z)**(P0az)
            
            alpha = 1
            beta = beta0*(m200/1e14)**(betaam)*(1+z)**(betaaz)
            xc = xc0*(m200/1e14)**(xcam)*(1+z)**(xcaz)
            gamma = -0.3
    
            pressure = P0*((x/xc)**gamma)*( 1. +(x/xc)**alpha )**(-(beta))
            return pressure
        
             
        
    def pdelta(self):
        rhocritz = self.halomodel.rhocritz
        return  Gnewt*self.m200_c*200*rhocritz*self.fb/(2*self.r200_c)
    
    
    def pressureprofile_x(self,x,z,m200,mind,zind):
        
        
        XH=.76 
        thermal_to_electron=2.0*(XH+1.0)/(5.0*XH+3.0)                            # thermal to electron pressure
        
        return thermal_to_electron * self.pdelta[mind,zind] * self.gasprofile_battaglia_AGN_pressure_x(x,z,m200)
       
            
    def make_yk_phys(self,nxs,m200,r200):
        
        xs=np.linspace(0,4,nxs)[1:]
        
        integral=np.zeros((self.k.shape[0],self.mhalos.shape[0],self.z.shape[0]))
        for mind in range(0,self.mhalos.shape[0]):
            for zind in range(0,self.z.shape[0]):   # vectorizing doesn't speed this up

                exsquaredpressure=xs**2*self.pressureprofile_x(xs[:],self.z[zind],m200[zind,mind],mind,zind)

                kphys=self.k*(1+self.z[zind])
                
                sinkxoverkx1 = np.sin(kphys[:,np.newaxis]*xs[:]*r200[zind,mind])/(kphys[:,np.newaxis]*xs[:]*r200[zind,mind])
                
                integral[:,mind,zind] = np.trapz((r200[zind,mind]**3*sinkxoverkx1*exsquaredpressure[:]),xs,axis=1)
        return integral

if __name__ == "__main__":
    
    zs = np.geomspace(0.01,5,50) 
    halomodel = hm.HaloModel(npts=1000,kpts=200,include_ykm=False)   
    halomodel._setup_k_z_m_grid_functions(zs,include_ukm=False,include_CIB=False)
    tsz_mod = tSZ(halomodel)
    print("calculating yk")
    yk = tsz_mod.make_yk_phys(1000,np.transpose(tsz_mod.m200_c),np.transpose(tsz_mod.r200_c))
    print('calculated. saving...')
    np.save("yk_precomputed.npy",yk)
    np.savez("yk_precomputed_sampling.npz",zsprecomp=halomodel.z,mhalos_precomp=np.exp(halomodel.lnms),k_precomp=halomodel.k)

    
    
    
    
        

  
        