#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.interpolate import interp1d
import common as c
import cosmology 



class binning(object):

    def __init__(self, conf_module = None) :

        if conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = c.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a hash string or configuration module to locate precomputed data") 
        
        self.csm = cosmology.cosmology(conf_module = self.conf)
        
        # parameters for biining

        self.zbins_chi = self.Chi_bin_boundaries(self.conf.z_min, self.conf.z_max, self.conf.N_bins)
        self.zbins_chicentral = self.Chi_bin_centers(self.conf.z_min, self.conf.z_max, self.conf.N_bins)
        self.zbins_z = self.csm.z_from_chi(self.zbins_chi)
        self.zbins_zcentral = self.csm.z_from_chi(self.zbins_chicentral)
        self.deltachi = self.zbins_chi[1]-self.zbins_chi[0]
        
        
    ################################################################
    ################### PI-BINNING, HAAR-BINNING AND COARSE GRAINING
    ################################################################
    
    def Chi_bin_boundaries(self, z_min, z_max, N) :
        Chi_min = self.csm.chi_from_z(z_min)
        Chi_max = self.csm.chi_from_z(z_max)
        Chi_boundaries = np.linspace(Chi_min, Chi_max, N+1)    
        return Chi_boundaries
    
    #Get comoving distances at center of of bins from Chi_bin_boundaries()
        
    def Chi_bin_centers(self, z_min, z_max, N) :
        Chi_boundaries = self.Chi_bin_boundaries(z_min, z_max, N)
        Chis = ( Chi_boundaries[:-1] + Chi_boundaries[1:] ) / 2.0
        return Chis
    
    #### Haar matrix for our choice of normalization
    
    def haar(self,kmax,k,t):

        p=0.
        if k != 0:
            p=int(np.log2(k))
        q=k-2**p+1
        twop=2**p
        
        haarout = 0.0
        
        if (q-1)/twop <= t < (q-0.5)/twop:
            haarout = np.sqrt(twop)
        if (q-0.5)/twop <= t < q/twop:
            haarout = -np.sqrt(twop)
    
        if k==0:
            haarout = 1
        
        return haarout/np.sqrt(kmax)


    def haar_wavelet(self,kmax,k,chis):
    
        if k != 0:
            hv = np.vectorize(self.haar, excluded=['kmax','k'])
            chis_bounds = self.Chi_bin_boundaries(self.conf.z_min, self.conf.z_max, kmax)
            dchi = chis_bounds[1]-chis_bounds[0]
            return hv(kmax,k,(chis-chis_bounds[0])/(chis_bounds[-1]-chis_bounds[0]))/np.sqrt(dchi)
        else:
            chis_bounds = self.Chi_bin_boundaries(self.conf.z_min, self.conf.z_max, kmax)
            dchi = chis_bounds[1]-chis_bounds[0]
            theta_b  = np.where( chis <= chis_bounds[0],0,1)*np.where( chis > chis_bounds[-1] , 0,1)
            return theta_b/ np.sqrt(dchi*kmax)
    
    def bin2haar_brute(self,kmax):
        
        chis_int = np.linspace(self.csm.chi_from_z(1e-2), self.csm.chi_from_z(self.conf.z_max+1), 6000)
        chis_bounds = self.Chi_bin_boundaries(self.conf.z_min,self.conf.z_max, kmax)
        H = np.zeros((kmax,kmax))
        
        for k in np.arange(kmax):
            for i in np.arange(kmax):
                theta_i  = np.where( chis_int <= chis_bounds[i],0,1)*np.where( chis_int >= chis_bounds[i+1],0,1)
                H[k,i] = np.trapz(self.haar_wavelet(kmax,k,chis_int)*theta_i,chis_int)
                
        return H
    
    def bin2haar(self, kmax):

        haarmatrixout = np.zeros((kmax,kmax))
        
        chis_bounds = self.Chi_bin_boundaries(self.conf.z_min,self.conf.z_max, kmax)
        dchi = chis_bounds[1]-chis_bounds[0]
    
        for i in range(kmax):
            for j in range(kmax):
                haarmatrixout[i,j] = self.haar(kmax,i,j/kmax)
    
        return haarmatrixout*np.sqrt(dchi)
    
    def haar2bin(self, kmax):
    
        chis_bounds = self.Chi_bin_boundaries(self.conf.z_min, self.conf.z_max, kmax)
        dchi = chis_bounds[1]-chis_bounds[0]
        return np.transpose(self.bin2haar(kmax)/dchi)
        
        
    def binmatrix(self,nbinfine,nbincoarse):

        # By convention we have nfinebin = 2^n = 2,4,8,16,32,64 etc and so 
        # nbinfine must be ncoarsebin <= nfinebin and equal to 2,4,8,etc. 

        binmatrix = np.zeros((nbincoarse,nbinfine))

        len = nbinfine/nbincoarse

        for i in range(nbincoarse):
            for j in range(nbinfine):
                if i*len <= j < (i+1)*len:
                    binmatrix[i,j] = 1

        return binmatrix/len
    
    def coarse_matrix(self,N_fine,N_coarse, M):
        
        Window = self.binmatrix(N_fine,N_coarse)
            
        return np.dot(np.dot(Window,M),np.transpose(Window))

    def coarse_vector(self,N_fine,N_coarse, M):

        Window = self.binmatrix(N_fine,N_coarse)

        return np.dot(Window,M)
    
    ################################################################
    ################### GALAXY BINNING 
    ################################################################    
    
    def photoz_prob(self,zs,z_a,z_b):
        
        def Int(zp,zr):   
            return np.exp(-(zp-zr)**2/2.0/(self.conf.sigma_photo_z*(1.0+zr))**2)
            
        zp_1 = np.logspace(np.log10(z_a),np.log10(z_b),3000)
        H = self.csm.H_z(zp_1)[:,np.newaxis]
        zp_2 = np.logspace(np.log10(0.001),np.log10(self.conf.z_max+2),6000)
        
        I1 = np.trapz(Int(zp_1[:,None],zs[None,:])/H ,zp_1, axis = 0 )
        I2 = np.trapz(Int(zp_2[:,None],zs[None,:])   ,zp_2, axis = 0 )
        
        
        return I1/I2
    
    
    def get_galaxy_window(self,i):   #generate a sample of the galaxy window functions for interpolation
    
        if c.exists(self.basic_conf_dir, 'galaxy_windows'+'-'+str(i) , dir_base =  'Cls/'+c.direc('g','g',self.conf)):
            return c.load(self.basic_conf_dir, 'galaxy_windows'+'-'+str(i) , dir_base =  'Cls/'+c.direc('g','g',self.conf))
        else:
            
            chis_int = np.linspace(0,self.csm.chi_from_z(self.conf.z_max+1.1),1000)
            zs_int   = self.csm.z_from_chi(chis_int)
            
            gal_samp = np.zeros(len(zs_int))
            
            if self.conf.LSSexperiment == 'LSST':
                
                z_a = self.zbins_z[i]
                z_b = self.zbins_z[i+1]    
                gal_samp = self.photoz_prob(zs_int,z_a,z_b)*self.csm.H_z(zs_int)/self.deltachi
                    
            elif self.conf.LSSexperiment == 'unwise_blue':
                  
                with open('data/unWISE/blue.txt', 'r') as FILE:
                    x = FILE.readlines()
                z = np.array([float(l.split(' ')[0]) for l in x])
       	        dndz = np.array([float(l.split(' ')[1]) for l in x])
       	        dndz_mod = dndz / 1.0 # conf.N_bins # code uses N_bin to divide all spectra including galaxy. This will factor out and give correct spectrum.
       	        gal_samp = interp1d(z,dndz_mod, kind= 'linear',bounds_error=False,fill_value=0)(zs_int)*self.csm.H_z(zs_int) 
                       
            elif self.conf.LSSexperiment == 'custom':  
                #DEFINE HERE YOUR CUSTOM GALAXY REDSHIFT WINDOW (at the moment, a signle window). 
                #BELOW WE LEAVE AN EXAMPLE WITH A SIMPLE GAUSSIAN WINDOW
                            
                sigma_example = 0.3
                z_center = 1
            
                gal_samp= 1.0/sigma_example/np.sqrt(2*np.pi)*np.exp(-(  (z_center-zs_int)/(sigma_example) )**2/2.0)

            else: 
                raise Exception("LSS experiment choice not defined.")
            
            gal_samp_interp = interp1d(zs_int,gal_samp, kind= 'linear',bounds_error=False,fill_value=0)
            
            c.dump(self.basic_conf_dir,gal_samp_interp, 'galaxy_windows'+'-'+str(i) , dir_base = 'Cls/'+c.direc('g','g',self.conf))
            
            return gal_samp_interp
                
        

    
    #############################################################
    ################### SOME UTILITY FUNCTIONS
    #############################################################  

    def nbin_tag(self,tag):
        
        if tag == 'g':
            if self.conf.LSSexperiment == 'LSST':
                return self.conf.N_bins 
            elif self.conf.LSSexperiment == 'unwise_blue':
                return 1
            elif self.conf.LSSexperiment == 'custom':
                return 1
            else: 
                raise Exception("LSS experiment choice not defined.")
        elif tag in ['tSZ', 'CIB', 'isw_lin','lensing','pCMB']:
            return 1   
        elif tag in ['m','taud','ml','vr','vt','e']:
            return self.conf.N_bins
        else:
            raise Exception("Tag not supported")
        
        

      
        


