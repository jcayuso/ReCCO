#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:56:42 2020

@author: jcayuso
"""

import numpy as np
from scipy.interpolate import interp1d
import time
import redshifts
from math import lgamma
import loginterp
import common as c
import copy
from scipy.linalg import cholesky
from scipy import linalg
import healpy
from scipy.ndimage import gaussian_filter
from numpy import linalg as la
import numpy as np

class estimator(object):
    
    #####################################################################
    ############ CLASS SET UP AND CL TOOLS
    #####################################################################
    
    def __init__(self, data_lmax = None, hashstr = None, conf_module = None) :
        import time
        
        if hashstr != None :
            self.basic_conf_dir = c.load_basic_conf(hashstr)
            self.conf = c.dict_to_obj(self.basic_conf_dir)
        elif conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = c.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a hash string or configuration module to locate precomputed data")
        
        if data_lmax == None:
            raise Exception ("Please provide data_lmax (This is not necessarily equal to the estimator lmax)")
        else:
            self.data_lmax = data_lmax
                    
        self.Cls = {}
        self.sims = {}
        self.haar = {}
        self.cs = {}
        self.realization = 0
        self.ml_map_bin_range = 0
        self.llcut = 0
        self.type = 0
        
        self.zb = redshifts.binning(basic_conf_obj = self.conf)
        self.csm = self.zb.csm
        self.deltachi = self.zb.deltachi
        self.nbin = self.conf.N_bins
        
        self.N_fine_modes = self.conf.N_bins
        self.lss = 'g'
        
        self.realnum = 1
    
        print("Default lss = 'g' . Modify with set_lss method.")
        print('Default N_fine_modes = '+str(self.N_fine_modes)+'. Modify with set_Nfine method.')
        
        #Some temporary parameters for frequency dependent noise from SO
        
        self.SO_FREQS = np.array([27, 39, 93, 145, 225, 280])
        self.dTs   = np.array([71, 36, 8.0, 10, 22, 54])
        self.beams = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
        
        
    def set_Nfine(self, Nfine):
        self.N_fine_modes= Nfine
        print('N_fine_modes reset to '+str(self.N_fine_modes))
        
    def set_lss(self, lss_string):
        self.lss= lss_string
        print('lss reset to '+str(self.lss))
        
    def load_theory_Cl(self,tag1,tag2):
        if c.exists(self.basic_conf_dir,'Cl_'+tag1+'_'+tag2+'_lmax='+str(self.data_lmax), dir_base = 'Cls'):
            return c.load(self.basic_conf_dir,'Cl_'+tag1+'_'+tag2+'_lmax='+str(self.data_lmax), dir_base = 'Cls')
        else:
            C = c.load(self.basic_conf_dir,'Cl_'+tag2+'_'+tag1+'_lmax='+str(self.data_lmax), dir_base = 'Cls')
            return np.transpose(C, axes =[0,2,1])

    def load_L(self,):
        return c.load(self.basic_conf_dir,'L_sample_lmax='+str(self.data_lmax), dir_base = 'Cls')
    
    def set_theory_Cls(self, add_ksz = True, add_ml = True, use_cleaned = False, frequency = None, get_haar = False):
        start = time.time()

        #frequency is only used if use_cleaned = False. If None, you have primary CMB + a simple gaussian white noise with beam. If you
        #use a frequency, at the moment it should be a SO one.
        self.Cls['lss-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,self.lss),self.load_L())
        self.Cls['taud-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud',self.lss),self.load_L())
        self.Cls['taud-taud'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud','taud'),self.load_L())
        self.Cls['v-v'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('v','v'), self.load_L())
        self.Cls['taud-v'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('taud','v'), self.load_L())
        self.Cls['vt-vt'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('vt', 'vt'), self.load_L())
        self.Cls['ml-ml'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ml', 'ml'), self.load_L())
        self.Cls['ml-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ml', self.lss), self.load_L())
        self.Cls['lensing-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('lensing',self.lss), self.load_L())
        
        self.Cls['pCMB-pCMB'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('pCMB','pCMB'), c.load(self.basic_conf_dir,'L_pCMB_lmax='+str(self.data_lmax), dir_base = 'Cls'))
        self.Cls['lensing-lensing'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('lensing','lensing'), self.load_L())
        self.Cls['kSZ-kSZ'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('kSZ','Nfine_'+str(self.N_fine_modes)),self.load_L())
        self.Cls['ML-ML'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ML','Nfine_'+str(self.N_fine_modes)),self.load_L())
                
        if use_cleaned == True:
            print("Using clean TT")
            self.Ttag = 'Tc'
            self.Cls['T-T']   =  loginterp.log_interpolate_matrix(self.load_theory_Cl('Tc','Tc'),self.load_L())
            self.Cls['T-lss'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('Tc',self.lss),self.load_L())
            self.Cls['lss-T'] =  np.transpose(self.Cls['T-lss'], axes =[0,2,1])
            
        else:

            if frequency is None:
            
                self.Ttag = 'T0'
                self.beam = self.conf.beamArcmin_T*np.pi/180./60.
                self.dT   = self.conf.noiseTuKArcmin_T*np.pi/180./60./self.conf.T_CMB

                self.Cls['T-T'] = self.Cls['pCMB-pCMB']\
                                    +self.CMB_noise(np.arange(self.data_lmax+1),self.beam, self.dT)
                self.Cls['T-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('isw_rs',self.lss), self.load_L())
                self.Cls['lss-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,'isw_rs'), self.load_L())
            
            else:
                try:
                    idx = np.where(self.SO_FREQS ==frequency)[0][0]
                except:
                    raise Exception("Please use one of the S0 frequencies")
                
               
                self.Ttag = 'T('+str(frequency)+')'
                self.beam = self.beams[idx]*np.pi/180./60.
                self.dT   = self.dTs[idx]*np.pi/180./60./self.conf.T_CMB

                self.Cls['T-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('T('+str(frequency)+')','T('+str(frequency)+')'),self.load_L())\
                                    +self.CMB_noise(np.arange(self.data_lmax+1),self.beam, self.dT)
                self.Cls['T-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('T('+str(frequency)+')',self.lss), self.load_L())
                self.Cls['lss-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,'T('+str(frequency)+')'), self.load_L())
            

        if add_ksz:
            self.Cls['T-T'] += self.Cls['kSZ-kSZ']
        if add_ml:
            self.Cls['T-T'] += self.Cls['ML-ML']
            
        if get_haar:
            
            print("Getting fine mode spectra")
            
            kmax =  self.N_fine_modes
            H = self.get_haar(kmax)
            Bc = self.zb.binmatrix(self.N_fine_modes, self.conf.N_bins)
            
            basic_conf_high = copy.deepcopy(self.basic_conf_dir)
            basic_conf_high['N_bins'] = self.N_fine_modes
            
            Ctaudlss_sample = c.load(basic_conf_high,'Cl_taud_'+self.lss+'_lmax='+str(self.data_lmax), dir_base = 'Cls')
            Cmllss_sample   = c.load(basic_conf_high,'Cl_ml_'+self.lss+'_lmax='+str(self.data_lmax), dir_base = 'Cls')
            
            Chaartaudlss_sample = np.swapaxes(np.dot(np.dot(H,Ctaudlss_sample), np.transpose(Bc)),0,1)
            Chaarmllss_sample   = np.swapaxes(np.dot(np.dot(H,Cmllss_sample), np.transpose(Bc)),0,1)
            
            
            self.Cls['haartaud-'+self.lss] = Chaartaudlss_sample
            self.Cls['haarml-'+self.lss]   = Chaarmllss_sample
        

            idx_cut = np.where(self.load_L()  <100)[0][-1]
            Cvvfine =  loginterp.log_interpolate_matrix(c.load(basic_conf_high,'Cl_v_v_lmax='+str(self.data_lmax), dir_base = 'Cls')[:idx_cut,:,:] ,self.load_L()[:idx_cut])
            Cvtvtfine =  loginterp.log_interpolate_matrix(c.load(basic_conf_high,'Cl_vt_vt_lmax='+str(self.data_lmax), dir_base = 'Cls')[:idx_cut,:,:] ,self.load_L()[:idx_cut])
            self.Cls['v_fine-v_fine'] = Cvvfine
            self.Cls['vt_fine-vt_fine'] = Cvtvtfine
            
            end = time.time()
            
            print("Cls loaded in t="+str(end-start))
        

    def set_Cl(self, tag1, tag2, CL, extra =''):
        self.data_lmax = CL.shape[0]-1
        if CL.shape[0] != self.data_lmax+1:
            raise Exception (" Input CL has to have first axis of length lmax+1")
        self.Cls[tag1+'-'+tag2+'-'+extra] = CL[self.load_L()]
        
    def load_tau(self,):
        return c.load(self.basic_conf_dir,'tau_binned', dir_base = '')
    
    
    
    #####################################################################
    ############ NOISE AND BIAS CALCULATION
    #####################################################################
    
    def get_haar(self,kmax):
        
        if 'bin2haar' in self.haar:
            return self.haar['bin2haar']
        else:
            H = self.zb.bin2haar(kmax)
            self.haar['bin2haar'] = H
            return self.haar['bin2haar']

    def wigner_symbol(self, ell, ell_1,ell_2):
         
        if not ((np.abs(ell_1-ell_2) <= ell) and (ell <= ell_1+ell_2)):
            return 0
     
        J = ell +ell_1 +ell_2
        if J % 2 != 0:
            return 0
        else:
            g = int(J/2)*1.0
            w = (-1)**(g)*np.exp((lgamma(2.0*g-2.0*ell+1.0)+lgamma(2.0*g-2.0*ell_1+1.0)+lgamma(2.0*g-2.0*ell_2+1.0)\
                                  -lgamma(2.0*g+1.0+1.0))/2.0 +lgamma(g+1.0)-lgamma(g-ell+1.0)-lgamma(g-ell_1+1.0)-lgamma(g-ell_2+1.0))
            
            return w
    
    def CMB_noise(self,Ls, BEAM, DT):
        Nl_CMB_T = np.zeros((len(Ls),1,1))
        with np.errstate(over='ignore'):
            Nl_CMB_T[:,0,0] = (DT**2.)*np.exp(Ls*(Ls+1.)*(BEAM**2.)/8./np.log(2.))

        return Nl_CMB_T
    
    def Cl_cal(self,Ls):
        ls = np.arange(self.data_lmax)
        Ncal_0 = np.exp(-(ls/10)**2)
        A = self.conf.sigma_cal/(np.sum((2.0*ls+np.ones(len(ls)))/(4*np.pi)*Ncal_0))
        Ncal = A*np.exp(-(ls/10)**2)
            
        return Ncal[Ls]
    
    
    def f(self, tag, alpha, gamma, l, l1, l2):
        
        if tag == 'v':
            C = self.Cls['taud-lss'][:,gamma,alpha]
            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*self.zb.deltachi*C[l2]*self.wigner_symbol(l, l1, l2)
            
        elif tag == 'v_fine':
            C = loginterp.log_interpolate(self.Cls['haartaud-'+self.lss][:,gamma,alpha],self.load_L())
            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
            
        elif tag == 'vt':
            C = self.Cls['ml-lss'][:, gamma, alpha]
            factor = 0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) * np.sqrt((2 * l + 1.0) * (2 * l1 + 1.0) * (2 * l2 + 1.0) / 4.0 / np.pi)*self.zb.deltachi * C[l2] * self.wigner_symbol(l, l1, l2)
           
        elif tag == 'vt_fine':
                        
            C = loginterp.log_interpolate(self.Cls['haarml-'+self.lss][:,gamma,alpha],self.load_L())
            factor = 0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) * np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
               
        elif tag =='cal':
            C = self.Cls['T-lss'][:,0,alpha]
            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
        elif tag =='pCMB':
            C = self.Cls['lensing-lss'][:,0,alpha]
            factor =  0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) *np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
        
        else:
            raise Exception("Weight f not supported for "+tag)
        
        return factor


    def g(self, tag, alpha, l, l1, l2, dndz = None):
        
        if dndz == None:
        
            CTT     = self.Cls['T-T'][:,0,0]
            CTlss   = self.Cls['T-lss'][:,0,alpha]
            Clsslss = self.Cls['lss-lss'][:,alpha,alpha]
    
            G =  (CTT[l2]*Clsslss[l1]*self.f(tag, alpha, alpha, l, l1, l2)-((-1)**(l+l1+l2))*CTlss[l1]*CTlss[l2]*self.f(tag, alpha, alpha, l, l2, l1))\
             /(CTT[l1]*CTT[l2]*Clsslss[l1]*Clsslss[l2] - (CTlss[l1]**2)*(CTlss[l2]**2))
                        
            return G
        
        elif dndz == 'unwise':
        
            CTT     = self.Cls['T-T'][:,0,0]
            CTlss   = self.Cls['T-lss'][:,0,0]
            Clsslss = self.Cls['lss-lss'][:,0,0]
    
            G =  (CTT[l2]*Clsslss[l1]*self.f(tag, 0, alpha, l, l1, l2)-((-1)**(l+l1+l2))*CTlss[l1]*CTlss[l2]*self.f(tag, 0, alpha, l, l2, l1))\
             /(CTT[l1]*CTT[l2]*Clsslss[l1]*Clsslss[l2] - (CTlss[l1]**2)*(CTlss[l2]**2))
                        
            return G
        
        else:
            raise Exception("dndz not valid")
    
    def cs1_alpha_gamma(self, lmax, tag_g, tag_f, alpha, gamma, ell, dndz = None):
        
        if str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell) in self.cs:
            return self.cs[str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell) ]
        else:
        
            L = np.unique(np.append(np.geomspace(2,lmax,300).astype(int),lmax))
            
            #First, let's avoid calculating sums that are identically zero
            
            if tag_f == 'v_fine':
                Lnz =np.where(loginterp.log_interpolate(self.Cls['haartaud-'+self.lss][:,gamma,alpha],self.load_L()) != 0.0)
            else:
                Lnz = L
    
            L_int = L[np.in1d(L, Lnz)]
            
            if len(L_int) == 0:
                c = 0
            else:
                a = []
                
                for l1_id, ell_2 in enumerate(L_int):
                    
                    terms = 0
                    
                    for ell_1 in np.arange(np.abs(ell_2-ell),ell_2+ell+1):
                        if ell_1 > lmax or ell_1 <2:   #triangle rule
                            continue
                        
                        if dndz == None:
                            terms += self.f(tag_f, alpha, gamma, ell, ell_1, ell_2)*self.g(tag_g, alpha, ell, ell_1, ell_2)
                        elif dndz == 'unwise':
                            terms += self.f(tag_f, 0, gamma, ell, ell_1, ell_2)*self.g(tag_g, alpha, ell, ell_1, ell_2, dndz = 'unwise')
                        else:
                            raise Exception("dndz not valid")
                      
                    a.append(terms)
                
                if len(L_int) == 1:
                    c = np.asarray(a)[0]
                elif (len(L_int) == 2 or len(L_int) == 3):
                    I = interp1d(L_int ,np.asarray(a), kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
                    c =   np.sum(I)
                else:
                    #Ignore last couple ell cause they can be problematic, regardless of lmax
                    I = interp1d(L_int[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
                    c =   np.sum(I)
                    
            
            self.cs[str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell) ] = c
            
            return c
            
        
    
    def cs2_alpha_gamma(self, lmax, tag_g, alpha, gamma, ell, dndz = None):    # This is used for the noise that comes from non-statistically asymmetric contributions
       
        L = np.unique(np.append(np.geomspace(2,lmax,300).astype(int),lmax))
        a = []
        
    
        if dndz == None:
    
            Clsslss_alpha_gamma  = self.Cls['lss-lss'][:,alpha,gamma]
            CTT                  = self.Cls['T-T'][:,0,0]
            CTlss_alpha          = self.Cls['T-lss'][:,0,alpha]
            CTlss_gamma          = self.Cls['T-lss'][:,0,gamma]
    
            for l1_id, ell_1 in enumerate(L):
                
                terms = 0
                
                for ell_2 in np.arange(np.abs(ell_1-ell),ell_1+ell+1):
                    if ell_2 > lmax or ell_2 <2:   #triangle rule
                        continue
                    g_alpha = self.g(tag_g, alpha, ell, ell_1, ell_2)
                    g_gamma_1 = self.g(tag_g, gamma, ell, ell_1, ell_2)
                    g_gamma_2 = self.g(tag_g, gamma, ell, ell_2, ell_1)
                    terms  += g_alpha*(g_gamma_1*CTT[ell_1]*Clsslss_alpha_gamma[ell_2]+((-1)**(ell+ell_1+ell_2))*g_gamma_2*CTlss_alpha[ell_2]*CTlss_gamma[ell_1] )
                
                a.append(terms)
        
        elif dndz == 'unwise':
            
            Clsslss_alpha_gamma  = self.Cls['lss-lss'][:,0,0]
            CTT                  = self.Cls['T-T'][:,0,0]
            CTlss_alpha          = self.Cls['T-lss'][:,0,0]
            CTlss_gamma          = self.Cls['T-lss'][:,0,0]
    
            for l1_id, ell_1 in enumerate(L):
                
                terms = 0
                
                for ell_2 in np.arange(np.abs(ell_1-ell),ell_1+ell+1):
                    if ell_2 > lmax or ell_2 <2:   #triangle rule
                        continue
                    g_alpha = self.g(tag_g, alpha, ell, ell_1, ell_2, dndz = dndz)
                    g_gamma_1 = self.g(tag_g, gamma, ell, ell_1, ell_2, dndz = dndz)
                    g_gamma_2 = self.g(tag_g, gamma, ell, ell_2, ell_1, dndz = dndz)
                    terms  += g_alpha*(g_gamma_1*CTT[ell_1]*Clsslss_alpha_gamma[ell_2]+((-1)**(ell+ell_1+ell_2))*g_gamma_2*CTlss_alpha[ell_2]*CTlss_gamma[ell_1] )
                
                a.append(terms)
                
        else:
            raise Exception("dndz not valid")
                        
                    
        I = interp1d(L[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
        
        
        return np.sum(I)
    
    
    def Noise_iso(self, lmax, tag_g, alpha, gamma, ell, dndz = None):
        Nalpha  = (2*ell+1)/self.cs1_alpha_gamma(lmax, tag_g, tag_g, alpha, alpha, ell, dndz = dndz)
        Ngamma  = (2*ell+1)/self.cs1_alpha_gamma(lmax, tag_g, tag_g, gamma, gamma, ell, dndz = dndz)
        return self.cs2_alpha_gamma(lmax,tag_g, alpha, gamma, ell, dndz = dndz)*Nalpha*Ngamma/(2*ell+1)
    
    def R(self, lmax, tag_g, tag_f, alpha,gamma,ell, dndz = None):
        
        if tag_f in ['v_fine','vt_fine']:
            if gamma < self.nbin:
                return 0

        num = self.cs1_alpha_gamma(lmax, tag_g , tag_f, alpha, gamma, ell, dndz = dndz)
        den = self.cs1_alpha_gamma(lmax, tag_g , tag_g, alpha, alpha, ell, dndz = dndz)
        return  num/den

    
    def Noise_a_from_b(self, lmax, tag_g, tag_f1, tag_f2, alpha, gamma, ell, dndz = None):
        
        if tag_g in ['v_fine','vt_fine']:
            raise Exception ('tag not supported as primary tag. Select est_tag among v or vt') #Update this message when new estimators are considered
            
        if tag_f1 not in ['v_fine','vt_fine'] and tag_f2 not in ['v_fine','vt_fine']:
            
            if tag_f1 == 'cal' or tag_f2 == 'cal':
                if tag_f1 != tag_f2:
                    raise Exception ('Calibration bias only supported when b1_tag = b2_tag = cal')
                else:
                    C = self.Cl_cal(ell)*np.ones((1,1))
            else:
                if tag_f1 == 'pCMB':
                    C = self.Cls['pCMB-pCMB'][ell]
                else:
                    C = loginterp.log_interpolate_matrix(self.load_theory_Cl(tag_f1, tag_f2), self.load_L())[ell]
                
            N1,N2 = C.shape
            Noise = 0
            R1 = np.zeros(N1)
            R2 = np.zeros(N2)
            
            for i in np.arange(N1):
                R1[i] = self.R(lmax, tag_g, tag_f1, alpha,i,ell, dndz = dndz)
            for j in np.arange(N1):
                R2[j] = self.R(lmax, tag_g, tag_f2, gamma,j,ell, dndz = dndz)
                    
            Noise = np.dot(np.dot(R1,C),R2)
        
            return Noise
        
        else:
                         
            C = self.Cls[tag_f1+'-'+tag_f2][ell,:,:]
                
            N1,N2 = C.shape
            Noise = 0
            R1 = np.zeros(N1)
            R2 = np.zeros(N2)
            
            for i in np.arange(N1):
                R1[i] = self.R(lmax, tag_g, tag_f1, alpha,i,ell, dndz = dndz)
            for j in np.arange(N1):
                R2[j] = self.R(lmax, tag_g, tag_f2, gamma,j,ell, dndz = dndz)
            
            H = self.zb.bin2haar(self.N_fine_modes)
            
            Noise = np.dot(np.dot(   np.dot(R1,H)  ,C),   np.dot(R2,H)  )
                    
            return Noise
        

    #####################################################################
    ############ NOISE wrapper functions
    #####################################################################
            
    def Noise_iso_alpha(self,lmax ,est_tag, alpha,L, dndz = None):
        N = np.zeros(len(L))
        for lid, l in enumerate(L):
            N[lid] = self.Noise_iso(lmax, est_tag, alpha, alpha, l, dndz = dndz)
        return N
    
    def Noise_iso_ell(self,lmax ,est_tag, ell, dndz = None):
        N = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                N[alpha,gamma] = self.Noise_iso(lmax, est_tag, alpha, gamma, ell, dndz = dndz)
        return N

    
    def R_ell(self,lmax ,est_tag, bias_tag, ell, dndz = None):
        R = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                R[alpha,gamma] =  self.R(lmax, est_tag, bias_tag,alpha,gamma,ell, dndz = dndz)
        return R
           
    def Noise_a_from_b_alpha(self,lmax, est_tag, b1_tag, b2_tag, alpha, L, dndz = None):
        N = np.zeros(len(L))
        for lid, l in enumerate(L):
            N[lid] = self.Noise_a_from_b(lmax, est_tag, b1_tag, b2_tag, alpha, alpha, l, dndz = dndz)
        return N
    
    def Noise_a_from_b_ell(self,lmax, tag_g, tag_f1, tag_f2, ell, dndz = None):
        N = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                N[alpha,gamma] = self.Noise_a_from_b(lmax, tag_g, tag_f1, tag_f2, alpha, gamma, ell, dndz = dndz)
        return N
    
    def Noise_a_from_b_matrix(self,lmax, tag_g, tag_f1, tag_f2, L, dndz = None):
        for l in L:
            N = np.zeros((self.nbin,self.nbin))
            for alpha in np.arange(self.nbin):
                for gamma in np.arange(self.nbin):
                    N[alpha,gamma] = self.Noise_a_from_b(lmax, tag_g, tag_f1, tag_f2, alpha, gamma, l, dndz = dndz)
            c.dump(self.basic_conf_dir, N,'N_'+str(tag_g)+'_'+str(tag_f1)+'_'+str(tag_f2)+'_Nfine'+str(self.N_fine_modes)+'_l='+str(l)+'_lmax='+str(lmax), dir_base = 'analysis')
            print('Done l = '+str(l))
        return
    

    def pmode_vv(self, lmax, L, fine = True, cal = True, dndz = None):
                
        SN = np.zeros((len(L),self.nbin))
        
        #Approximation for R and Cn, practically ell independent
        R  = self.R_ell(lmax ,'v', 'v', 2, dndz = dndz)
        Cn = self.Noise_iso_ell(lmax ,'v', 2, dndz = dndz)
              
        for lid, ell in enumerate(L):
            if fine:
                Cn += c.load(self.basic_conf_dir,'N_v_v_fine_v_fine_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
            if cal:
                Cn += c.load(self.basic_conf_dir,'N_v_cal_cal_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
                                    
            #signal
            C = self.Cls['v-v'][ell]
            Cs = np.dot(np.dot(R,C),np.transpose(R))
        
            #First diagonalization
            
            w1,v1 = np.linalg.eigh(Cn)
            
            R1 = np.transpose(v1)
            R2 = np.zeros_like(Cn)
            
            for i in np.arange(self.nbin):
                R2[i,i] = 1.0/np.sqrt(w1[i])
                
            #second diagonalization
                
            R21 = np.dot(R2,R1)
            Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
            w3,v3 = np.linalg.eigh(Cs_p)
            R3 = np.transpose(v3)
            
            Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                
            for j in np.arange(self.nbin):
                SN[lid,j] = Cs_pp[j,j]
            
        return L, SN

    def pmode_vtvt(self, lmax, L, fine = True, cal = True, dndz = None):
                
        SN = np.zeros((len(L),self.nbin))
        
        #Approximation for R and Cn, practically ell independent
        R  = self.R_ell(lmax ,'vt', 'vt', 2, dndz = dndz)
        Cn = self.Noise_iso_ell(lmax ,'vt', 2, dndz = dndz)
              
        for lid, ell in enumerate(L):
            if fine:
                Cn += c.load(self.basic_conf_dir,'N_vt_vt_fine_vt_fine_Nfine512_l='
                             +str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
            if cal:
                Cn += c.load(self.basic_conf_dir,'N_vt_cal_cal_Nfine512_l='+str(ell)
                             +'_lmax='+str(lmax), dir_base = 'analysis')
                                    
            #signal
            C = self.Cls['vt-vt'][ell]
            Cs = np.dot(np.dot(R,C),np.transpose(R))
        
            #First diagonalization
            
            w1,v1 = np.linalg.eigh(Cn)
            
            R1 = np.transpose(v1)
            R2 = np.zeros_like(Cn)
            
            for i in np.arange(self.nbin):
                R2[i,i] = 1.0/np.sqrt(w1[i])
                
            #second diagonalization
                
            R21 = np.dot(R2,R1)
            Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
            w3,v3 = np.linalg.eigh(Cs_p)
            R3 = np.transpose(v3)
            
            Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                
            for j in np.arange(self.nbin):
                SN[lid,j] = Cs_pp[j,j]
            
        return L, SN

    
    def pmode_gg(self,Ns,lmax, dndz = None):
                
        SN = []
        
        for N in Ns:
            
            SN_N = 0
            
            basic_conf_N = copy.deepcopy(self.basic_conf_dir)
            basic_conf_N['N_bins'] = N
            
        
            Nshot = loginterp.log_interpolate_matrix(c.load(basic_conf_N,
                                                            'Nlshot_g_g_lmax='+str(self.data_lmax),
                                                            dir_base = 'Cls'),self.load_L())
            Cgg = loginterp.log_interpolate_matrix(c.load(basic_conf_N,
                                                          'Cl_g_g_noshot_lmax='+str(self.data_lmax),
                                                          dir_base = 'Cls'),self.load_L())
            
            for ell in np.arange(1,lmax+1):
                            
                #signal and noise for ell
                Cs = Cgg[ell,:,:]
                Cn = Nshot[ell,:,:]
            
                #First diagonalization
                
                w1,v1 = np.linalg.eigh(Cn)
                
                R1 = np.transpose(v1)
                R2 = np.zeros_like(Cn)
                
                for i in np.arange(N):
                    R2[i,i] = 1.0/np.sqrt(w1[i])
                    
                #second diagonalization
                    
                R21 = np.dot(R2,R1)
                Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
                w3,v3 = np.linalg.eigh(Cs_p)
                R3 = np.transpose(v3)
                #R321 = np.dot(R3,R21)
                
                Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                    
                for j in np.arange(N):
                    SN_N += Cs_pp[j,j]*(2*ell+1)
                    
            SN.append(SN_N)
            
            
        return (Ns,SN)
    
    
    def isPD(self,B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlg:
            return False
    
    #####################################################################
    ############ GAUSSIAN SIMULATIONS
    #####################################################################
    def nearestPD(self,A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        try:
            _ = la.cholesky(A3)
            return A3
        except la.LinAlg:
            return False
            

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3

    def covmat_sample_ell_hijack(self, labels, lid):
        ns = []
        lsparse = self.load_L()

        diag32 = np.diag(np.ones(32))
        zerosa = diag32
        zerosb = np.ones((32,32))-diag32
        
        rat = np.array([0.99999, 0.99999, 0.99999, 0.99997, 0.99996, 0.99991, 0.99977,
       0.99948, 0.99905, 0.99855, 0.99775, 0.99682, 0.99576, 0.99437,
       0.99317, 0.99301, 0.99424, 0.99509, 0.99577, 0.99624, 0.9966 ,
       0.99686, 0.99705, 0.99718, 0.99728, 0.99734, 0.99736, 0.99737,
       0.99737, 0.99737, 0.99736, 0.99734, 0.99731, 0.99726, 0.99716,
       0.99698, 0.99669, 0.99633, 0.99594, 0.99555, 0.9952 , 0.9949 ,
       0.99455, 0.9939 , 0.99301, 0.99209, 0.99122, 0.9904 , 0.98916,
       0.98753, 0.98585, 0.98412, 0.98183, 0.97918, 0.97627, 0.97275,
       0.96868, 0.96406, 0.95861, 0.95238, 0.94509, 0.93674, 0.92705,
       0.91593, 0.90314, 0.88858, 0.87185, 0.85299, 0.83151, 0.80722,
       0.78002, 0.74946, 0.71558, 0.6782 , 0.63699, 0.59205, 0.59195])
        
        for label1 in labels:
            n1 = []
            for label2 in labels:
                covi = self.load_theory_Cl(label2,label1)[lid,:,:]
                if lid > 19:
                    covi = covi*zerosa+covi*zerosb*rat[lid-20]
                
                n1.append(covi) # order of labels is okay, dont change

            ns.append(np.vstack(n1))
                     
        return np.hstack(ns)

    
    def covmat_sample_ell(self, labels, lid):
        ns = []

        for label1 in labels:
            n1 = []
            for label2 in labels:
                n1.append(self.load_theory_Cl(label2,label1)[lid,:,:]) # order of labels is okay, dont change

            ns.append(np.vstack(n1))
                     
        return np.hstack(ns)


    def covmat_healpy(self, labels, lswitch):
        Cls = []
        num = len(labels)
        obs = []
        
        
        for idx in np.arange(num):
            dim_l = self.load_theory_Cl(labels[idx],labels[idx]).shape[1]
            for d in np.arange(dim_l):
                obs.append((labels[idx],d))
        
        
        for i in np.arange(len(obs)):
            for j in np.arange(i,len(obs)):
                Cls.append(loginterp.log_interpolate(self.load_theory_Cl(obs[i][0],
                                                                         obs[j][0])[:,obs[i][1],
                                                                                    obs[j][1]],
                                                     self.load_L())[:lswitch])

                                                     
        return Cls
        
    
    def check_symmetric(self, a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)
    
       
    def check_correlationmatrix(self,cl):
        
        
        for lid, l in enumerate(self.load_L(self.data_lmax)):
    
            symmetric = self.check_symmetric(cl[l,:,:])
            
            if symmetric == False:
                return print("matrix not symmetric at l="+str(l))
        
            eigenvalues = linalg.eigvals(cl[l,:,:])
        
            if np.any(eigenvalues<1e-25)==True:
                return print("matrix has negative eigenvalues at l="+str(l))
        
            cholesky(cl[l,:,:])

        return print("covmat is symmetric, is positive definite, and cholseky decomposition defined at all l")
    
    
    def alm_maker_cholesky(self,labels,lmax):
        
        # Note, inputs are the un-interpolated spectra.
        
        # Add possibility of correlated tracers and flag to output map instead of alms.
        # Add sequential writing to file in order to handle large number of bins and nside.

        dim_list = []
        
        for hadi in [0]:
            
            for lab in labels:
                dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
            tot_dim = np.sum(dim_list)
            
            start = time.time()
            
            print("cholesky")
            lsparse = self.load_L()
            lmax_in = len(lsparse)
            almsize = healpy.Alm.getsize(lmax)
            alms = np.zeros((tot_dim,almsize),dtype=complex)
            L = np.zeros((lmax_in,tot_dim,tot_dim))
        
            L_out = loginterp.log_interpolate_matrix(L, lsparse)
 
            for lid in range(lmax_in):
            
                if lsparse[lid] ==1:
                    continue

                covmats = self.covmat_sample_ell_hijack(labels, lid).copy()
                
                try:
                    L[lid,:,:] = cholesky(covmats, lower = True)
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    print('cholesky fails at l='+str(lsparse[lid])+' lid=',str(lid),' for ',str(labels))
                    print('calculating nearest postiive definite matrix')

            L_out = loginterp.log_interpolate_matrix(L, lsparse)
            
            print("generating realization")
            
            almsize_temp150 = healpy.Alm.getsize(150)
            alms_temp150 = np.zeros((tot_dim,almsize_temp150),dtype=complex)
            almsize_temp1500 = healpy.Alm.getsize(1500)
            alms_temp1500 = np.zeros((tot_dim,almsize_temp1500),dtype=complex)

            ind_forl=[]
            for l in range(lmax):
                if c.exists(self.basic_conf_dir,'cholesky_inds_real='+str(self.realization)
                            +'_l='+str(l),dir_base = 'sims3'):
                    if np.mod(l,1000)==0: print('l=',l,' Cholesky exists...')
                    inds_ = c.load(self.basic_conf_dir,'cholesky_inds_real='+str(self.realization)
                                   +'_l='+str(l),dir_base = 'sims3')
                    alms_ = c.load(self.basic_conf_dir,'cholesky_alms_real='+str(self.realization)
                                   +'_l='+str(l),dir_base = 'sims3')
                    alms[:,inds_]=alms_
                    continue

                print('simulating l=',l)
                ind_forl_flag = False
                for m in range(l):
                    vals = L_out[l,:,:]@np.random.normal(size=tot_dim) + (1j)*L_out[l,:,:]@np.random.normal(size=tot_dim)
                    ind=healpy.Alm.getidx(lmax, l, m)
                    alms[:,ind]=vals/np.sqrt(2)
                    if ind_forl_flag==False:
                        ind_forl = [ind]
                        ind_forl_flag = True
                    else:
                        ind_forl = np.concatenate([ind_forl,[ind]])
                    
                    if l<150:
                        ind_temp150=healpy.Alm.getidx(150, l, m)
                        alms_temp150[:,ind_temp150]=vals/np.sqrt(2)
                    if l<1500:
                        ind_temp1500=healpy.Alm.getidx(1500, l, m)
                        alms_temp1500[:,ind_temp1500]=vals/np.sqrt(2)
                    
                if l==150:
                    c.dump(self.basic_conf_dir,alms_temp150,
                            'cholesky_alms_150_real='+str(self.realization)
                            +'_lmax=150',dir_base = 'sims3')
                if l==1500:
                    c.dump(self.basic_conf_dir,alms_temp1500,
                            'cholesky_alms_1500_real='+str(self.realization)
                            +'_lmax=1500',dir_base = 'sims3')

                c.dump(self.basic_conf_dir,ind_forl,
                        'cholesky_inds_real='+str(self.realization)
                        +'_l='+str(l),dir_base = 'sims3')
                c.dump(self.basic_conf_dir,alms[:,ind_forl],
                        'cholesky_alms_real='+str(self.realization)
                        +'_l='+str(l),dir_base = 'sims3')
    
            end = time.time()
            
            print("cholesky sims done in t="+str(end-start))
        
            results = []
            base = 0
            for i in np.arange(len(labels)):
                
                results.append(alms[base:base+dim_list[i],:])

                base += dim_list[i]
                
            return results
    
    def alm_maker_healpix(self,labels,lswitch):

        if 1==1:
            start = time.time()
            # The number of maps nbin should be 2,4,8,16,32,64,128 etc.
        
            # Format the covariance matrix for synalm
        
            dim_list = []
        
            for lab in labels:
                dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
    
            print("formatting covmat")
            
            if c.exists(self.basic_conf_dir,
                        'healpy_alms_real='+str(self.realization),
                        dir_base = 'sims3'):
                print('healpy_alms_real='+str(self.realization)+'... exists')
                synalms = c.load(self.basic_conf_dir,
                                 'healpy_alms_real='+str(self.realization),
                                 dir_base = 'sims3')
            else:
                cltot = self.covmat_healpy(labels, lswitch)
           
                print("generating realization for healpy alm")
            
                synalms = healpy.synalm(cltot,lswitch,new=False)
                
                c.dump(self.basic_conf_dir,synalms,
                        'healpy_alms_real='+str(self.realization),
                       dir_base = 'sims3')

            results = []
            base = 0
            for i in np.arange(len(labels)):
                results.append(synalms[base:base+dim_list[i],:])
                base += dim_list[i]

############################################################
#             [[c.dump(self.basic_conf_dir,alms[i,j],
#                    'healpix_alm_real='+str(self.realization)
#                    +'_'+str(i)+'_'+str(j)+'.p',dir_base = 'sims3')
#               for i in range(len(synalms))] for j in range(len(synalms[0]))]
############################################################
            end = time.time()
            
            print("healpix sims done in t="+str(end-start))
            

            return results
    
    
    
    def alm_maker_hybrid(self, labels, lswitch, lmax):
        print('entering hybrid')# (only healpix)')
        alms = self.alm_maker_cholesky(labels,lmax)
        alms_low = self.alm_maker_healpix(labels,lswitch)
        
        for l in range(0,lswitch):
            for m in range(0,l+1):
                lmind = healpy.Alm.getidx(lmax, l, m)
                lmind_low = healpy.Alm.getidx(lswitch, l, m)
                
                for idx in np.arange(len(labels)):
                    alms[idx][:,lmind] = alms_low[idx][:,lmind_low]
                    alms[idx][:,lmind] = alms_low[idx][:,lmind_low]
        print('exciting hybrid')
        return alms
    
    def get_maps_and_alms(self,labels,nside,lmax):
        
        npix = healpy.nside2npix(nside)
        dim_list = []
        map_list = []
        
        for lab in labels:
            dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
            map_list.append(np.zeros((self.load_theory_Cl(lab,lab).shape[1],npix)) )
                     
        lswitch = 40
        
        alms = self.alm_maker_hybrid(labels, lswitch, lmax)
        
        print('done with alms')
        return map_list, alms
        
    
    def bare_quad_est(self,nside, n_level, nbin, bin_width, Tfield, lssfield, beam_window, cllsslssbinned, ClTT, clTlssbinned, cltaudlssbinned):

        lcut = 3*nside
        ones =  np.ones(3*nside)
        cut = np.where(np.arange(3*nside)<lcut, 1, 1e-30)
        
        cllsslssbinned[0,:,:] = 1e15
        xizeta_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))
            
        dTlm = healpy.map2alm(Tfield)
        dTlm_beamed = healpy.almxfl(dTlm,(1./beam_window)*cut)
        
        
        for i in range(nbin):
            
            Cltaudd = cltaudlssbinned[:,i,i]*bin_width
            Cldd = cllsslssbinned[:,i,i]
            ClTd = clTlssbinned[:,0,i]
    
            dlm_in = healpy.almxfl(healpy.map2alm(lssfield[i]),cut)
            
            dTlm_xi = healpy.almxfl(dTlm_beamed,np.divide(ones, ClTT, out=np.zeros_like(ones), where=ClTT!=0))
            dlm_zeta = healpy.almxfl(dlm_in, np.divide(Cltaudd, Cldd, out=np.zeros_like(Cltaudd), where=Cldd!=0))
             
            
            if n_level!=0:
                ffactor1 = ClTd**(2*n_level)
                ffactor2 = (ClTT * Cldd)**(n_level)
                filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)
                dTlm_xi_f = healpy.almxfl(dTlm_xi,filterf)
                dlm_zeta_f = healpy.almxfl(dlm_zeta, filterf)
            else:
                dTlm_xi_f = dTlm_xi
                dlm_zeta_f = dlm_zeta
                
    
            xizeta_n[i] = healpy.alm2map(dTlm_xi_f, nside,verbose=False)*healpy.alm2map(dlm_zeta_f,nside,verbose=False)
            
            
            dTlm_xibar = healpy.almxfl(dTlm_beamed, np.divide(Cltaudd, ClTd, out=np.zeros_like(Cltaudd), where=ClTd!=0) )
            dlm_zetabar = healpy.almxfl(dlm_in, np.divide(ones, ClTd, out=np.zeros_like(ones), where=ClTd!=0) )
            
            
            ffactor1 = ClTd**(2*n_level+2)
            ffactor2 = (ClTT * Cldd)**(n_level+1)
            filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)

            dTlm_xibar_f = healpy.almxfl(dTlm_xibar, filterf)
            dlm_zetabar_f = healpy.almxfl(dlm_zetabar, filterf)
            
            
            xizetabar_n[i] = healpy.alm2map(dTlm_xibar_f, nside,verbose=False)*healpy.alm2map(dlm_zetabar_f,nside,verbose=False)
            
        return xizeta_n, xizetabar_n
    
    def bare_quad_est_ml(self,nside, n_level, nbin,
                         bin_width,
                         dTlm, dlm,
                         beam_window, lcut,
                         cllsslssbinned, ClTT,
                         clTlssbinned, clmllssbinned,bin_choice):

        ones =  np.ones(3*nside)
        cut = np.where(np.arange(3*nside)<lcut, 1, 1e-30)

        cllsslssbinned[0,:,:] = 1e15
        
        start = time.time()
        
        xizeta_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))
            
        xizetal_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xilzeta_n = np.zeros((nbin,healpy.nside2npix(nside)))

        xizetalbar_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xilzetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))

#         dTlm = healpy.map2alm(Tfield)
        dTlm_beamed = healpy.almxfl(dTlm,(1./beam_window)*cut)
        
        lvec = np.arange(0,3*nside)
        lfac = lvec*(1.+lvec)

        for i in range(nbin):
            
            print('we are in bare_quad bin=',i)
            if bin_choice!=i:
                print('passing on bin=',i)
                continue
            
            print('calculating bin=',i)
            Clmld = clmllssbinned[:,i,i]*bin_width
            Cldd = cllsslssbinned[:,i,i]
            ClTd = clTlssbinned[:,0,i]
    
            dlm_in = dlm[i]
            
            dTlm_xi = healpy.almxfl(dTlm_beamed,np.divide(ones, ClTT,
                                                          out=np.zeros_like(ones),
                                                          where=ClTT!=0))
            dlm_zeta = healpy.almxfl(dlm_in, np.divide(Clmld, Cldd,
                                                       out=np.zeros_like(Clmld),
                                                       where=Cldd!=0))
             
            dTlm_xi_l = healpy.almxfl(dTlm_beamed,np.divide(lfac, ClTT,
                                                            out=np.zeros_like(ones),
                                                            where=ClTT!=0))
            dlm_zeta_l = healpy.almxfl(dlm_in, np.divide(lfac*Clmld, Cldd,
                                                         out=np.zeros_like(Clmld),
                                                         where=Cldd!=0))

            if n_level!=0:
                ffactor1 = ClTd**(2*n_level)
                ffactor2 = (ClTT * Cldd)**(n_level)
                filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)
                
                dTlm_xi_f = healpy.almxfl(dTlm_xi,filterf)
                dlm_zeta_f = healpy.almxfl(dlm_zeta, filterf)
                
                dTlm_xi_f_l = healpy.almxfl(dTlm_xi_l,filterf)
                dlm_zeta_f_l = healpy.almxfl(dlm_zeta_l, filterf)

            else:
                dTlm_xi_f = dTlm_xi
                dlm_zeta_f = dlm_zeta
                    
                dTlm_xi_f_l = dTlm_xi_l
                dlm_zeta_f_l = dlm_zeta_l

            xi_f = healpy.alm2map(dTlm_xi_f, nside,verbose=False)
            zeta_f = healpy.alm2map(dlm_zeta_f,nside,verbose=False)
            xi_f_l = healpy.alm2map(dTlm_xi_f_l, nside,verbose=False)
            zeta_f_l = healpy.alm2map(dlm_zeta_f_l,nside,verbose=False)
            
            xizeta_n[i]  = xi_f*zeta_f
            xizetal_n[i] = xi_f*zeta_f_l
            xilzeta_n[i] = xi_f_l*zeta_f
            
            ## now the barred quantites

            dTlm_xibar = healpy.almxfl(dTlm_beamed,
                                       np.divide(Clmld, ClTd,
                                                 out=np.zeros_like(Clmld),
                                                 where=ClTd!=0) )
            dlm_zetabar = healpy.almxfl(dlm_in,
                                        np.divide(ones, ClTd,
                                                  out=np.zeros_like(ones),
                                                  where=ClTd!=0) )

            dTlm_xibar_l = healpy.almxfl(dTlm_beamed,
                                         np.divide(lfac*Clmld, ClTd,
                                                   out=np.zeros_like(Clmld),
                                                   where=ClTd!=0) )
            dlm_zetabar_l = healpy.almxfl(dlm_in,
                                          np.divide(lfac, ClTd,
                                                    out=np.zeros_like(ones),
                                                    where=ClTd!=0) )
            
            ffactor1 = ClTd**(2*n_level+2)
            ffactor2 = (ClTT * Cldd)**(n_level+1)
            filterf = np.divide(ffactor1, ffactor2,
                                out=np.zeros_like(ffactor1),
                                where=ffactor2!=0)

            dTlm_xibar_f = healpy.almxfl(dTlm_xibar, filterf)
            dlm_zetabar_f = healpy.almxfl(dlm_zetabar, filterf)
            
            dTlm_xibar_f_l = healpy.almxfl(dTlm_xibar_l, filterf)
            dlm_zetabar_f_l = healpy.almxfl(dlm_zetabar_l, filterf)

            xizetabar_n[i] = (healpy.alm2map(dTlm_xibar_f, nside,verbose=False)
                              *healpy.alm2map(dlm_zetabar_f,nside,verbose=False))
            
            xilzetabar_n[i] = (healpy.alm2map(dTlm_xibar_f_l, nside,verbose=False)
                               *healpy.alm2map(dlm_zetabar_f,  nside,verbose=False))
            xizetalbar_n[i] = (healpy.alm2map(dTlm_xibar_f,   nside,verbose=False)
                               *healpy.alm2map(dlm_zetabar_f_l,nside,verbose=False))
            
            end = time.time()
            
            print("bare_quad done in t="+str(end-start))
        
        return xizeta_n-xizetabar_n, (xizetal_n-xilzeta_n
                                      -xilzetabar_n+xizetalbar_n)
    
    def reconstruct(self, nside, nsideout, n_level, nbin , bin_width, Tfield, gmaps, vmaps, beam_window, cllsslss, clTT, clTlss, cltaudlss, Noise):
        

        xizeta = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar = np.zeros((nbin,healpy.nside2npix(nside)))
        binned_qe = np.zeros((nbin,healpy.nside2npix(nsideout)))
          
            
        for n in range(n_level+1):
            xizetat, xizetabart = self.bare_quad_est(nside, n, nbin, bin_width, Tfield, gmaps, beam_window, cllsslss, clTT, clTlss,cltaudlss)
            xizeta += xizetat
            xizetabar += xizetabart

        for binn in range(nbin):
            veff_reconstlm = healpy.almxfl(healpy.map2alm(xizeta[binn]-xizetabar[binn],lmax=(3*nsideout-1)),Noise[binn])
            binned_qe[binn] = healpy.alm2map(veff_reconstlm, nsideout)

        return binned_qe

    def reconstruct_ml(self,
                       nside,
                       nsideout,
                       n_level,
                       nbin ,
                       bin_width,
                       Tlm,
                       glm,
                       vlm,
                       beam_window,
                       lcut,
                       cllsslss,
                       clTT,
                       clTlss,
                       clmllss,
                       Noise,bin_choice):

        term1 = np.zeros((nbin,healpy.nside2npix(nside)))
        term2 = np.zeros((nbin,healpy.nside2npix(nside)))

        binned_qe = np.zeros((nbin,healpy.nside2npix(nsideout)))

        for n in range(n_level+1):
            if n==0:
                term1, term2 = self.bare_quad_est_ml(nside, n, nbin,
                                                     bin_width,
                                                     Tlm,
                                                     glm,
                                                     beam_window, lcut,
                                                     cllsslss, clTT,
                                                     clTlss,clmllss,bin_choice)
            else:
                term1t, term2t = self.bare_quad_est_ml(nside, n, nbin,
                                                       bin_width,
                                                       Tlm,
                                                       glm,
                                                       beam_window, lcut,
                                                       cllsslss, clTT,
                                                       clTlss,clmllss,bin_choice)
                term1 += term1t
                term2 += term2t

        lvec = np.arange(0,3*nsideout)
        lfac = lvec*(1.+lvec)

        factor = 1e30
        for binn in range(nbin):
            print('we are in reconstruct_ml bin=',binn)
            if bin_choice!=binn:
                print('passing on bin=',binn)
                continue
            
            print('we are in reconstruct_ml bin=',binn)
                
            term1[binn] = term1[binn]/factor
            veff_reconstlm1 = healpy.almxfl(healpy.map2alm(term1[binn], lmax=(3*nsideout-1)), lfac*Noise[binn])

            term2[binn] = term2[binn]/factor
            veff_reconstlm2 = healpy.almxfl(healpy.map2alm(term2[binn], lmax=(3*nsideout-1)), Noise[binn])

            binned_qe[binn] = ( healpy.alm2map((veff_reconstlm1)*0.5*factor, nsideout)
                               +healpy.alm2map((veff_reconstlm2)*0.5*factor, nsideout))

        return binned_qe

    def get_qe_sims(self,nside, nsideout,n_level, plcut, real_num,real_num2, use_cleaned = True, frequency = None, mask = False):
        
        print('plcut:',plcut)
        self.set_theory_Cls(add_ksz = True, add_ml = False, use_cleaned = use_cleaned, frequency = frequency)
        self.Ttag = 'T0'
        lcut = plcut
        
        if use_cleaned:
            beam_window = np.ones(3*nside)   # We have to determine how to work the beaming here
        else:
            ls = np.arange(3*nside)
            beam_window = np.exp(-ls*(ls+1)*(self.beam**2)/(16.*np.log(2)))
        
        clTT      = self.Cls['T-T'][:3*nside,0,0]
        clTlss    = self.Cls['T-lss'][:3*nside]
        cltaudlss = self.Cls['taud-lss'][:3*nside]
        cllsslss  = self.Cls['lss-lss'][:3*nside]
        clksz     = self.Cls['kSZ-kSZ'][:3*nside,0,0]
        
 
        #GET THEORY NOISE . ONE CAN SPEED UP THIS ASSUMING ELL DEPENDENCE
        
        print("Getting theory noise")
        
        if c.exists(self.basic_conf_dir,'Nvv_'
                    +str(nside)+'_'+str(nsideout)+'_'+str(plcut),
                    dir_base = 'sims2'):
            Noise = c.load(self.basic_conf_dir,'Nvv_'
                           +str(nside)+'_'+str(nsideout)+'_'+str(plcut),
                           dir_base = 'sims2')
        else:
            
            Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
            Noise_int = np.zeros((len(Lsamp),self.nbin))
             
            for lid, l  in enumerate(Lsamp):
                for i in np.arange(self.nbin):
                    #Noise_int[lid,i] = self.Noise_iso(3*nside, 'v', i, i, l)
                    print('lcut:',lcut)
                    Noise_int[lid,i] = self.Noise_iso(lcut, 'v', i, i, l)
                
            Noise = np.swapaxes(loginterp.log_interpolate_vector(Noise_int, Lsamp),0,1)
            
            c.dump(self.basic_conf_dir,Noise,'Nvv_'
                   +str(nside)+'_'+str(nsideout)+'_'+str(plcut), dir_base = 'sims2')
            
        print("Theory noise ready")
        
        print("Getting rotation matrix ")
        
        if c.exists(self.basic_conf_dir,
                    'Rvv_'+str(nside)+'_'+str(nsideout)+'_'+str(plcut),
                    dir_base = 'sims2'):
            R = c.load(self.basic_conf_dir,
                       'Rvv_'+str(nside)+'_'+str(nsideout)+'_'+str(plcut),
                       dir_base = 'sims2')
        else:
            
            R = np.zeros((self.nbin,self.nbin))
             
            for i in np.arange(self.nbin):
                for j in np.arange(self.nbin):
                    #R[i,j] = self.R(3*nside,'v','v',i,j,2)
                    R[i,j] = self.R(lcut,'v','v',i,j,2)
                    
            c.dump(self.basic_conf_dir,R,'Rvv_'
                   +str(nside)+'_'+str(nsideout)+'_'+str(plcut), dir_base = 'sims2')
            
        print("Rotation matrix ready")
        
#        real_num = self.realnum
                
        for r in np.array([real_num]):
            
            print('real = '+str(r))
            
            if c.exists(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)
                        +'_full'+'_real='+str(r)+'+'+str(real_num2)+'_mask='+str(mask)
                        +'_nlevel='+str(n_level)+'_'+str(plcut), dir_base = 'sims2'):
                print("r = "+str(r)+" already done")
                continue
                     
                
            if c.exists(self.basic_conf_dir,'v_'+str(nside)+'_real='
                        +str(r)+'+'+str(real_num2)+'_bin='+str(0)+'_'+str(plcut), dir_base = 'sims2'):
                
                print("Loading pre-existing sims")
                
                vmaps    = np.zeros((self.nbin,healpy.nside2npix(nside)))
                taudmaps = np.zeros((self.nbin,healpy.nside2npix(nside)))
                lssmaps  = np.zeros((self.nbin,healpy.nside2npix(nside)))
                Tmap     = np.zeros(healpy.nside2npix(nside))
                
                for b in np.arange(self.nbin):
                    
                    vmaps[b,:] = c.load(self.basic_conf_dir,'v_'
                                        +str(nside)+'_real='
                                        +str(r)+'+'+str(real_num2)+'_bin='+str(b)+'_'+str(plcut),
                                        dir_base = 'sims2')
                    taudmaps[b,:] = c.load(self.basic_conf_dir,
                                           'taud_'+str(nside)+'_real='
                                           +str(r)+'+'+str(real_num2)+'_bin='+str(b)+'_'+str(plcut),
                                           dir_base = 'sims2')
                    lssmaps[b,:] = c.load(self.basic_conf_dir,
                                          'lss_ksz_'+'+'+str(real_num2)+str(nside)+'_real='
                                          +str(r)+'_bin='+str(b)+'_'+str(plcut),
                                          dir_base = 'sims2')
   
                Tmap = c.load(self.basic_conf_dir,
                              self.Ttag+'_'+str(nside)
                              +'_real='
                              +str(r)+'+'+str(real_num2)+'_bin='+str(0)+'_'+str(plcut),
                              dir_base = 'sims2')
                
            else:
                
                print("Getting sims")
                
                if self.Ttag != 'T0':
                
                    sims, alms = self.get_maps_and_alms(['v','taud','g',
                                                         self.Ttag],
                                                        nside,3*nside)
                    vmaps    = sims[0]
                    taudmaps = sims[1]
                    lssmaps  = sims[2]
                    Tmap    = sims[3][0,:]
                    
                    for b in np.arange(self.nbin):
                        
                        c.dump(self.basic_conf_dir,vmaps[b,:],
                               'v_'+str(nside)
                               +'_real='+str(r)+'+'+str(real_num2)+'_bin='+str(b)+'_'+str(plcut),
                               dir_base = 'sims2')
                        c.dump(self.basic_conf_dir,taudmaps[b,:],
                               'taud_'
                               +str(nside)+'_real='+
                               str(r)+'+'+str(real_num2)+'_bin='+str(b)+'_'+str(plcut),
                               dir_base = 'sims2')
                        c.dump(self.basic_conf_dir,lssmaps[b,:],
                               'lss_ksz_'
                               +str(nside)+'_real='+str(r)+'+'+str(real_num2)
                               +'_bin='+str(b)+'_'+str(plcut), dir_base = 'sims2')
       
                    c.dump(self.basic_conf_dir,Tmap,
                           self.Ttag+'_'
                           +str(nside)+'_real='+str(r)+'+'+str(real_num2)
                           +'_bin='+str(0)+'_'+str(plcut), dir_base = 'sims2')

                else:
                
                    sims, alms = self.get_maps_and_alms(['v','taud','g'],nside,3*nside)
                    vmaps    = sims[0]
                    taudmaps = sims[1]
                    lssmaps  = sims[2]
                    #Tmap    =  healpy.synfast(self.Cls['pCMB-pCMB'][:3*nside,0,0],nside)
                    Tmap    =  healpy.synfast(self.Cls['T-T'][:3*nside,0,0],nside)
                    
                    for b in np.arange(self.nbin):
                        
                        c.dump(self.basic_conf_dir,vmaps[b,:],'v_'
                               +str(nside)+'_real='+str(r)+'_bin='+str(b)+'_'+str(plcut), dir_base = 'sims2')
                        c.dump(self.basic_conf_dir,taudmaps[b,:],'taud_'
                               +str(nside)+'_real='+str(r)+'_bin='+str(b)+'_'+str(plcut), dir_base = 'sims2')
                        c.dump(self.basic_conf_dir,lssmaps[b,:],'lss_ksz_'
                               +str(nside)+'_real='+str(r)+'_bin='+str(b)+'_'+str(plcut), dir_base = 'sims2')
       
                    c.dump(self.basic_conf_dir,Tmap,self.Ttag+'_'
                           +str(nside)+'_real='+str(r)+'_bin='+str(0)+'_'+str(plcut), dir_base = 'sims2')
                
                
                
                  
            kszmap = self.ksz_map_maker(taudmaps, vmaps, nside)
          
            if not c.exists(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='
                            +str(r)+'_'+str(plcut), dir_base = 'sims2'):
            
                vrot = np.zeros((self.nbin,healpy.nside2npix(nsideout)))
                
                for b in range(self.nbin):
                    for i in np.arange(self.nbin):
                        vrot[b,:] += R[b,i]*healpy.pixelfunc.ud_grade(vmaps[i,:], nside_out = nsideout)
                        #update R so it also depends on ell
                c.dump(self.basic_conf_dir,vrot,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='
                       +str(r)+'_'+str(plcut), dir_base = 'sims2')

            

            #if use_cleaned:
            Tfield_gauss = healpy.synfast(clksz,nside)+Tmap
            Tfield_full =  kszmap + Tmap
            #else:
            #    Tfield_gauss = (healpy.smoothing(healpy.synfast(clksz,nside)
            #                                     + Tmap*0.,fwhm=self.beam)
            #                    + 0.*healpy.synfast(self.dT*self.dT
            #                                        *np.ones(3*nside),nside))
            #    Tfield_full =  (healpy.smoothing(kszmap
            #                                     + 0.*Tmap,fwhm=self.beam)
            #                    + 0.*healpy.synfast(self.dT*self.dT
            #                                        *np.ones(3*nside),nside))

            
            clTT[0:2]=0
            
            if mask:
                Tfield_gauss = Tfield_gauss*np.load('SO_mask_N2048.npy')
                Tfield_full  = Tfield_full*np.load('SO_mask_N2048.npy')
                                    
            print("Reconstructing velocity")
                
            qe_gauss = self.reconstruct(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_gauss,
                    lssmaps, vmaps, beam_window, cllsslss, clTT, clTlss, cltaudlss,Noise)
            qe_full = self.reconstruct(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_full,
                    lssmaps, vmaps, beam_window, cllsslss, clTT, clTlss, cltaudlss,Noise)
    
            c.dump(self.basic_conf_dir,qe_gauss,'qe_'+str(nside)+'_'
                   +str(nsideout)+'_gauss'+'_real='
                   +str(r)+'_mask='+str(mask)+'_nlevel='
                   +str(n_level)+'_'+str(plcut), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,qe_full,'qe_'+str(nside)+'_'
                   +str(nsideout)+'_full'+'_real='
                   +str(r)+'_mask='+str(mask)+'_nlevel='
                   +str(n_level)+'_'+str(plcut), dir_base = 'sims2')
        
        pass
    
    def get_qe_sims_ml(self,nside, nsideout,
                       n_level, lcut, real_num, real_num2, tempfac, use_cleaned = True, mask = False, bin_choice=0):
        
        plcut = lcut
        alt_dir = self.basic_conf_dir # '/rds/general/use:wr/sch14/ephemeral/'
        lcuts = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1350,1500,1750,2000,2250,2500,2750,3000]
        lcuts_real = lcuts[real_num2]
        print(plcut,lcuts_real,'bin_coice=',bin_choice)
    
        self.set_theory_Cls(add_ksz = False, add_ml = False,
                            use_cleaned = use_cleaned)
        self.Ttag = 'T0'
        beam_window = np.ones(3*nside)
       
        print("entering self.set_theory_Cls")
        self.Cls['T-T'][:3*nside,0,0]=self.Cls['T-T'][:3*nside,0,0]*tempfac**2.
        self.Cls['T-lss'][:3*nside]=self.Cls['T-lss'][:3*nside]*tempfac
        
        if lcuts_real>0: self.Cls['T-T'][(3*nside-lcuts_real):3*nside,0,0]=1e-30
        if lcuts_real>0: self.Cls['T-lss'][(3*nside-lcuts_real):3*nside,0,0]=1e-30
        
        clTT      = self.Cls['T-T'][:3*nside,0,0]
        clTlss    = self.Cls['T-lss'][:3*nside]
        clmllss   = self.Cls['ml-lss'][:3*nside]
        cllsslss  = self.Cls['lss-lss'][:3*nside]
        clML      = self.Cls['ML-ML'][:3*nside,0,0]
        
        if tempfac<1.: self.Cls['T-T'][:3*nside,0,0]+=self.Cls['ML-ML'][:3*nside,0,0]
        if lcuts_real>0: self.Cls['T-T'][(3*nside-lcuts_real):3*nside,0,0]=1e-30
        if lcuts_real>0: self.Cls['T-lss'][(3*nside-lcuts_real):3*nside,0,0]=1e-30

        
        #GET THEORY NOISE . ONE CAN SPEED UP THIS ASSUMING ELL DEPENDENCE
        
        print("Getting theory noise")
        
       # snote: this was wrong, no? geomspace was not aqequate
        Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
        Noise_int = np.zeros((len(Lsamp),self.nbin))
        print("calculating Nvtvts")
        for lid, l  in enumerate(Lsamp):
            for i in np.arange(self.nbin):
                ntfile = 'Nvtvt_'+str(nside)+'_L='+str(lid)+'_bin='+str(i)+'_'+str(nsideout)+'_'+str(plcut)+'_'+str(tempfac)
                if c.exists(alt_dir,
                            ntfile,
                            dir_base = 'sims3'):
                    try:
                        Noise_int[lid,i] = c.load(alt_dir,ntfile,dir_base = 'sims3')
                    except EOFError:
                        print('ran out of input error given')
                        Noise_int[lid,i] = self.Noise_iso(lcut, 'vt', i, i, l)
                        print('N saving lid=',l,' l=',l,' i=',i)
                        c.dump(alt_dir,Noise_int[lid,i],ntfile,dir_base = 'sims3')
                else:
                    Noise_int[lid,i] = self.Noise_iso(lcut, 'vt', i, i, l)
                    print('N saving lid=',l,' l=',l,' i=',i)
                    c.dump(alt_dir,Noise_int[lid,i],ntfile,dir_base = 'sims3')
                    
                c.dump(alt_dir,Noise_int[lid,i],ntfile,dir_base = 'sims3')

        Noise = np.swapaxes(loginterp.log_interpolate_vector(Noise_int, Lsamp),0,1)
        print("Theory noise ready")
        
        print("Getting rotation matrix")
        R = np.zeros((self.nbin,self.nbin))
             
        for i in np.arange(self.nbin):
            for j in np.arange(self.nbin):
                rotfile = ('Rvtvt_'+str(nside)+'_L='+str(l)+'_bin='+str(i)+'_'
                           +str(j)+'_'+str(nsideout)+'_'+str(plcut)+'_'+str(tempfac))
                if c.exists(alt_dir,rotfile,dir_base = 'sims3'):
                    try: R[i,j] = c.load(alt_dir,rotfile, dir_base = 'sims3')
                    except EOFError:
                        print('ran out of input error given')
                        R[i,j] = self.R(lcut,'vt','vt',i,j,2)
                        print('R saving bin=',i,j)
                        c.dump(alt_dir,R[i,j],rotfile,dir_base = 'sims3')
                else:
                    R[i,j] = self.R(lcut,'vt','vt',i,j,2)
                    print('R saving bin=',i,j)
                    c.dump(alt_dir,R[i,j],rotfile,dir_base = 'sims3')
        print("Rotation matrix ready")
        
        real_num = self.realization
        for r in [real_num]:
            print('real = '+str(r))
            
#             if c.exists(alt_dir,
#                        'qe_vt_'+str(nside)+'_'
#                        +str(nsideout)+'_gauss'
#                        +'_real='+str(r)+'_mask='+str(mask)
#                        +'_nlevel='+str(n_level)
#                        +'_'+str(self.llcut)+'_tempfac='+str(tempfac)
#                        +'_bin='+str(bin_choice), dir_base = 'sims3') and self.type==1:
#                 print("r = "+str(r)+" gaussian sims already done")
#                 continue
#             if c.exists(alt_dir,
#                        'qe_vt_'+str(nside)+'_'
#                        +str(nsideout)+'_full'
#                        +'_real='+str(r)+'_mask='+str(mask)
#                        +'_nlevel='+str(n_level)
#                        +'_'+str(self.llcut)+'_tempfac='+str(tempfac)
#                        +'_bin='+str(bin_choice), dir_base = 'sims3') and self.type==0:
#                 print("r = "+str(r)+" full sims already done")
#                 continue
                
            if c.exists(alt_dir,
                        'vt_'+str(nside)+'_real='
                        +str(r)+'_bin='+str(0), dir_base = 'sims3'):
                
                print("Loading pre-existing sims")
                
                start = time.time()

                vtmaps  = np.zeros((self.nbin,healpy.nside2npix(nside)))
                mlmaps  = np.zeros((self.nbin,healpy.nside2npix(nside)))
                lssmaps = np.zeros((self.nbin,healpy.nside2npix(nside)))
                Tmap    = np.zeros(healpy.nside2npix(nside))
                
                for b in np.arange(self.nbin):

                    vtmaps[b,:] = c.load(alt_dir,
                                         'vt_'+str(nside)+'_real='
                                        +str(r)+'_bin='+str(b),
                                         dir_base = 'sims3')
                    mlmaps[b,:] = c.load(alt_dir,
                                         'ml_'+str(nside)+'_real='
                                         +str(r)+'_bin='+str(b),
                                         dir_base = 'sims3')
                    lssmaps[b,:] = c.load(alt_dir,
                                          'lss_ml_'+str(nside)+'_real='
                                          +str(r)+'_bin='+str(b),
                                          dir_base = 'sims3')
   
                Tmap = c.load(alt_dir,
                              self.Ttag+'_'
                              +str(nside)+'_real='
                              +str(r)+'_bin='+str(0),
                              dir_base = 'sims3')
                MLmap = c.load(alt_dir,
                               'MLmap_'+str(nside)+'_real='
                               +str(r)+'_bin='+str(0),
                               dir_base = 'sims3')
                end = time.time()
                print('time elapsed loading: ', end - start)

            else:
                
                print("Getting sims")
                
                if self.Ttag != 'T0':
                
                    sims, alms = self.get_maps_and_alms(['vt','ml','g',self.Ttag],
                                                         100,3*nside)
                    vtmaps  = sims[0]
                    mlmaps  = sims[1]
                    lssmaps = sims[2]
                    Tmap    = sims[3][0,:]
                    
                    MLmap  = self.ml_map_maker(alms[0], alms[1],  nside)

                    for b in np.arange(self.nbin):

                        c.dump(alt_dir,
                               vtmaps[b,:],
                               'vt_'+str(nside)
                               +'_real='+str(r)+'_bin='
                               +str(b),
                               dir_base = 'sims3')
                        c.dump(alt_dir,
                               mlmaps[b,:],
                               'ml_'+str(nside)+'_real='
                               +str(r)+'_bin='
                               +str(b),
                               dir_base = 'sims3')
                        c.dump(alt_dir,
                               lssmaps[b,:],
                               'lss_ml_'+str(nside)+'_real='
                               +str(r)+'_bin='
                               +str(b),
                               dir_base = 'sims3')
       
                    c.dump(alt_dir,
                           Tmap,self.Ttag+'_'
                           +str(nside)+'_real='
                           +str(r)+'_bin='+str(0),
                           dir_base = 'sims3')
                    c.dump(alt_dir,
                           MLmap,'MLmap_'
                           +str(nside)+'_real='
                           +str(r)+'_bin='+str(0),
                           dir_base = 'sims3')

                else:
                    print('get_maps_and_alms([ml,vt,g]')
                    sims, alms = self.get_maps_and_alms(['ml','vt','g'],
                                                        nside,3*nside)
                    ml_alm = alms[0]
                    vt_alm = alms[1]
                    lss_alm = alms[2]

                    print('making ML map')
                    if c.exists(self.basic_conf_dir,
                                'MLmap_real='+str(self.realization)
                                +'_bin=all',dir_base = 'sims3'):
                        print('ML map exists')
                        MLmap = c.load(self.basic_conf_dir,
                                       'MLmap_real='+str(self.realization)
                                       +'_bin=all',dir_base = 'sims3')
                    else:
                        print('making ML map from sims')
                        MLmap  = self.ml_map_maker(alms[0], alms[1],  nside)
                    print('finished making ML map')
                    for b in np.arange(self.nbin):
                        continue
                        print('saving alms for bin=',b)
                        c.dump(alt_dir,
                               alms[0][b,:],'ml_alm_'
                               +str(nside)
                               +'_real='+str(r)
                               +'_bin='+str(b),
                               dir_base = 'sims3')
                        c.dump(alt_dir,
                               alms[1][b,:],'vt_alm_'
                               +str(nside)+'_real='
                               +str(r)+'_bin='+str(b),
                               dir_base = 'sims3')
                        c.dump(alt_dir,
                               alms[2][b,:],'lss_ml_alm_'
                               +str(nside)+'_real='
                               +str(r)+'_bin='+str(b),
                               dir_base = 'sims3')
            ####
            if c.exists(alt_dir,
                        'MLmap_gauss'+str(nside)+'_'+str(nsideout)
                        +'_real='+str(r), dir_base = 'sims3'):
                MLmap_gauss = c.load(alt_dir,'MLmap_gauss'+str(nside)+'_'+str(nsideout)
                                     +'_real='+str(r), dir_base = 'sims3')
                print('MLmap_gauss exists')
            else:
                print('simulating MLmap_gauss')
                MLmap_gauss = healpy.synfast(clML,nside,lmax=3*nside)
                c.dump(alt_dir,MLmap_gauss,
                       'MLmap_gauss_'+str(nside)+'_'+str(nsideout)
                       +'_real='+str(r), dir_base = 'sims3')
            ####
            if c.exists(alt_dir,
                        'Tmap_0_'+str(nside)+'_'+str(nsideout)
                        +'_real='+str(r), dir_base = 'sims3'):
                Tmap_0 = c.load(alt_dir,'Tmap_0_'+str(nside)+'_'+str(nsideout)
                                +'_real='+str(r), dir_base = 'sims3')
                print('Tmap_0 exists')
            else:
                print('simulating Tmap_0')
                Tmap_0 = healpy.synfast(self.Cls['T-T'][:3*nside,0,0],nside,lmax=3*nside)
                c.dump(alt_dir,Tmap_0,'Tmap_0_'+str(nside)+'_'+str(nsideout)
                       +'_real='+str(r), dir_base = 'sims3')
            ####
            if c.exists(alt_dir,
                        'T_alm_'+str(nside)+'_'+str(nsideout)
                        +'_full'+'_real='+str(r)+'_mask='+str(mask)+'_tempfac='
                        +str(tempfac), dir_base = 'sims3'):
                
                Talms_full = c.load(alt_dir,'T_alm_'+str(nside)+'_'+str(nsideout)
                                    +'_full'+'_real='+str(r)+'_mask='+str(mask)+'_tempfac='
                                    +str(tempfac), dir_base = 'sims3')
                Talms_gauss = c.load(alt_dir,'T_alm_'+str(nside)+'_'+str(nsideout)
                                    +'_gauss'+'_real='+str(r)+'_mask='+str(mask)+'_tempfac='
                                    +str(tempfac), dir_base = 'sims3')
                print("T_alm already exists")
            else:
                print("producing Talms")
                
                Talms_gauss = healpy.sphtfunc.map2alm(MLmap_gauss+tempfac*Tmap_0,lmax=3*nside)
                Talms_full =  healpy.sphtfunc.map2alm(MLmap+tempfac*Tmap_0,lmax=3*nside)
                
                c.dump(alt_dir,
                       Talms_gauss,'T_alm_'+str(nside)+'_'+str(nsideout)
                        +'_gauss'+'_real='+str(r)+'_mask='+str(mask)
                       +'_tempfac='+str(tempfac), dir_base = 'sims3')
                c.dump(alt_dir,
                       Talms_full,'T_alm_'+str(nside)+'_'+str(nsideout)
                        +'_full'+'_real='+str(r)+'_mask='+str(mask)
                       +'_tempfac='+str(tempfac), dir_base = 'sims3')
                
                
            clTT[0:2]=0
            
#             if mask:
#                 Tfield_gauss = Tfield_gauss*np.load('SO_mask_N2048.npy')
#                 Tfield_full  = Tfield_full*np.load('SO_mask_N2048.npy')
             if mask:
                Talms_gauss = healpy.sphtfunc.map2alm(healpy.sphtfunc.alm2map(Talms_gauss,nside,lmax=3*nside)
                        *np.load('/rds/general/user/sch14/home/class_new/python/SO_mask_N2048.npy'),
                        lmax=3*nside)
                Talms_full  = healpy.sphtfunc.map2alm(healpy.sphtfunc.alm2map(Talms_full,nside,lmax=3*nside)
                        *np.load('/rds/general/user/sch14/home/class_new/python/SO_mask_N2048.npy'),
                        lmax=3*nside)
                
            print('lcut=',self.llcut)
            print("Reconstructing transverse velocity")
            start = time.time()
            if self.type == 1:
                qe_gauss = self.reconstruct_ml(nside, nsideout,
                                           n_level, self.nbin,
                                           self.deltachi,
                                           Talms_gauss,
                                           lss_alm,vt_alm,
                                           beam_window,self.llcut,
                                           cllsslss, clTT,
                                           clTlss, clmllss,
                                           Noise,bin_choice)
                end = time.time()
                print('time elapsed reconstructing gauss: ', end - start)
                c.dump(alt_dir,qe_gauss[bin_choice,:],
                       'qe_vt_'+str(nside)+'_'
                       +str(nsideout)+'_gauss'
                       +'_real='+str(r)+'_mask='+str(mask)
                       +'_nlevel='+str(n_level)
                       +'_'+str(self.llcut)+'_tempfac='+str(tempfac)
                       +'_bin='+str(bin_choice), dir_base = 'sims3')

            elif self.type == 0:
                start = time.time()
                qe_full  = self.reconstruct_ml(nside, nsideout,
                                           n_level, self.nbin,
                                           self.deltachi,
                                           Talms_full,
                                           lss_alm, vt_alm,
                                           beam_window,self.llcut,
                                           cllsslss, clTT,
                                           clTlss, clmllss,
                                           Noise,bin_choice)
                end = time.time()
                print('time elapsed reconstructing full: ', end - start)

                c.dump(alt_dir,qe_full[bin_choice,:],
                        'qe_vt_'+str(nside)
                       +'_'+str(nsideout)+'_full'
                       +'_real='+str(r)+'_mask='
                       +str(mask)+'_nlevel='
                       +str(n_level)+'_'+str(self.llcut)
                       +'_tempfac='+str(tempfac)
                       +'_bin='+str(bin_choice), dir_base = 'sims3')
        
        pass

    def get_clqe_sims(self,nside, nsideout,n_level, lcut, real_num, real_num2, mask = True):
        plcut = lcut
        #real_num = self.realnum
        lcuts = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1350,1500,1750,2000,2250,2500,2750,3000]
        lcuts_real = lcuts[real_num2]
        for r in [real_num]:
            
            qe_full    = c.load(self.basic_conf_dir,
                                'qe_'+str(nside)+'_'+str(nsideout)
                                +'_full'+'_real='+str(r)
                                +'+'+str(real_num2)+'_mask='
                                +str(mask)+'_nlevel='
                                +str(n_level)+'_'+str(plcut),
                                dir_base = 'sims3')
            qe_gauss   = c.load(self.basic_conf_dir,
                                'qe_'+str(nside)+'_'+str(nsideout)
                                +'_gauss'+'_real='+str(r)
                                +'+'+str(real_num2)+'_mask='
                                +str(mask)+'_nlevel='
                                +str(n_level)+'_'+str(plcut),
                                dir_base = 'sims3')
            vrot = c.load(self.basic_conf_dir,'vactualrot_'
                          +str(nside)
                          +'_'+str(nsideout)
                          +'_real='+str(r)+'+'+str(real_num2)
                          +'_'+str(plcut), dir_base = 'sims3')
            
            print(r)
            
            Cvv_recon       = np.zeros((3*nsideout,self.nbin))
            Cvv_actual_rot  = np.zeros((3*nsideout,self.nbin))
            Cvv_diff        = np.zeros((3*nsideout,self.nbin))
            Cvv_noise       = np.zeros((3*nsideout,self.nbin))

            Cvv_rec_act     = np.zeros((3*nsideout,self.nbin))
            Cvv_rec_noi     = np.zeros((3*nsideout,self.nbin))
            Cvv_act_noi     = np.zeros((3*nsideout,self.nbin))
                        
            for b in np.arange(self.nbin):
                
                Cvv_rec_act = healpy.sphtfunc.anafast(qe_full[b],vrot[b])
                Cvv_rec_noi = healpy.sphtfunc.anafast(qe_full[b],qe_gauss[b])
                Cvv_act_noi = healpy.sphtfunc.anafast(vrot[b],qe_gauss[b])
                
                Cvv_recon[:,b]       = healpy.sphtfunc.anafast(qe_full[b],qe_full[b])
                Cvv_actual_rot[:,b]  = healpy.sphtfunc.anafast(vrot[b],vrot[b])
                Cvv_diff[:,b]        = healpy.sphtfunc.anafast(qe_full[b]-vrot[b],
                                                               qe_full[b]-vrot[b])
                Cvv_noise[:,b]       = healpy.sphtfunc.anafast(qe_gauss[b],
                                                               qe_gauss[b])

            c.dump(self.basic_conf_dir,Cvv_rec_act,
                    'Cvv_rec_act_'+str(nside)+'_'+
                   str(nsideout)+'_real='+str(r)+
                   '+'+str(real_num2)+'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')
            c.dump(self.basic_conf_dir,Cvv_rec_noi,
                    'Cvv_rec_noi_'+str(nside)+'_'+
                   str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)+'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')
            c.dump(self.basic_conf_dir,Cvv_act_noi,
                    'Cvv_act_noi_'+str(nside)+'_'+
                   str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)+'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')

            c.dump(self.basic_conf_dir,Cvv_recon,
                    'Cvv_recon_'+str(nside)+'_'+
                   str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)+'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')
            c.dump(self.basic_conf_dir,Cvv_actual_rot,
                    'Cvv_actual_rot_'+str(nside)+'_'
                   +str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)+'_'
                   +str(plcut), dir_base = 'sims3')
            c.dump(self.basic_conf_dir,Cvv_diff,
                   'Cvv_diff_'+str(nside)
                   +'_'+str(nsideout)+'_real='
                   +str(r)+'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')
            c.dump(self.basic_conf_dir,Cvv_noise,
                   'Cvv_noise_'+str(nside)+'_'
                   +str(nsideout)+'_real='
                   +str(r)+'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='+str(n_level)+'_'
                   +str(plcut), dir_base = 'sims3')
            
        pass
    
    def get_clqe_sims_ml(self,nside, nsideout, n_level,plcut,real_num,real_num2,tempfac, bin_choice,
                         use_cleaned = True, mask = True):
        
#        real_num = self.realnum
        lcuts = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1350,1500,1750,2000,2250,2500,2750,3000]
        lcuts_real = lcuts[real_num2]

        alt_dir = self.basic_conf_dir #'/rds/general/user/sch14/ephemeral/'
        
        for r in [real_num]:
            
            qe_full    = c.load(alt_dir,
                                'qe_vt_'+str(nside)+'_'+
                                str(nsideout)+'_full'+'_real='
                                +str(r)+'+'+str(real_num2)
                                +'_mask='+str(mask)
                                +'_nlevel='+str(n_level)
                                +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                                dir_base = 'sims3')
            qe_gauss   = c.load(alt_dir,
                                'qe_vt_'+str(nside)+'_'+
                                str(nsideout)+'_gauss'+'_real='
                                +str(r)+'+'+str(real_num2)
                                +'_mask='+str(mask)
                                +'_nlevel='+str(n_level)
                                +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                                dir_base = 'sims3')
            vtrot = c.load(alt_dir,
                           'vtactualrot_'
                           +str(nside)+'_'+str(nsideout)
                           +'_real='+str(r)+'+'+str(real_num2)
                           +'_'+str(plcut), dir_base = 'sims3')
            
            print(r)
            
            Cvtvt_recon       = np.zeros((3*nsideout,self.nbin))
            Cvtvt_actual_rot  = np.zeros((3*nsideout,self.nbin))
            Cvtvt_diff        = np.zeros((3*nsideout,self.nbin))
            Cvtvt_noise       = np.zeros((3*nsideout,self.nbin))
            
            Cvtvt_rec_act     = np.zeros((3*nsideout,self.nbin))
            Cvtvt_rec_noi     = np.zeros((3*nsideout,self.nbin))
            Cvtvt_act_noi     = np.zeros((3*nsideout,self.nbin))

            for b in np.arange(self.nbin):
                if b!=bin_choice: continue

                Cvtvt_recon[:,b]       = healpy.sphtfunc.anafast(qe_full,
                                                                 qe_full)
                Cvtvt_actual_rot[:,b]  = healpy.sphtfunc.anafast(vtrot[b],
                                                                 vtrot[b])
                Cvtvt_diff[:,b]        = healpy.sphtfunc.anafast(qe_full
                                                                 -vtrot[b],
                                                                 qe_full
                                                                 -vtrot[b])
                Cvtvt_noise[:,b]       = healpy.sphtfunc.anafast(qe_gauss,
                                                                 qe_gauss)
                
                Cvtvt_rec_act[:,b] = healpy.sphtfunc.anafast(qe_full,
                                                             vtrot[b])
                Cvtvt_rec_noi[:,b] = healpy.sphtfunc.anafast(qe_full,
                                                             qe_gauss)
                Cvtvt_act_noi[:,b] = healpy.sphtfunc.anafast(vtrot[b],
                                                             qe_gauss)

            c.dump(alt_dir,
                    Cvtvt_rec_act[:,bin_choice],
                   'Cvtvt_rec_act_'+str(nside)+'_'
                   +str(nsideout)+'_real='
                   +str(r)+'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='+str(n_level)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            c.dump(alt_dir,
                   Cvtvt_rec_noi[:,bin_choice],
                   'Cvtvt_rec_noi_'+str(nside)+'_'
                   +str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='+str(n_level)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            c.dump(alt_dir,
                   Cvtvt_act_noi[:,bin_choice],
                   'Cvtvt_act_noi_'+str(nside)+'_'
                   +str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='+str(n_level)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')

            c.dump(alt_dir,
                   Cvtvt_recon[:,bin_choice],
                   'Cvtvt_recon_'+str(nside)
                   +'_'+str(nsideout)+'_real='
                   +str(r)+'+'+str(real_num2)+'_mask='
                   +str(mask)+'_nlevel='
                   +str(n_level)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            c.dump(alt_dir,
                   Cvtvt_actual_rot[:,bin_choice],
                   'Cvtvt_actual_rot_'
                   +str(nside)
                   +'_'+str(nsideout)
                   +'_real='
                   +str(r)+'+'
                   +str(real_num2)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            c.dump(alt_dir,
                   Cvtvt_diff[:,bin_choice],
                   'Cvtvt_diff_'+str(nside)+'_'
                   +str(nsideout)+'_real='+str(r)
                   +'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='
                   +str(n_level)+'_'
                   +str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            c.dump(alt_dir,
                   Cvtvt_noise[:,bin_choice],
                   'Cvtvt_noise_'
                   +str(nside)+'_'
                   +str(nsideout)
                   +'_real='+str(r)
                   +'+'+str(real_num2)
                   +'_mask='+str(mask)
                   +'_nlevel='
                   +str(n_level)
                   +'_'+str(plcut)+'_tempfac='
                                +str(tempfac)+'_bin='+str(bin_choice),
                   dir_base = 'sims3')
            
        pass
    
    def ksz_map_maker(self, taud_map, v_map, nside):
        
        npix = healpy.nside2npix(nside)

        vmaps =  v_map
        taudmaps = taud_map
        kszmap = np.zeros(npix)

        for i in range(self.nbin):
            kszmap[:] += vmaps[i,:]*taudmaps[i,:]*self.deltachi

        return kszmap
    
    def ml_map_maker(self, alm_vt, alm_ml, nside):

        npix = healpy.nside2npix(nside)
        mlmap = np.zeros(npix)

#         for i in range(self.nbin):
        for i in self.ml_map_bin_range:
            print('ml map maker bin=',i)
            if c.exists(self.basic_conf_dir,
                        'MLmap_real='+str(self.realization)
                        +'_bin='+str(i),dir_base = 'sims3'):

                    mlmap[:] += c.load(self.basic_conf_dir,'MLmap_real='+str(self.realization)
                        +'_bin='+str(i),dir_base = 'sims3')
                    print('...MLmap already exists bin=',i)
                    continue
            
            map_ml_i, ml_d_theta, ml_d_phi = healpy.sphtfunc.alm2map_der1(alm_ml[i],
                                                                          nside)
            map_vt_i, vt_d_theta, vt_d_phi = healpy.sphtfunc.alm2map_der1(alm_vt[i],
                                                                          nside)

            mlmap_ = (ml_d_theta * vt_d_theta + ml_d_phi *
                      vt_d_phi)*self.deltachi
            
            c.dump(self.basic_conf_dir,mlmap_,
                           'MLmap_real='+str(self.realization)
                           +'_bin='+str(i),dir_base = 'sims3')
            
            mlmap[:] +=  mlmap_
            
        c.dump(self.basic_conf_dir,mlmap,'MLmap_real='+str(self.realization)
               +'_bin=all',dir_base = 'sims3')
            
        return mlmap
