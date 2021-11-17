#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:24:21 2019

@author: fionamccarthy
"""

from __future__ import print_function
from __future__ import absolute_import
import numpy as np

class CIB(object):

    def __init__(self, cosmo = None) :
        if cosmo == None :
            raise Exception ("You have to provide a cosmology class. Use cosmology.py to generate a cosmology class") 
 
        self.csm = cosmo
        
        self.experiment="Websky" 

        self.planck=6.626e-34
        self.lightspeed=2.99792458e8
        self.kBoltzmann=1.38e-23
             
        
        
        self.T0=20.7
        self.alpha=0.2
        self.beta=1.6
        self.delta=2.4
        self.sigmasqlm=0.3
        self.Meff=10**12.3
        self.Mmin=10**10.1#1470588235294.1174#10**10.1
        #Mmin=1470588235294.1174#10**10.1
        
        self.t4 = True
        
        if not self.t4:
            self.L0=6.725699999999999e-08*1.35e-5#5.2313668860490*1.35e-5*1e-5/10*6
        else:
            self.L0= 0.004755788078226362*1000*1.35e-5#*2#6.015648958508135*1.35e-5#1.35e-5*0.004755788078226362#5.2313668860490*1.35e-5
        #1131394909046.434*1.35e-5*0.9
        self.zplateau=2


    def TD(self,z): #Dust temperature of starforming galaxyT
                return self.T0*(1+z)**self.alpha
    
    
    
    def Planck(self,nu,T): #Planck law for black body radiation
        return 2*(nu)**3*(1/(np.exp(self.planck*nu/(self.kBoltzmann*T))-1))*self.planck/self.lightspeed**2
    
    def SED(self,nu,T,z):
        #eq 8 of 1611.04517
          
                nu=nu*1e9 #multiply by 10^^9 to go from GHz->Hz
                SED=np.zeros(nu.shape)
              #  nu02s=353*1e9*(1+z)
                if self.t4:
                    SED=(nu/T)**self.beta*self.Planck(nu,T)*1/T**4
                else:
                    SED=(nu**self.beta)*self.Planck(nu,T)
                return SED
            
            
    def redshiftevolutionofl(self,z):
                
                answer=np.zeros(len(z))
                
                answer=(1+z)**self.delta
                if len(answer[z>=self.zplateau])>0:
                    answer[z>=self.zplateau]=(1+self.zplateau)**self.delta
                return answer
    def Sigma(self,M):
        
              
                answer= M*(1/(2*np.pi*self.sigmasqlm)**(1/2))*np.exp(-(np.log10(M)-np.log10(self.Meff))**2/(2*self.sigmasqlm))
                answer[M<self.Mmin]=0
                return answer
    def Lnu(self,nu,z,M): #spectral luminosity radiance
                Lir=self.L0#*1.35e-5 #total IR luminosity
                sed=self.SED(nu*(1+z),self.TD(z),z) #normalised such that the integral over all nu is 1.
                
                ans =  Lir*sed*self.Sigma(M)*self.redshiftevolutionofl(z)
                return ans
            
    
    def Scentral(self,nu,z,Mhalo):
        
        #flux from luminosity; eg eq 7 of 1611.04517
            chi=self.csm.chi_from_z(z)
            answer=self.Lnu(nu,z,Mhalo)/((4*np.pi)*chi**2*(1+z)) # in units of [Lnu]/Mpc**2=solar_luminosity/Mpc**2/Hz
            answer[answer>self.Scut(nu)]=0
            return answer
    
    def Luminosity_from_flux(self,S,z):
        #gives luminosity in [S] * Mpc**2
        return  4 * np.pi * self.csm.chi_from_z(z)**2*(1+z)*S
    
    def subhalo_mass_function(self,Msub,Mhost):
                jiang_gamma_1  = 0.13
                jiang_alpha_1   = -0.83
                jiang_gamma_2   = 1.33
                jiang_alpha_2 = -0.02
                jiang_beta_2     = 5.67
                jiang_zeta      = 1.19
                return (((jiang_gamma_1*((Msub/Mhost)**jiang_alpha_1))+
                 (jiang_gamma_2*((Msub/Mhost)**jiang_alpha_2)))*
                 (np.exp(-(jiang_beta_2)*((Msub/Mhost)**jiang_zeta))))
            
    
    def satellite_intensity(self,nu,zs,mhalos):
            satellite_masses=mhalos.copy()[:-1]
        
            dndms=self.subhalo_mass_function(satellite_masses,mhalos[:,np.newaxis])
            return np.trapz((dndms[:,:,np.newaxis]*self.Scentral(nu,zs,satellite_masses[:,np.newaxis])[np.newaxis,:,:]),np.log(satellite_masses),axis=1)
    
    def conversion(self,nu):
        if nu==353:
            return 287.45
        elif nu==545:
            return 58.04
        elif nu==857:
            return 2.27
    def sn(self,nu1,nu2):
            if [nu1,nu2]==[857,857]:
                ans= 5364
            elif [nu1,nu2]==[857,545] or [nu1,nu2]==[545,857]:
                ans= 2702
            elif [nu1,nu2]==[857,353] or [nu1,nu2]==[353,857]:
                ans= 953
            
            elif [nu1,nu2]==[545,545]:
                ans= 1690
            elif [nu1,nu2]==[545,353] or [nu1,nu2]==[353,545]:
                ans= 626
            
            elif [nu1,nu2]==[353,353] :
                ans= 262        
            else:
                ans= 0
            return ans*1/self.conversion(nu1)*1/self.conversion(nu2)
    
    def Scut(self,nu):
           #   experiment="Planck"
              if self.experiment=="Planck":
                  fluxcuts=np.array([400,350,225,315,350,710,1000])*1e-3
                  frequencies=[100,143,217,353,545,857,3000] # in gHz!!
            
                  if nu in frequencies:
                      return fluxcuts[frequencies.index(nu)]
              elif self.experiment=="Websky":
                      return 400*1e-3
              elif self.experiment=="Ccatprime":
                  frequencies=[220,280,350,410,850,3000]
                  fluxcuts=np.array([225,300,315,350,710,1000])*1e-3
                  if nu in frequencies:
                      return fluxcuts[frequencies.index(nu)]
    def prob(self,dummys,logexpectation_s,sigma):
            return 1/np.sqrt((2*np.pi*sigma**2))*np.exp(-(dummys[:,np.newaxis,np.newaxis]-logexpectation_s)**2/(2*sigma**2))
                  
    def dndlns(self,halomodel,dummys,logexpectation_s,sigma):
            mhalos=np.exp(halomodel.lnms)
            nfn=halomodel.nfn[np.newaxis]
            p=self.prob(dummys,logexpectation_s,sigma)
            integrand=nfn*p
          
            return np.trapz(integrand,mhalos,axis=1) 
        
    def shot_noise(self,nu,sigma,fluxes,zs,halomodel):
            chis=self.csm.chi_from_z(zs)
            
            fluxes[fluxes==0]=1e-100
            logfluxes=np.log(fluxes)
            dummylogs=np.linspace(np.min(logfluxes[logfluxes>-200])-0.5,min(self.Scut(nu),100),200)
            dnds=self.dndlns(halomodel,dummylogs,logfluxes,sigma)
           
             
            
            return np.trapz(chis**2*(np.trapz(dnds*np.exp(dummylogs[:,np.newaxis])**2,dummylogs,axis=0)),chis)
    def shot_noise_binned(self,nu,sigma,fluxes,zs,halomodel,zmin,zmax):
        
            chis=self.csm.chi_from_z(zs)
            
            fluxes[fluxes==0]=1e-100
            logfluxes=np.log(fluxes)
            dummylogs=np.linspace(np.min(logfluxes[logfluxes>-200])-0.5,min(self.Scut(nu),100),200)
            dnds=self.dndlns(halomodel,dummylogs,logfluxes,sigma)
           
            integrand = chis**2*(np.trapz(dnds*np.exp(dummylogs[:,np.newaxis])**2,dummylogs,axis=0))
            integrand[zs<zmin]=0
            integrand[zs<zmax]=0
    
            return np.trapz(integrand,chis)











