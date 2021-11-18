# -*- coding: utf-8 -*-
# CODE TO GENERATE POWER SPECTRA PK FROM HALO MODEL

import numpy as np
import halomodel
import common as c
import time
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d




def get_mthreshHOD_lsst(zs):
    fname = "data/mthreshHOD_lsst_new.txt"
    dt = ({ 'names' : ('zs', 'mmin'),'formats' : [np.float, np.float] })
    data = np.loadtxt(fname,dtype=dt) 
    return interp1d(data['zs'],data['mmin'],bounds_error=False,fill_value="extrapolate")(zs)

def getmthreshHODstellar(LSSexperiment,zs): #HOD threshold for a given experiment experiment as a function of red shift.
    if LSSexperiment == 'LSST':
        mthreshHODstellar = get_mthreshHOD_lsst(zs)
    elif LSSexperiment == 'unwise_blue':
        mthreshHODstellar = halomodel.mthreshHODdefault
    elif LSSexperiment == 'custom':
        mthreshHODstellar = halomodel.mthreshHODdefault
    else:
        raise Exception ("LSS experiment not supported.")      
    return mthreshHODstellar


class pks(object):

    def __init__(self, conf_module = None):

        if conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = c.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a configuration module to locate precomputed data") 
        self.hmod = None

    def check_pks(self,tag1,tag2,fq1,fq2):
        
        full_bool = c.exists(self.basic_conf_dir,  'p_'+c.retag(tag1)+c.retag(tag2)+'_f1='+str(fq1)+'_f2 ='+str(fq2) , dir_base = 'pks')
        lin_bool  = c.exists(self.basic_conf_dir,  'p_linear_'+c.retag(tag1)+c.retag(tag2)+'_f1='+str(fq1)+'_f2 ='+str(fq2) , dir_base = 'pks')
    
        return full_bool, lin_bool
    
    def load_pks(self,tag1,tag2,fq1,fq2, linear = False):
        if linear:
            return c.load(self.basic_conf_dir, 'p_linear_'+c.retag(tag2)+c.retag(tag2)+'_f1='+str(fq2)+'_f2 ='+str(fq2), dir_base = 'pks')
        else:
            return c.load(self.basic_conf_dir, 'p_'+c.retag(tag2)+c.retag(tag2)+'_f1='+str(fq2)+'_f2 ='+str(fq2), dir_base = 'pks')
    
    def start_halo(self,):
        if self.conf.z_min < 0.01 or self.conf.z_max > 5.0:
                    raise Exception("Halo model currently only supported for z_min >= 0.01 and z_max <= 5 ")    
        print("Starting halo model")
        self.hmod = halomodel.HaloModel(conf_module = self.conf)
        self.mthreshHODstellar = getmthreshHODstellar(self.conf.LSSexperiment,self.conf.zs_hm)         
        self.hmod._setup_k_z_m_grid_functions(self.conf.zs_hm,self.mthreshHODstellar,include_ukm=True,include_CIB= True)
        
    def get_pks(self,tag1,tag2,fq1,fq2):
        
        B1,B2 = self.check_pks(tag1,tag2,fq1,fq2)
        
        if B1 and B2:
            pass
        else:         
            if c.retag(tag1) in ['e','g','m','tSZ','CIB'] and c.retag(tag2) in ['e','g','m','tSZ','CIB'] :
                
                print("Computing "+c.retag(tag1)+c.retag(tag2)+" P(k,z) interpolating function from halo model")
                
                def spec(tag):      
                    if tag == 'e':
                        return 'gas'
                    elif tag == 'g':
                        if self.conf.LSSexperiment == 'LSST':
                            return 'gal'
                        elif self.conf.LSSexperiment == 'unwise_blue':   # we model the unWISE blue sample as a linearly biased tracer of dark matter 
                            return 'm'
                        elif self.conf.LSSexperiment == 'custom':   
                            return 'gal'
                        else:
                            raise Exception ("LSS experiment not supported.")
                    else:
                        return tag
                
                if self.hmod is None:
                    self.start_halo()                       
                    
                start = time.time()
                

                P_1h_sampled = self.hmod.P_1h(spec(c.retag(tag1)),spec(c.retag(tag2)),self.conf.ks_hm,self.conf.zs_hm, mthreshHOD=self.mthreshHODstellar,frequency=fq1,frequency2=fq2,gasprofile=self.conf.gasprofile,A=self.conf.A_electron)
                P_2h_sampled = self.hmod.P_2h(spec(c.retag(tag1)),spec(c.retag(tag2)),self.conf.ks_hm,self.conf.zs_hm, mthreshHOD=self.mthreshHODstellar,frequency=fq1,frequency2=fq2,gasprofile=self.conf.gasprofile,A=self.conf.A_electron)
  
                Pmm_lin_sampled = self.hmod.pk   
                if tag1 == 'g':
                    if self.conf.LSSexperiment == 'unwise_blue':
                        P_1h_sampled    *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                        P_2h_sampled    *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                        Pmm_lin_sampled *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                    else:
                        Pmm_lin_sampled *= self.hmod.bias_galaxy(self.conf.zs_hm,self.hmod.logmmin,self.hmod.logmmax,mthreshHOD=self.mthreshHODstellar)[:,np.newaxis]
                if tag2 == 'g':
                    if self.conf.LSSexperiment == 'unwise_blue':
                        P_1h_sampled    *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                        P_2h_sampled    *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                        Pmm_lin_sampled *= (0.8+1.2*self.conf.zs_hm)[:,np.newaxis]
                    else:
                        Pmm_lin_sampled *= self.hmod.bias_galaxy(self.conf.zs_hm,self.hmod.logmmin,self.hmod.logmmax,mthreshHOD=self.mthreshHODstellar)[:,np.newaxis]
                
                Pfull_sampled = P_1h_sampled + P_2h_sampled     
                pkfull = interp2d(self.conf.ks_hm,self.conf.zs_hm,Pfull_sampled, kind = 'linear',bounds_error=False,fill_value=0.0)
                
                if 'tSZ' in [tag1,tag2] or 'CIB' in [tag1,tag2]:
                    pklin = interp2d(self.conf.ks_hm,self.conf.zs_hm,P_2h_sampled, kind = 'linear',bounds_error=False,fill_value=0.0)
                else:
                    pklin = interp2d(self.conf.ks_hm,self.conf.zs_hm,Pmm_lin_sampled, kind = 'linear',bounds_error=False,fill_value=0.0)
                
                c.dump(self.basic_conf_dir, pkfull, 'p_'+c.retag(tag1)+c.retag(tag2)+'_f1='+str(fq1)+'_f2 ='+str(fq2), dir_base = 'pks')
                c.dump(self.basic_conf_dir, pklin , 'p_linear_'+c.retag(tag1)+c.retag(tag2)+'_f1='+str(fq1)+'_f2 ='+str(fq2), dir_base = 'pks')
                
                
                end = time.time()       
                print("Seconds to compute P_"+c.retag(tag1)+c.retag(tag1)+":", end-start)
                
                pass
                
            else:
                raise Exception("Power spectra for "+c.retag(tag1)+c.retag(tag1)+" not yet supported")    
 
        




