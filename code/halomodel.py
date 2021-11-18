
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import scipy
import scipy.integrate as integrate
from scipy.interpolate import interp2d,interp1d
import common as cm
import time
import cosmology
import CIB_Planck
import CIB_Websky

mthreshHODdefault = np.array( [10.5] ) 
allsky_squaredeg = 4*np.pi*(180.**2./np.pi**2.)
allsky_arcmin2 = 4*np.pi*(10800/np.pi)**2
allsky_sterad = 4*np.pi

class HaloModel(object):

    def __init__(self, conf_module = None, npts=1000,include_ykm=True,use_counterterms = False,mmin_nfn = 1e10):
        
        if conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = cm.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a hash string or configuration module to locate precomputed data") 
            
                #cosmology functions
        self.csm = cosmology.cosmology(conf_module = self.conf)
            
        if self.conf.CIB_model=="Planck":
            self.CIB = CIB_Planck.CIB(cosmo = self.csm)
        elif self.conf.CIB_model=="Websky":
            self.CIB = CIB_Websky.CIB(cosmo = self.csm)
                    
        #flags
        self.onehdamping = False #damping envelope
        self.onehdamping_kstar = 0.03
     
        # Setup the cosmology, range of halo mass, and collapse model.

        #--------- Define the cosmology (TODO replace by Bint)
          
        self.omegam = self.conf.Omega_c + self.conf.Omega_b
        self.H0mpc = self.csm.H0(self.conf.h)
        self.rhocrit = (self.conf.h**2)*2.77536627*10**(11)  # Critical density today in M_\odot Mpc^-3
        self.rhomatter = self.omegam*self.rhocrit     

        #----- useful constants

        self.mElect = 4.579881126194068e-61
        self.thompson_SI = 6.6524e-29

        #-------- Define the dark matter halo mass and collapse model

        self.npts=npts #400            # Number of points equally spaced in log_10 m to sample. Default = 400
        self.logmmin=8  #4          # log_10 m/M_\odot of the minimum mass halo to sample. for kmax=1000, should be > 4. Default = 4.
        self.logmmax=16  #17          # log_10 m/M\odit of the maximum mass halo to sample. Default = 17.
        

        self.c1=0.3222     # Uncomment for Sheth-Torman mass function:
        self.c2=0.707
        self.p=0.3

        self.deltac=1.686                         # Critical density for collapse.           

        #sampling variables
        self.k = self.conf.ks_hm
        self.z = [] #filled later

        self.mdef=self.conf.mdef
        
        self.include_ykm = include_ykm
        
        self.use_counterterms = use_counterterms
        self.mmin_nfn = mmin_nfn /(self.conf.H0/100) 
        

    
    #generates self.mthresHOD[z] after z sampling has been set up
    def _setup_mthreshHOD(self,mthreshHOD):
        #Threshold stellar mass for galaxy HOD at all z. Default: 10.5 
        if mthreshHOD.shape[0] == self.z.shape[0]:
            self.mthreshHOD=mthreshHOD
        else:
            #if nthresh is length 1, broadcast over z
            self.mthreshHOD=mthreshHOD[0]*np.ones(self.z.shape[0])

    def bias(self,nu):
        y=np.log10(200)
        A=1+0.24*y*np.exp(-(4/y)**4)
        a=0.44*y-0.88
        B=0.183
        b=1.5
        C=0.019+0.107*y+0.19*np.exp(-(4/y)**4)
        c=2.4
        return 1-A*nu**a/(nu**a+1.686**a)+B*nu**b+C*nu**c
    #set up class variables for the k and z sampling demanded in power spectrum evaluations
    def _setup_k_z_m_grid_functions(self,z,mthreshHOD= mthreshHODdefault,include_ukm=True,include_CIB=True,):
        
        
        if np.array_equal(self.z,z): #no need to setup the grid functions if they are the same as on the last call
            #now check if the HOD also is unchanged.
            if np.array_equal(self.mthreshHOD,mthreshHOD): #yes, unchanged
                return
            else: #no, changed. set up HOD threshold and return.
                self._setup_mthreshHOD(mthreshHOD)
                return

        start = time.time() #<<<<
        # ------- Define the grid in redshift, k and mass.
        
        self.z = z
        self.logm=np.linspace(self.logmmin,self.logmmax,self.npts)             # Mass grid, linear in log_10 m.
        self.lnms=np.log(10**(self.logm))                                      # Mass grid, linear in ln m.
        self.mhalos = 10**self.logm
        #--------- set up HOD m_thresh
        self._setup_mthreshHOD(mthreshHOD)
        end = time.time() #<<<< HOD setup
        print("HOD setup time:", end-start)
        
        #-------- Calculate cosmology, CAMB power spectra, and variables in the collapse model. (Takes a minute, go get a coffee and come back :).

        # Calculate the linear and non-linear matter power spectrum at each redshift using CAMB. 
        # Units are cubic Mpc as a function of k (e.g. no factors of h).
      
        #self.pknl = self.csm.camb_Pk_nonlin(self.k,self.z) 
        self.pk   = self.csm.camb_Pk_lin(self.k,self.z) 

        # A few background parameters using CAMB functions
        self.Dist = self.csm.chi_from_z(self.z)      # Comoving distance as a function of z.
        self.hubbleconst=self.csm.H_z(self.z)                    # H(z)     
        self.rhocritz=self.rhocrit*(self.hubbleconst**2)/(self.H0mpc**2)      # Critical density as a function of z. 
        self.omegamz=1./(1.+(1-self.omegam)/(self.omegam*(1+self.z)**3))      # Omega_m as a function of z 

        # Get the growth function from CAMB

        growthfn = self.csm.data.get_redshift_evolution(.0001, self.z, ['delta_baryon'])  #Extract the linear growth function from CAMB.
        self.growthfn = growthfn/growthfn[0]

        #--------- Define some variables.

        self.nu = np.zeros([self.npts,self.z.shape[0]])              # nu=delta_c^2/sigma^2
        #self.mstar = np.zeros([self.z.shape[0]])                     # Mass where nu=1
        self.conc = np.zeros([self.npts,self.z.shape[0]])            # Concentration parameter for NFW profile conc=r_vir/r_s as a function of mass and z.
        self.concHI = np.zeros([self.npts,self.z.shape[0]])          # Concentration parameter for HI profile conc=r_vir/r_s as a function of mass and z.
        self.rvir3 = np.zeros([self.npts,self.z.shape[0]])           # Virial radius cubed as a function of mass and z
        self.sigmam2=np.zeros([self.npts,self.z.shape[0]])           # Variance using top hat window at mass M.
        self.dlogsigmadlogm=np.zeros([self.npts,self.z.shape[0]])    # d\ln\sigma/d\ln m
        self.fsigma=np.zeros([self.npts,self.z.shape[0]])            # Collapse fraction assuming Press-Schechter or Sheth-Torman.
        self.nfn=np.zeros([self.npts,self.z.shape[0]])               # Differential number density of halos n(m,z). Note, this is comoving number density.
        self.halobias=np.zeros([self.npts,self.z.shape[0]])          # Linear halo bias

        # Loops to calculate variables above.

        # Redshift loop.
        start=time.time()
        Rs=(3.*10**(self.logm)/(4.*np.pi*self.rhomatter))**(1./3)
        # Argument of the window functions. Note, Rs and k are both comoving.
        xs = Rs[:,np.newaxis]*self.k
        # Define the window function at different masses. For numerical accuracy, use Taylor expansion at low kR.
        self.window = np.piecewise(xs,[xs>.01,xs<=.01],[lambda xs: (3./(xs**3))*(np.sin(xs)-(xs)*np.cos(xs)), lambda xs: 1.-xs**2./10.])
        # \sigma^2 as a function of mass, implement redshift dependence through the growth function.
        self.sigmam2 = np.transpose( (self.growthfn[np.newaxis,:]**2)*integrate.simps(self.k*self.k*self.pk[0]*self.window**2/(2.*np.pi**2), self.k) )[:,:,0]
        self.nu = self.deltac**2/self.sigmam2
        # Derivative of the window function, use Taylor expansion at low kR.
        self.dwindow2dm = -np.piecewise(xs,[xs>.01,xs<=.01],[lambda xs: (6./xs**6)*(np.sin(xs)-xs*np.cos(xs))*(np.sin(xs)*(-3.+xs**2) + 3.*xs*np.cos(xs)), lambda xs: -2.*xs**2./15.])
        # Calculate log derivative of \sigma; note that powers of the growth function cancel so no redshift dependence.
        self.dlogsigmadlogm = np.transpose( (1./self.sigmam2[:,0])*integrate.simps(self.k*self.k*self.pk[0]*self.dwindow2dm/(4.*np.pi**2), self.k) )
        # Collapse fraction assuming Press-Schechter or Sheth-Torman.
        
        
        if self.conf.halomassfunction != 'Tinker':
        
            self.fsigma = self.c1*np.sqrt(2.*self.c2/np.pi)*(1.+(self.sigmam2/(self.c2*self.deltac**2))**self.p)*(self.deltac/np.sqrt(self.sigmam2))*np.exp(-0.5*self.c2*self.deltac**2/self.sigmam2)
            # Put together pieces of the differential number density and halo bias. NOTE: this is overwritten if we are using Tinker halo mass function.
            self.nfn=(self.rhomatter/(10**(2.*self.logm))*self.dlogsigmadlogm)[:,np.newaxis]*self.fsigma
            self.halobias=1.+(self.c2*(self.deltac**2/self.sigmam2)-1.)/self.deltac+(2.*self.p/self.deltac)/(1.+ (self.c2*(self.deltac**2/self.sigmam2))**self.p )
            # Calculate the NFW concentration parameter and halo virial radius.
            #self.deltav = 178*self.omegamz**(0.45)                  # Virial overdensity, approximated as in Eke et al. 1998.  
            
            
            
        self.fb=self.conf.Omega_b/self.conf.Omega_m 
        omega = self.omegam*(1+self.z)**3/(self.omegam*(1+self.z)**3+(1-self.omegam))
        self.deltav = 18*np.pi**2+82*(omega - 1)-39*(omega -1)**2 # is this okay?
        #self.mstar = self.logm[(np.abs(self.nu-1.)).argmin(axis=1)]
        # Several choices for the NFW halo concentration parameter as a function of mass. Here, assumed deterministic.
        #conc[:,j] = 9.*(10**(-.13*(logm-mstar[j]*np.ones(logm.shape[0]))))/(1+z[j])    # Bullock model from Cooray/Sheth
        self.concHI = 4.*25.*(10**(self.logm[:,np.newaxis]-11.))**(-.109)/(1+self.z)**0.1 # model from HI_halo_model notebook
        #TODO: make this robust to halo mass definition
        # # Cube of the virial radius. Note, this is a physical distance, not a comoving distance.
        #self.rvir3 = 3.*(10**self.logm[:,np.newaxis])/(4.*np.pi*self.deltav*self.rhocritz)
        self.density_vir = self.deltav * self.rhocritz
        self.density_200c = 200 * self.rhocritz
        self.density_200d = 200 * self.rhomatter *(1+self.z)**3
        self.conc = self.duffy_concentration(self.conf.mdef,self.mhalos[:,np.newaxis],self.z[np.newaxis,:])#7.85*(( self.h*0.5*10**(self.logm[:,np.newaxis]-12) )**(-0.081))*(1+self.z)**(-0.71) # Duffy08 all model from 1005.0411
        #concentration is mass definition dependent. the parametric forms are in https://arxiv.org/pdf/0804.2486.pdf

        #setup halo radius and compute different mass definitions:
        if self.conf.mdef == 'm200d':

            self.r200_3_d = 3.*(10**self.logm[:,np.newaxis])/(4.*np.pi*self.density_200d)
            self.r200_d = self.r200_3_d**(1/3)
            
            self.mvir = np.zeros(self.r200_d.shape)
            self.rvir = np.zeros(self.r200_d.shape)

            self.m200_c = np.zeros(self.r200_d.shape)
            self.r200_c = np.zeros(self.r200_d.shape)

            for mind in range(0,self.npts):
                for zind in range(0,self.z.shape[0]):

                    self.mvir[mind,zind],self.rvir[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_vir[zind],self.r200_d[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
                    self.m200_c[mind,zind],self.r200_c[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_200c[zind],self.r200_d[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
                
        elif self.conf.mdef == 'm200c':

            self.r200_3_c = 3.*(10**self.logm[:,np.newaxis])/(4.*np.pi*self.density_200c)
            self.r200_c = self.r200_3_c**(1/3)
            
            self.mvir = np.zeros(self.r200_d.shape)
            self.rvir = np.zeros(self.r200_d.shape)

            self.m200_d = np.zeros(self.r200_c.shape)
            self.r200_d = np.zeros(self.r200_c.shape)

            for mind in range(0,self.npts):
                for zind in range(0,self.z.shape[0]):

                    self.mvir[mind,zind],self.rvir[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_vir[zind],self.r200_c[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
                    self.m200_d[mind,zind],self.r200_d[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_200d[zind],self.r200_c[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
        elif self.conf.mdef == 'mvir':

            self.r_3_vir = 3.*(10**self.logm[:,np.newaxis])/(4.*np.pi*self.density_vir)
            self.rvir = self.r_3_vir**(1/3)
            
            self.m200_c = np.zeros(self.rvir.shape)
            self.r200_c = np.zeros(self.rvir.shape)

            self.m200_d = np.zeros(self.rvir.shape)
            self.r200_d = np.zeros(self.rvir.shape)

            for mind in range(0,self.npts):
                for zind in range(0,self.z.shape[0]):

                    self.m200_c[mind,zind],self.r200_c[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_200c[zind],self.rvir[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
                    self.m200_d[mind,zind],self.r200_d[mind,zind] = self.m1_to_m2(self.conc[mind,zind],self.density_200d[zind],self.rvir[mind,zind],self.mhalos[mind])#(200,zind,duffy_concentration(duffyparams_mvir,m_vir[mind],z[zind]),r_vir[mind,zind],m_vir[mind])
        self.rvir3 = self.rvir**3
            
        if self.conf.halomassfunction == 'Tinker':  #to use the tinker mass function you need your mdef to be m200d.
            assert self.conf.mdef =="m200d"   #TODO: make this work with different mass definitions?
            mhalos=np.exp(self.lnms)
            zslessthan3=self.z.copy()
            zslessthan3[zslessthan3>3]=3
            beta0 = 0.589
            gamma0 = 0.864
            phi0 = -0.729
            eta0 = -0.243
            beta  = beta0  * (1+zslessthan3)**(0.20)
            phi   = phi0   * (1+zslessthan3)**(-0.08)
            eta   = eta0   * (1+zslessthan3)**(0.27)
            gamma = gamma0 * (1+zslessthan3)**(-0.01)
            izs,ialphas = np.loadtxt("alpha_consistency.txt",unpack=True) # FIXME: hardcoded #see hmvec.py, Mat Madhavacheril's code, for the computation of this file.
            alpha = interp1d(izs,ialphas,bounds_error=True)(zslessthan3)
            nu=np.sqrt(self.nu)  # Tinker's nu is the square root of our nu.
            self.fsigma=alpha*np.sqrt(self.nu)*(1. + (beta*nu)**(-2.*phi))*(nu**(2*eta))*np.exp(-gamma*nu**2./2.)#A*((np.sqrt(self.sigmam2)/b)**-a+1)*np.exp(-c/self.sigmam2)# self.c1*np.sqrt(2.*self.c2/np.pi)*(1.+(self.sigmam2[i,j]/(self.c2*self.deltac**2))**self.p)*(self.deltac/np.sqrt(self.sigmam2[i,j]))*np.exp(-0.5*self.c2*self.deltac**2/self.sigmam2[i,j])
            dndm=self.rhomatter/mhalos[:,np.newaxis]**2*self.fsigma*self.dlogsigmadlogm[:,np.newaxis]
            self.dndm=dndm
            self.nfn=dndm
            self.halobias=self.bias(np.sqrt(self.nu))
            
            if self.use_counterterms:
               self.set_counterterms(self.mmin_nfn)
                            
            
            
        end=time.time()
        print("Redshift loop time:",end-start)

        #-----p_mm_2h_cts Properties of the dark matter halo.

        # Calculate the normalized FT of the NFW profile as a function of halo mass and z. Note, the FT is in terms of the 
        # comoving wavenumber.
        
        
        if include_ukm:
            start=time.time()
            # Changing this loop to slicing makes things slow. Why?
            self.ukm = np.zeros([self.npts,self.z.shape[0],self.k.shape[0]])
            for j in range(0,self.z.shape[0]):
                for i in range(0,self.logm.shape[0]):
                    c = self.conc[i,j]
                    mc = np.log(1+c)-c/(1.+c)
                    if self.mdef=="m200d":
                        
                        rs = (self.r200_3_d[i,j]**(0.33333333))/c
                    elif self.mdef=="mvir":
                        rs = (self.rvir3[i,j]**(0.33333333))/c
                        
                    elif self.mdef=="m200c":
                        rs = (self.r200_3_c[i,j]**(0.33333333))/c
    
    
                    # Include a factor of (1+z) to account for physical k vs comoving k.
                    x = self.k*rs*(1+self.z[j])
                    Si, Ci = scipy.special.sici(x)
                    Sic, Cic = scipy.special.sici((1.+c)*x)
                    self.ukm[i,j,:] = (np.sin(x)*(Sic-Si) - np.sin(c*x)/((1+c)*x) + np.cos(x)*(Cic-Ci))/mc

            end=time.time() # <<< HERE
            print("ukm loop time:",end-start)
    
            #----- Calculate normalized FT's of the gas profiles as a function of mass, z, and k (comoving). interpolate to the present k and z sampling.
                            
            self.ukgas = self.calc_gasprofiles(self.conf.gasprofile,cutoffradius=None)
           
        
        #ykm for tsz
        if self.include_ykm:
                              
            self.ykm= self.make_yk_phys()

        #---CIB fluxes
        if include_CIB:
            self.central_flux={}
            self.satflux={}
            print('setting up CIB fluxes')
            t1=time.time()
    
            freqs = self.conf.cleaning_frequencies[self.conf.cleaning_mode]
                 
            for frequency in freqs:
                        
                self.central_flux[str(frequency)]=(self.CIB.Scentral(frequency,z,np.exp(self.lnms)[:,np.newaxis]))
                self.satflux[str(frequency)]=((self.CIB.satellite_intensity(frequency,z,np.exp(self.lnms))))
                #impose  flux cut:
                self.satflux[str(frequency)][self.central_flux[str(frequency)]==0]=0
                self.central_flux[str(frequency)][self.satflux[str(frequency)]>self.CIB.Scut(frequency)]=0
                self.satflux[str(frequency)][self.satflux[str(frequency)]>self.CIB.Scut(frequency)]=0
    
            print("computed CIB fluxes in",time.time()-t1)
        #----- Properties of HI in dark matter halos.

        # Calculate the normalized FT of the HI profile as a function of halo mass and z. Note, the FT is in terms of the 
        # comoving wavenumber.
        
        #TODO: check for mass definition compatibility!

        self.ukHI = np.zeros([self.npts,self.z.shape[0],self.k.shape[0]]) 
        for j in range(0,self.z.shape[0]):
            for i in range(0,self.logm.shape[0]):
                c = self.concHI[i,j]
                rs = (self.rvir3[i,j]**(0.33333333))/c
                Rv = self.rvir3[i,j]**(0.33333333)
                # Include a factor of (1+z) to account for physical k vs comoving k.
                krs = self.k*rs*(1+self.z[j])
                sici_075krs = scipy.special.sici(0.75*krs)
                sici_075concHIkrs = scipy.special.sici((0.75+c)*krs)
                sici_1concHIkrs = scipy.special.sici((1.+c)*krs)
                sici_krs    = scipy.special.sici(krs)
                uhi_1 = -(-12.*sici_075krs[1]*np.sin(0.75*krs)\
                  +12.*sici_075concHIkrs[1]*np.sin(0.75*krs)\
                  +sici_1concHIkrs[1]*(4.*krs*np.cos(krs)-12.*np.sin(krs))\
                  +sici_krs[1]*(-4.*krs*np.cos(krs)+12.*np.sin(krs))\
                  -4.*np.sin(c*krs)/(1.+c)\
                  +12.*np.cos(0.75*krs)*sici_075krs[0] - 12.*np.cos(krs)*sici_krs[0]\
                  -4.*krs*np.sin(krs)*sici_krs[0]\
                  -12.*np.cos(0.75*krs)*sici_075concHIkrs[0]\
                  +sici_1concHIkrs[0]*(12.*np.cos(krs)+4.*krs*np.sin(krs)))/krs
                self.ukHI[i,j,:] = -uhi_1/(-4.*Rv/(rs+Rv) - np.log(rs) - 8.*np.log(rs+Rv) + 9.*np.log(rs+4.*Rv/3.))      
    
    def get_ykm(self,):
        
        if cm.exists(self.basic_conf_dir,  'yk_battaglia' , dir_base = 'hm_data'):
            return cm.load(self.basic_conf_dir, 'yk_battaglia', dir_base = 'hm_data')     
        else:
            pass
            
            
            
            
    def rescale_electronprofile(self,A):
        ukgas_rescaled = np.zeros(self.ukgas.shape)
        for zind in range(len(self.z)):
            interp_ue = interp2d(self.k,self.logm,self.ukgas[:,zind,:])
            interp_nfw = interp2d(self.k,self.logm,self.ukm[:,zind,:])
            ukgas_rescaled[:,zind,:] = interp_ue(self.k[:]*A,self.logm)/interp_nfw(self.k*A,self.logm)*self.ukm[:,zind,:]
        return ukgas_rescaled
    # Function to compute dark matter density profile in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius
    # mv = mass
    # rvals = values of r to return profile at. 
    # Note: All distances are physical distances, not comoving distances.
    def dmprofile(self,c,rv,mv,rvals):
        xvals=c*rvals/rv
        nfw=1./(xvals*(1+xvals)**2)
        mc=np.log(1.+c ) - c/(1.+c)
        rhos=(c**3)*mv/(4.*mc*np.pi*rv**3)
        return rhos*nfw
    def m1_to_m2(self,duffy_concentration_m1,density_definition_m2,r1,m1):
        #this function converts mass from definition 1 to definition 2
        #inputs: duffy_concentration_m1 = concentration of halo as definied by m1
        #        density_definition_m2 = mean density of halo with mass definition you want to convert to
        #        r1 = radius of halo defined by definition of m1 (with density defined by m1's density definition)
        #        m1 = mass of halo 
        #eg to go from mvir to m200c arguments should be (concentration_vir(mvir),200*rhocritz,rvir,mvir)
        c = duffy_concentration_m1
        rdeltavals = np.linspace(r1/10,2*r1,1000)
        mc=np.log(1.+c ) - c/(1.+c)
        mcr=np.log(1.+ (c*rdeltavals/r1) ) - (c*rdeltavals/r1)/(1.+(c*rdeltavals/r1))
        fn = np.abs(rdeltavals**3 - 3.*m1*mcr/(4.*np.pi*density_definition_m2*mc))
        rdelta = rdeltavals[fn.argmin()]
        mcrdelta = np.log(1.+ (c*rdelta/r1) ) - (c*rdelta/r1)/(1.+(c*rdelta/r1))
        mdelta = mcrdelta*m1/mc
        return mdelta,rdelta
    
    def duffy_concentration(self,mdef,M,z): #concentrations from https://arxiv.org/pdf/0804.2486.pdf
        if mdef == "mvir":
            A = 7.85
            B = -0.081
            C = -0.71
        elif mdef == "m200d":
            A = 10.14
            B = -0.081
            C = -1.01
        elif mdef == "m200c":
            
            A = 5.71
            B = -0.084
            C = -0.47
                
        Mpivot=2e12*1/self.conf.h
        return A*(M/Mpivot)**B*(1+z)**C
    
    def set_counterterms(self,mmin):
                    ##  this sets the halo mass function and halo bias below mmin
                    ## (which is the minimum mass over which they were calibrated against nbody sims)
                    ## to a counter term according to arXiv:1511.02231
                    ## 
                    ## For Tinker 2008, mmin should be 1e10 or 1e11 h^{-1} M_odot
            
            
                    mmin_nfn_ongrid = self.mhalos[self.mhalos>=mmin][0]
                    self.mindex_mmin_nfn = list(self.mhalos).index(mmin_nfn_ongrid)

            
                    newnfn = self.nfn.copy()
                
                    constraint_integrand = self.mhalos[self.mhalos>mmin,np.newaxis]/self.rhomatter*self.nfn[self.mhalos>mmin,:]
                    integral_nfn = np.trapz(constraint_integrand,self.mhalos[self.mhalos>mmin],axis=0)
                
                    
                    self.nmin = nmin = (1-integral_nfn)*self.rhomatter/mmin
                    
                    constraint_integrand_bias = self.halobias[self.mhalos>mmin]*self.mhalos[self.mhalos>mmin,np.newaxis]/self.rhomatter*self.nfn[self.mhalos>mmin,:]
                    integral_bias = np.trapz(constraint_integrand_bias,self.mhalos[self.mhalos>mmin],axis=0)
                
                    self.biasmin = biasmin = (1-integral_bias)*self.rhomatter/mmin/nmin
                
                    newbias = self.halobias.copy()
                
                    for x in range(0,newnfn.shape[-1]):
                        (newnfn[self.mhalos<mmin,x]) = 0#nmin[x]
            
                    for x in range(0,newbias.shape[-1]):
                        newbias[self.mhalos<mmin,x] = biasmin[x]
                    self.halobias = newbias.copy()
                   # self.nfn = newnfn.copy()
                
    '''    
    # Function to convert r_vir and m_vir to r_delta and m_delta.
    # delta = overdensity (e.g. 200, 500, etc)
    # zindex = redshift index to do the computation
    # c = concentration
    # rv = virial radius
    # mv = virial mass
    # Note: All distances are physical distances, not comoving distances.
    def mrvir_to_mrdelta(self,delta,zindex,c,rv,mv):
        rdeltavals = np.linspace(rv/10,2*rv,1000)
        mc=np.log(1.+c ) - c/(1.+c)
        mcr=np.log(1.+ (c*rdeltavals/rv) ) - (c*rdeltavals/rv)/(1.+(c*rdeltavals/rv))
        fn = np.abs(rdeltavals**3 - 3.*mv*mcr/(4.*np.pi*delta*self.rhocritz[zindex]*mc))
        rdelta = rdeltavals[fn.argmin()]
        mcrdelta = np.log(1.+ (c*rdelta/rv) ) - (c*rdelta/rv)/(1.+(c*rdelta/rv))
        mdelta = mcrdelta*mv/mc
        return mdelta,rdelta
    '''
    #-------- HOD following the "SIG_MOD1" HOD model of 1104.0928 and 1103.2077 with redshift dependence from 1001.0015.
    # See also 1512.03050, where what we are using corresponds to their "Baseline HOD" model.

    # Function to compute the stellar mass Mstellar from a halo mass mv at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    def Mstellar_halo(self,z,log10mhalo):
        a = 1./(1+z)
        if (z<=0.8):
            Mstar00=10.72
            Mstara=0.55
            M1=12.35
            M1a=0.28
            beta0=0.44
            beta_a=0.18
            gamma0=1.56
            gamma_a=2.51
            delta0=0.57
            delta_a=0.17
        if (z>0.8):
            Mstar00=11.09
            Mstara=0.56
            M1=12.27
            M1a=-0.84
            beta0=0.65
            beta_a=0.31
            gamma0=1.12
            gamma_a=-0.53
            delta0=0.56
            delta_a=-0.12            
        log10M1 = M1 + M1a*(a-1)
        log10Mstar0 = Mstar00 + Mstara*(a-1)
        beta = beta0 + beta_a*(a-1)
        gamma = gamma0 + gamma_a*(a-1)
        delta = delta0 + delta_a*(a-1)
        log10mstar = np.linspace(-18,18,1000)
        mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
        mstar = np.interp(log10mhalo,mh,log10mstar)
        return mstar

    # Function to compute halo mass as a function of the stellar mass.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    def Mhalo_stellar(self,z,log10mstellar):
        a = 1./(1+z) 
        if (z<=0.8):
            Mstar00=10.72
            Mstara=0.55
            M1=12.35
            M1a=0.28
            beta0=0.44
            beta_a=0.18
            gamma0=1.56
            gamma_a=2.51
            delta0=0.57
            delta_a=0.17
        if (z>0.8):
            Mstar00=11.09
            Mstara=0.56
            M1=12.27
            M1a=-0.84
            beta0=0.65
            beta_a=0.31
            gamma0=1.12
            gamma_a=-0.53
            delta0=0.56
            delta_a=-0.12            
        log10M1 = M1 + M1a*(a-1)
        log10Mstar0 = Mstar00 + Mstara*(a-1)
        beta = beta0 + beta_a*(a-1)
        gamma = gamma0 + gamma_a*(a-1)
        delta = delta0 + delta_a*(a-1)
        log10mstar = log10mstellar
        log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
        return log10mh

    # Number of central galaxies as a function of halo mass and redshift.
    # lnms = natural log of halo masses
    # z = redshifts
    # log10Mst_thresh = log10 of the stellar mass threshold. Defined above as mthresh with the other free parameters.
    # returns array[mass,z]
    def Ncentral(self,lnms,z,log10Mst_thresh):
        logm10=np.log10(np.exp(lnms))
        log10Mst=self.Mstellar_halo(z,logm10) 
        sigmalogmstar=0.2
        log10Mst_threshar=log10Mst_thresh*np.ones(log10Mst.shape[0])
        arg = (log10Mst_threshar-log10Mst)/(np.sqrt(2)*sigmalogmstar)
        return 0.5-0.5*scipy.special.erf(arg)

    # Number of satellite galaxies.
    # lnms = natural log of halo masses
    # z = redshifts
    # log10Mst_thresh = log10 of the stellar mass threshold. Defined above as mthresh with the other free parameters.
    # returns array[mass,z]
    def Nsatellite(self,lnms,z,log10Mst_thresh):
        logm10=np.log10(np.exp(lnms))
        Bsat=9.04
        betasat=0.74
        alphasat=1.
        Bcut=1.65
        betacut=0.59
        Msat=(10.**(12.))*Bsat*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betasat)
        Mcut=(10.**(12.))*Bcut*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betacut)
        return self.Ncentral(lnms,z,log10Mst_thresh)*((np.exp(lnms)/Msat)**alphasat)*np.exp(-Mcut/(np.exp(lnms)))

    ####---- functions for P_ge where e is all mass
    # Number of centrals in a mass bin
    # returns array[mass,z]
    def Ncentral_binned(self,lnms,z,log10Mst_thresh,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        logm10=np.log10(np.exp(lnms))
        log10Mst=self.Mstellar_halo(z,logm10)
        sigmalogmstar=0.2
        log10Mst_threshar=log10Mst_thresh*np.ones(log10Mst.shape[0])
        arg = (log10Mst_threshar-log10Mst)/(np.sqrt(2)*sigmalogmstar)
        Ncentral=np.zeros(lnms.shape[0])
        Ncentral[(lnms>=lnmlow)&(lnms<=lnmhigh)]=0.5-0.5*scipy.special.erf(arg[(lnms>=lnmlow)&(lnms<=lnmhigh)])
        #print (Ncentral)
        return Ncentral
    
    # Number of satellites in a mass bin
    # returns array[mass,z]
    def Nsatellite_binned(self,lnms,z,log10Mst_thresh,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        logm10=np.log10(np.exp(lnms))
        Bsat=9.04
        betasat=0.74
        alphasat=1.
        Bcut=1.65
        betacut=0.59
        Msat=(10.**(12.))*Bsat*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betasat)
        Mcut=(10.**(12.))*Bcut*10**((self.Mhalo_stellar(z,log10Mst_thresh)-12)*betacut)
        Nsatellite = self.Ncentral(lnms,z,log10Mst_thresh)*((np.exp(lnms)/Msat)**alphasat)*np.exp(-Mcut/(np.exp(lnms)))
        Nsatellite[(lnms<=lnmlow)] = 0.0
        Nsatellite[(lnms>=lnmhigh)] = 0.0
        #print (Nsatellite)
        return Nsatellite
    ####----
    # Average comoving number density of galaxies.
    # lnms = natural log of halo masses
    # z = redshifts
    # nfn = halo mass function
    # Ngal = number of galaxies as a function of halo mass and redshift
    def Nbargal(self,lnms,z,nfn,Ngal):
        nbargal=np.zeros([z.shape[0]])
        for j in range(0,z.shape[0]):
            nbargal[j]=integrate.simps(np.exp(lnms)*nfn[:,j]*Ngal[:,j],lnms)
        return nbargal

    # Function to calculate mass-binned galaxy bias.
    def _galaxybias(self,lnms,z,nfn,Ngalc,Ngals,halobias,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalcin=Ngalc[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalsin=Ngals[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        halobiasin=halobias[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ngalcin+Ngalsin)
        bgal = np.zeros([z.shape[0]])
        for j in range(0,z.shape[0]):
            bgal[j]=integrate.simps((np.exp(lnmsin))*nfnin[:,j]*(Ngalcin[:,j]+Ngalsin[:,j])*halobiasin[:,j]/nbargaltot[j],lnmsin)
        return bgal
    
    # Function to calculate mass-binned galaxy number density.
    def Nbargal_binned(self,lnms,z,nfn,Ngal,log10mlow,log10mhigh):
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalin=Ngal[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        #print (lnmsin.shape[0])
        nbargalbinned=np.zeros(z.shape[0])
        for j in range(0,z.shape[0]):
            nbargalbinned[j]=integrate.simps(np.exp(lnmsin)*nfnin[:,j]*Ngalin[:,j],lnmsin)
        return nbargalbinned
    
    # Function to compute gas density profile from Komatsu and Seljak (astro-ph/0106151) in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # rstar = value at which slope of gas profile matches that of NFW. Choose to be within a factor of 2 larger or smaller than rv, and no effect on profile.
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile(self,c,rv,mv,rvals,rstar):
        xstar=c*rstar/rv
        xvals=c*rvals/rv
        nfw=1./(xvals*(1+xvals)**2)
        mc=np.log(1.+c ) - c/(1.+c)
        rhos=(c**3)*mv/(4.*mc*np.pi*rv**3)
        gamma=1.15+0.01*(c-6.5)
        indexxstar=(np.abs(xvals - xstar)).argmin()
        sstar=-(1.+2.*xstar/(1.+xstar))
        mxstar=np.log(1.+xstar) - xstar/(1.+xstar)
        mxstarint = 1. - np.log(1.+xstar)/xstar
        eta0=-(3./(gamma*sstar))*(c*mxstar/(xstar*mc)) + 3.*(gamma-1)*c*mxstarint/(gamma*mc)
        gas = ((1. - (3./eta0)*((gamma-1.)/gamma)*(c/mc)*(1.-np.log(xvals+1.)/(xvals)))**(1./(gamma-1.)))
        gasrescale=(nfw[indexxstar]/gas[indexxstar])*rhos*(self.conf.ombh2/self.conf.omch2)
        return np.piecewise(xvals,[xvals<=xstar,xvals>xstar],[lambda xvals: gasrescale*((1. - (3./eta0)*((gamma-1.)/gamma)*(c/mc)*(1.-np.log(xvals+1.)/(xvals)))**(1./(gamma-1.))), lambda xvals: rhos*(self.conf.ombh2/self.conf.omch2)/(xvals*(1+xvals)**2)])

    # Function to compute gas density profile from Battaglia (1607.02442) AGN feedback model in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # zindex = index specifying the redshift 
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile_battaglia_AGN(self,rvals,zindex,mind):    
        m200, r200 = self.m200_c[mind,zindex],self.r200_c[mind,zindex]

        rho0 = 4000.*((m200/(10**(14)))**(0.29))*(1+self.z[zindex])**(-0.66)
        alpha = 0.88*((m200/(10**(14)))**(-0.03))*(1+self.z[zindex])**(0.19)
        beta = 3.83*((m200/(10**(14)))**(0.04))*(1+self.z[zindex])**(-0.025)
        xc = 0.5
        gamma = -0.2
        xxs = rvals/r200 
        ans = self.rhocritz[zindex]*rho0*((xxs/xc)**gamma)*( 1. +(xxs/xc)**alpha )**(-(beta+gamma)/alpha)
        return ans

    # Function to compute gas density profile from Battaglia (1607.02442) Shock Heating feedback model in units of M_\odot Mpc^-3, specifying
    # c = concentration
    # rv = virial radius of dark matter halo
    # mv = mass of dark matter halo
    # rvals = values of r to return profile at (in units of Mpc). 
    # zindex = index specifying the redshift 
    # Note: All distances are physical distances, not comoving distances.
    def gasprofile_battaglia_SH(self,c,rv,mv,rvals,zindex):    
        m200, r200 = self.mrvir_to_mrdelta(200,zindex,c,rv,mv)
        rho0 = 19000.*((m200/(10**(14)))**(0.09))*(1+self.z[zindex])**(-0.95)
        alpha = 0.70*((m200/(10**(14)))**(-0.017))*(1+self.z[zindex])**(0.27)
        beta = 4.43*((m200/(10**(14)))**(0.005))*(1+self.z[zindex])**(0.037)
        xc = 0.5
        gamma = -0.2
        xxs = rvals/r200
        ans = self.rhocritz[zindex]*rho0*((xxs/xc)**gamma)*( 1. +(xxs/xc)**alpha )**(-(beta+gamma)/alpha)
        return ans

    #------ Calculate the normalized FT of gas density profile as a function of mass and z

    def calc_gasprofiles(self, gasprofile,cutoffradius=None): #TODO (MM)
    
        if cm.exists(self.basic_conf_dir,  gasprofile , dir_base = 'hm_data'):
            print("Loading precomputed gas profile")
            return cm.load(self.basic_conf_dir, gasprofile, dir_base = 'hm_data')        
        else:
            print("Calculating gas profile (this can take between 30-60 min, but need to do only once for a given config)")
            start = time.time()
            if cutoffradius is None:
                cutoffradius = self.r200_c
            #!!!!!!!This takes up to 30 minutes, for convenience I've precomputed this in the files 'gasprofile_universal.npy'
            # 'gasprofile_AGN.npy' and 'gasprofile_SH.npy' for the parameters:            
            nzeds = len(self.z)
            nks   = len(self.k)         
          
            ukout = np.zeros((self.npts,nzeds,nks))
            
            if gasprofile in ['universal']:
    
                for j in range(0,self.z.shape[0]):
                    for i in range(0,self.logm.shape[0]):
                        c=self.conc[i,j]
                        rv=(cutoffradius[i,j])
                        rvals=np.linspace(0.0001,rv,1000)
                        rstar=rv
                        mv=10**self.logm[i]
                        mgas=4.*np.pi*np.trapz((rvals**2)*self.gasprofile(c,rv,mv,rvals,rstar),rvals)
                        for q in range(0,self.k.shape[0]):
                            kphys=self.k[q]*(1+self.z[j])
                            ukout[i,j,q] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys*rvals)/(kphys*rvals))*self.gasprofile(c,rv,mv,rvals,rstar)/mgas,rvals)
    
    
            if gasprofile in ['AGN']:
                for j in range(0,self.z.shape[0]):
    
                    for i in range(0,self.logm.shape[0]): 
                
                        rv=(cutoffradius[i,j])
                        rvals=np.linspace(0.0001,rv,1000)
    
                        gasp = self.gasprofile_battaglia_AGN(rvals,j,i)
                        mgas_AGN = 4.*np.pi*np.trapz((rvals**2)*gasp,rvals)
                        kphys=self.k*(1+self.z[j])
    
    
                        ukout[i,j,:] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys[:,np.newaxis]*rvals)/(kphys[:,np.newaxis]*rvals))*gasp/mgas_AGN,rvals,axis=1)
    
                
            if gasprofile in ['SH']:
                
                for j in range(0,self.z.shape[0]):
                    for i in range(0,self.logm.shape[0]):
                        c=self.conc[i,j]
                        #rv=(self.rvir3[i,j])**(0.3333333)
                        rv=(cutoffradius[i,j])
    
                        rvals=np.linspace(0.0001,rv,10000)
                        rstar=rv
                        mv=10**self.logm[i]
                        mgas_SH = 4.*np.pi*np.trapz((rvals**2)*self.gasprofile_battaglia_SH(c,rv,mv,rvals,j),rvals)
                        for q in range(0,self.k.shape[0]):
                            kphys=self.k[q]*(1+self.z[j])
                            ukout[i,j,q] = np.trapz(4.*np.pi*(rvals**2)*(np.sin(kphys*rvals)/(kphys*rvals))*self.gasprofile_battaglia_SH(c,rv,mv,rvals,j)/mgas_SH,rvals)
            
                
            cm.dump(self.basic_conf_dir, ukout, gasprofile, dir_base = 'hm_data')
            
            end = time.time()
            print("ukgas loop time:",end-start)
            return ukout

    # Calc yk 
    
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
        
             
        
    def pdelta(self,):
        Gnewt=4.518294724674995e-48 
        rhocritz = self.rhocritz
        return  Gnewt*self.m200_c*200*rhocritz*self.fb/(2*self.r200_c)
    
    
    def pressureprofile_x(self,x,z,m200,mind,zind):
        
        
        XH=.76 
        thermal_to_electron=2.0*(XH+1.0)/(5.0*XH+3.0)                            # thermal to electron pressure
        
        return thermal_to_electron * self.pdelta()[mind,zind] * self.gasprofile_battaglia_AGN_pressure_x(x,z,m200)
       
            
    def make_yk_phys(self,):
        
        if cm.exists(self.basic_conf_dir,  'yk_battaglia' , dir_base = 'hm_data'):
            print("Loading precomputed yk profile")
            return cm.load(self.basic_conf_dir, 'yk_battaglia', dir_base = 'hm_data')   
        else:
            print("Calculating yk profile (this can take between 30-60 min, but need to do only once for a given config)")
            start = time.time()
                
            xs=np.linspace(0,4,1000)[1:]
            
            nzeds = len(self.z)
            nks   = len(self.k)     
            
            m200 = self.m200_c 
            r200 = self.r200_c 
              
            integral = np.zeros((self.npts,nzeds,nks))
            
            for mind in range(0,self.npts):
                for zind in range(0,nzeds ):   # vectorizing doesn't speed this up
    
                    exsquaredpressure=xs**2*self.pressureprofile_x(xs[:],self.z[zind],m200[mind,zind],mind,zind)
    
                    kphys=self.k*(1+self.z[zind])
                    
                    sinkxoverkx1 = np.sin(kphys[:,np.newaxis]*xs[:]*r200[mind,zind])/(kphys[:,np.newaxis]*xs[:]*r200[mind,zind])
                    
                    integral[mind,zind,:] = np.trapz((r200[mind,zind]**3*sinkxoverkx1*exsquaredpressure[:]),xs,axis=1)
                    
            cm.dump(self.basic_conf_dir, integral, 'yk_battaglia', dir_base = 'hm_data')
            
            end = time.time()
            print("ukgas loop time:",end-start)
            return integral
    
   
    def MHI_halo(self,z,lnmhalo):
        
        # Mass of HI as a function of halo mass.
        
        # We use the MHI = M0*Mh^alpha*exp(-Mmin/Mh) relation (e.g. 1804.09180).
        # We interpolate the best-fit parameter values from 1804.09180 Table 6. to get the redshift evolution of this
        # relation
        # Note that Mmin values are divided by h, so that the units are [Msolar]
        
        z_array = np.array([0,1,2,3,4,5])
        alpha_array = np.array([0.49, 0.76, 0.8, 0.95, 0.94, 0.9])
        Mmin_array = np.array([5.2e10, 2.6e10, 2.1e10, 4.8e9, 2.1e9, 1.9e9])/self.conf.h
        
        Mhalo = np.exp(lnmhalo)
        
        MHI=np.ones([lnmhalo.shape[0],z.shape[0]])
        for j in range(0,z.shape[0]):
            Mmin = np.interp(z[j], z_array, Mmin_array)
            alpha = np.interp(z[j], z_array, alpha_array)
            for i in range(0,lnmhalo.shape[0]):
                MHI[i,j] = Mhalo[i]**alpha*np.exp(-Mmin/Mhalo[i])
            omega_HI_unnorm = np.trapz(np.exp(self.lnms)*MHI[:,j]*self.nfn[:,j],self.lnms)/self.rhocrit
            MHI[:,j] *= self.omega_HI(z[j])/omega_HI_unnorm
        return MHI
    
    
    def omega_HI(self,z):
        # Fractional density in HI, as defined using 
        # Replaced with Eq. B1 of 1810.09572, originally from 1506.02037
        return (4*1e-4*(1.+z)**0.6)

    def TbHI(self,z):
        # Brightness temperature of HI as a function of redshift. 
        # It defines the amplitude of the HI power spectrum, entering the P_HIHI as TbHI^2.
        # This formula is taken from 1810.09572 (Appendix B) and is given in units of [mK]:
        return 180.*self.omega_HI(z)*(1+z)**2*self.conf.h/np.sqrt(1-self.omegam + self.omegam*(1+z)**3)
            
    # Function to calculate the one-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # log10mlow = log10 lower mass bound 
    # log10mhigh = log10 upper mass bound
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    
    def _Ponehalo_binned(self,z,lnms,k,uk1,uk2,nfn,rhomatter,spec,log10mlow,log10mhigh):
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------  
        lnmlow=np.log(10.**log10mlow)
        lnmhigh=np.log(10.**log10mhigh)
        lnmsin=lnms[(lnms>=lnmlow)&(lnms<=lnmhigh)]
        uk1in=uk1[(lnms>=lnmlow)&(lnms<=lnmhigh),:,:]
        uk2in=uk2[(lnms>=lnmlow)&(lnms<=lnmhigh),:,:]
        nfnin=nfn[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalcin=Ngalc[(lnms>=lnmlow)&(lnms<=lnmhigh),:]
        Ngalsin=Ngals[(lnms>=lnmlow)&(lnms<=lnmhigh),:]

        onehalointegrand = np.zeros([lnmsin.shape[0],z.shape[0],k.shape[0]])

        if spec in ['mm','mgas','gasgas']:
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(3.*lnmsin[i])/(rhomatter**2)

        if spec in ['hgas']:
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(2.*lnmsin[i])/(rhomatter*self.nbar_halo(z,log10mlow,log10mhigh))         

        if spec in ['galgal']:
            Ntot=Ngalcin+Ngalsin
            nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ntot)
            onehalointegrand=np.zeros((lnms.shape[0],z.shape[0],k.shape[0]))
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if Ngalc[i,j]>10**(-16):
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:] + (Ngals[i,j]**2)*(uk1[i,j,:]**2.)/Ngalc[i,j])
                    else:
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:])
                        
                      
        if spec in ['galm','galgas']:
            Ntot=Ngalcin+Ngalsin
            nbargaltot=self.Nbargal(lnmsin,z,nfnin,Ntot)
            nbarc=self.Nbargal(lnmsin,z,nfnin,Ngalcin)
            nbars=self.Nbargal(lnmsin,z,nfnin,Ngalsin)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = (nfnin[i,j]*np.exp(2.*lnmsin[i])/(rhomatter*nbargaltot[j]))*(Ngalcin[i,j]*uk2in[i,j,:] + Ngalsin[i,j]*uk1in[i,j,:]*uk2in[i,j,:])

        if spec in ['HIHI']:
            rho_HI = self.rhocrit*self.omega_HI(z)
            MHI = self.MHI_halo(z,lnmsin)
            TbHIz = self.TbHI(z)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(lnmsin[i])*MHI[i,j]**2/rho_HI[j]**2*TbHIz[j]**2

        if spec in ['HIgas']:
            rho_HI = self.rhocrit*self.omega_HI(z)
            MHI = self.MHI_halo(z,lnmsin)
            TbHIz = self.TbHI(z)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin.shape[0]):
                    onehalointegrand[i,j,:] = uk1in[i,j,:]*uk2in[i,j,:]*nfnin[i,j]*np.exp(2.*lnmsin[i])*MHI[i,j]/rho_HI[j]/rhomatter*TbHIz[j]
        
        Ponehalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                if k[q]>.01:
                    Ponehalo[j,q] = integrate.simps(onehalointegrand[:,j,q],lnmsin)
                else:
                    Ponehalo[j,q] = 10.**(-16.)

        if self.onehdamping:
            Ponehalo *= (1.-np.exp(-(k/self.onehdamping_kstar)**2.))
        #if spec in ['galgal']:
         #   return Ponehalo,onehalointegrand
        return Ponehalo

    # Function to calculate the two-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # halobias = precomputed linear halo bias
    # log10mlow = log10 lower mass bound 
    # log10mhigh = log10 upper mass bound
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    # log10mlow2 = log10 lower mass bound for cross mass bin spectra
    # log10mhigh2 = log10 upper mass bound for cross mass bin spectra
    def _Ptwohalo_binned(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,log10mlow,log10mhigh,log10mlow2=None,log10mhigh2=None):
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------
        if (log10mlow2==None)or(log10mhigh2==None):
            log10mlow2 = log10mlow
            log10mhigh2 = log10mhigh
        
        #consistency=np.zeros([z.shape[0]])

        lnmlow1=np.log(10.**log10mlow)
        lnmhigh1=np.log(10.**log10mhigh)
        lnmsin1=lnms[(lnms>=lnmlow1)&(lnms<=lnmhigh1)]
        nfnin1=nfn[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        Ngalcin1=Ngalc[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        Ngalsin1=Ngals[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        halobiasin1=halobias[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:]
        uk1in=uk1[(lnms>=lnmlow1)&(lnms<=lnmhigh1),:,:]

        lnmlow2=np.log(10.**log10mlow2)
        lnmhigh2=np.log(10.**log10mhigh2)
        lnmsin2=lnms[(lnms>=lnmlow2)&(lnms<=lnmhigh2)]
        nfnin2=nfn[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        Ngalcin2=Ngalc[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        Ngalsin2=Ngals[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        halobiasin2=halobias[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:]
        uk2in=uk2[(lnms>=lnmlow2)&(lnms<=lnmhigh2),:,:]

        twohalointegrand1 = np.zeros([lnmsin1.shape[0],z.shape[0],k.shape[0]])
        twohalointegrand2 = np.zeros([lnmsin2.shape[0],z.shape[0],k.shape[0]])

        if spec in ['mm','mgas','gasgas']:    
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = uk1in[i,j,:]*halobiasin1[i,j]*nfnin1[i,j]*np.exp(2.*lnmsin1[i])/(rhomatter)
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(2.*lnmsin2[i])/(rhomatter)

        if spec in ['galgal']:
            Ntot1=Ngalcin1+Ngalsin1
            nbargaltot1=self.Nbargal(lnmsin1,z,nfnin1,Ntot1)
            Ntot2=Ngalcin2+Ngalsin2
            nbargaltot2=self.Nbargal(lnmsin2,z,nfnin2,Ntot2)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*(Ngalcin1[i,j]+Ngalsin1[i,j]*uk1in[i,j,:])/nbargaltot1[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = halobiasin2[i,j]*nfnin2[i,j]*np.exp(lnmsin2[i])*(Ngalcin2[i,j]+Ngalsin2[i,j]*uk2in[i,j,:])/nbargaltot2[j]

        if spec in ['galm','galgas']:
            Ntot1=Ngalcin1+Ngalsin1
            nbargaltot1=self.Nbargal(lnmsin1,z,nfnin1,Ntot1)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*(Ngalcin1[i,j]+Ngalsin1[i,j]*uk1in[i,j,:])/nbargaltot1[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(2.*lnmsin2[i])/(rhomatter)

        if spec in ['HIHI']:    
            rho_HI = self.rhocrit*self.omega_HI(z)
            MHI1 = self.MHI_halo(z,lnmsin1)
            MHI2 = self.MHI_halo(z,lnmsin2)
            TbHIz = self.TbHI(z)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = uk1in[i,j,:]*halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*MHI1[i,j]/rho_HI[j]*TbHIz[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(lnmsin2[i])*MHI2[i,j]/rho_HI[j]*TbHIz[j]

        if spec in ['HIgas']:
            rho_HI = self.rhocrit*self.omega_HI(z)
            MHI1 = self.MHI_halo(z,lnmsin1)
            TbHIz = self.TbHI(z)
            for j in range(0,z.shape[0]):
                for i in range(0,lnmsin1.shape[0]):
                    twohalointegrand1[i,j,:] = uk1in[i,j,:]*halobiasin1[i,j]*nfnin1[i,j]*np.exp(lnmsin1[i])*MHI1[i,j]/rho_HI[j]*TbHIz[j]
                for i in range(0,lnmsin2.shape[0]):
                    twohalointegrand2[i,j,:] = uk2in[i,j,:]*halobiasin2[i,j]*nfnin2[i,j]*np.exp(2.*lnmsin2[i])/(rhomatter)*TbHIz[j]
        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                Ptwohalo[j,q] = pk[j,q]*integrate.simps(twohalointegrand1[:,j,q],lnmsin1)*integrate.simps(twohalointegrand2[:,j,q],lnmsin2)
        return Ptwohalo

    #------------------------------
    #
    # if the 2-halo power spectrum is \int dM dn/dM b(M) A_1(M,z,k)*\int dM dn/dM b(M) A_2(M,z,k)*(Pk(k,z))
    # define the D_i(M,z,k)    here
    # 
    #------------------------------
    def Anu_CIB(self,frequency,ukm):
        #the CIB term at frequency frequency that appears in the 2-halo power spectrum
        central_flux = self.central_flux[str(frequency)]
        satflux = self.satflux[str(frequency)]
        twohalointegrand = (self.csm.chi_from_z(self.z)[np.newaxis,:,np.newaxis])**2*(central_flux[:,:,np.newaxis]+ukm*satflux[:,:,np.newaxis]) 
        return twohalointegrand
            
    def A_tSZ(self,ykm):
        # the tSZ term that appears in the 2-halo power spectrum
        c = self.csm.cspeed*1000
        tsz_factor=4*np.pi*(self.thompson_SI/(self.mElect*c**2))*(1+self.z)**2*self.conf.T_CMB
        twohalointegrand = tsz_factor[np.newaxis,:,np.newaxis]*ykm
        return twohalointegrand
            
    def A_gal(self,lnms,ukm):
        
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        
        twohalointegrand = (Ngalc[:,:,np.newaxis]+Ngals[:,:,np.newaxis]*ukm[:,:,:])/nbargaltot[np.newaxis,:,np.newaxis]
        return twohalointegrand
    def A_gas(self,lnms,ukgas,rhomatter):
        twohalointegrand = ukgas*np.exp(lnms[:,np.newaxis,np.newaxis])/(rhomatter) 
        return twohalointegrand
    # Function to calculate the one-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    # 'CIBCIB' = CIB-CIB
    # 'CIBgas' = CIB-gas
    # 'CIBgal' = CIB-galaxies
    # 'CIBtSZ' = CIB-tSZ
    # 'tSZtSZ' = tSZ-tSZ     
    # 'tSZgal' = tSZ-galaxies
    # 'tSZgas' = tSZ-gas
    
    def _Ponehalo(self,z,lnms,k,uk1,uk2,nfn,rhomatter,spec,frequency=None,frequency2=None):
        #frequency and frequency2 are CIB frequencies
        
        if  spec[:1] == 'm':
            spec1 = 'm'
        else:
            spec1 = spec[:3]
        if spec[-1] == 'm':
            spec2 = 'm'
        else:
            spec2=spec[-3:]
            
        def Akmz(spec,uk,frequency=None):
            if spec == 'm' or spec == 'gas':
                return self.A_gas(lnms,uk,rhomatter)
            if spec == 'CIB':
                return self.Anu_CIB(frequency,uk)
            if spec == 'tSZ':
                return self.A_tSZ(uk)
            if spec == 'gal':
                return self.A_gal(lnms,uk)
        
        
        #------- Number of satellite and halo galaxies
        
        if 'gal' in spec:
            Ngalc = np.zeros([self.npts,self.z.shape[0]])
            Ngals = np.zeros([self.npts,self.z.shape[0]])
            for j in range(0,self.z.shape[0]):
                Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
                Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
            nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)

        #-------  
            
        if spec not in ['galgal','CIBCIB','CIBgal']:
            onehalointegrand = nfn[:,:,np.newaxis]*np.exp(lnms[:,np.newaxis,np.newaxis])*Akmz(spec1,uk1,frequency)*Akmz(spec2,uk2,frequency2)
          
        elif spec in ['galgal']:
           # Ngalc[Ngalc<1e-16]=0
           # onehalointegrand = (nfn[:,:,np.newaxis]*np.exp(lnms[:,np.newaxis,np.newaxis])/(nbargaltot[np.newaxis,:,np.newaxis]**2))*(2.*Ngals[:,:,np.newaxis]*uk1+ (Ngals[:,:,np.newaxis]**2)*(uk1**2.)/Ngalc[:,:,np.newaxis])
            onehalointegrand=np.zeros((lnms.shape[0],z.shape[0],k.shape[0]))
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if Ngalc[i,j]>10**(-16):
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:] + (Ngals[i,j]**2)*(uk1[i,j,:]**2.)/Ngalc[i,j])
                    else:
                        onehalointegrand[i,j,:] = (nfn[i,j]*np.exp(lnms[i])/(nbargaltot[j]**2))*(2.*Ngals[i,j]*uk1[i,j,:])
                      
        elif spec in ['CIBCIB']:  
      
            central_flux = self.central_flux[str(frequency)]
            satflux = self.satflux[str(frequency)]
            central_flux2 = self.central_flux[str(frequency2)]
            satflux2 = self.satflux[str(frequency2)]
            onehalointegrand=(self.csm.chi_from_z(z)[np.newaxis,:,np.newaxis])**4*(np.exp(lnms[:,np.newaxis,np.newaxis])*
                                                                                     nfn[:,:,np.newaxis]*uk1[:,:,:]*((central_flux[:,:,np.newaxis]*satflux2[:,:,np.newaxis]+central_flux2[:,:,np.newaxis]*satflux[:,:,np.newaxis])+satflux[:,:,np.newaxis]*satflux2[:,:,np.newaxis]*uk1[:,:,:]) ) #check this
                   
        elif spec in ['CIBgal']:
            central_flux = self.central_flux[str(frequency)]
            satflux = self.satflux[str(frequency)]
            onehalointegrand = (self.csm.chi_from_z(z)[np.newaxis,:,np.newaxis])**2/nbargaltot[np.newaxis,:,np.newaxis]*nfn[:,:,np.newaxis]*uk1*(np.exp(lnms[:,np.newaxis,np.newaxis])*(central_flux[:,:,np.newaxis]*Ngals[:,:,np.newaxis]+satflux[:,:,np.newaxis]*Ngalc[:,:,np.newaxis]+satflux[:,:,np.newaxis]*Ngals[:,:,np.newaxis]*uk1 ) ) #check this
        
        Ponehalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                if k[q]>.01:
                    Ponehalo[j,q] = integrate.simps(onehalointegrand[:,j,q],lnms)
                else:
                    Ponehalo[j,q] = 10.**(-16.)

        if self.onehdamping:
            Ponehalo *= (1.-np.exp(-(k/self.onehdamping_kstar)**2.))                    
                
        return Ponehalo

    # Function to calculate the two-halo auto and cross power spectrum for matter, gas, galaxies.
    # z =  redshifts
    # lnms = natural log of halo masses
    # k = comoving wavenumbers
    # uk1 = FT of galaxy run (if galaxy part of correlation), otherwise mass or gas
    # uk2 = FT of mass or gas.
    # nfn = halo mass function
    # Ngalc = number of central galaxies as a function of halo mass and redshift
    # Ngals = number of satellite galaxies as a function of halo mass and redshift
    # rhomatter =  current density in matter
    # halobias = precomputed linear halo bias
    # spec = specifies the correlation function 
    # 'mm' = matter-matter
    # 'gasgas' = gas-gas
    # 'galgal' = galaxy-galaxy
    # 'mgas' = matter-gas
    # 'galm' = galaxy-matter
    # 'galgas' = galaxy-gas
    # 'CIBCIB' = CIB-CIB
    # 'CIBgas' = CIB-gas
    # 'CIBgal' = CIB-galaxies
    # 'CIBtSZ' = CIB-tSZ
    # 'tSZtSZ' = tSZ-tSZ     
    # 'tSZgal' = tSZ-galaxies
    # 'tSZgas' = tSZ-gas
    
    def _Ptwohalo(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,frequency = None,frequency2=None):
        #frequency and frequency2 are the CIB frequencies
        if  spec[:1] == 'm':
            spec1 = 'm'
        else:
            spec1 = spec[:3]
        if spec[-1] == 'm':
            spec2 = 'm'
        else:
            spec2=spec[-3:]
            
        #calculate consistency condition
        #TODO: check consistency function for observables other than mass and electrons.
        
        if 'gas' in spec or 'm' in spec:
            consistency=np.ones([z.shape[0]])
            if not self.use_counterterms:
                for j in range(0,z.shape[0]):
                    consistency[j] = integrate.simps(np.exp(2.*lnms)*nfn[:,j]*halobias[:,j]/(self.omegam*self.rhocrit),lnms)

        def Akmz(spec,uk,frequency=None):
            if spec == 'm' or spec == 'gas':
                return self.A_gas(lnms,uk,rhomatter)/consistency[np.newaxis,:,np.newaxis]
            if spec == 'CIB':
                return self.Anu_CIB(frequency,uk)
            if spec == 'tSZ':
                return self.A_tSZ(uk)
            if spec == 'gal':
                return self.A_gal(lnms,uk)
            
        twohalointegrand1 = np.exp(self.lnms[:,np.newaxis,np.newaxis])*self.nfn[:,:,np.newaxis]*self.halobias[:,:,np.newaxis]*Akmz(spec1,uk1,frequency)
                
        
        if self.use_counterterms:
                    
                    twohalointegrand1 += self.nmin[:,np.newaxis] * self.mmin_nfn * self.biasmin[:,np.newaxis] * Akmz(spec1,uk1,frequency)[self.mindex_mmin_nfn]

        
        if spec2 == spec1 and frequency == frequency2:
            twohalointegrand2 = twohalointegrand1
        else:
            twohalointegrand2 = np.exp(self.lnms[:,np.newaxis,np.newaxis])*self.nfn[:,:,np.newaxis]*self.halobias[:,:,np.newaxis]*Akmz(spec2,uk2,frequency2)
       
            if self.use_counterterms:

                    twohalointegrand2 += self.nmin[:,np.newaxis] * self.mmin_nfn * self.biasmin[:,np.newaxis] * Akmz(spec2,uk2,frequency)[self.mindex_mmin_nfn]
 
       
        
       
        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        
        
            
        
        
        #TODO: maybe turn into slicing instead of loop?
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                Ptwohalo[j,q] = pk[j,q]*integrate.simps(twohalointegrand1[:,j,q],lnms)*integrate.simps(twohalointegrand2[:,j,q],lnms)



        return Ptwohalo
    
    #version of P_ge where we want all electrons (no mass cut), but only the galaxies within the mass bin
    def _Ptwohalo_halobinned_gasunbinned(self,z,lnms,k,pk,uk1,uk2,nfn,rhomatter,halobias,spec,log10mlow,log10mhigh):
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral_binned(self.lnms,self.z[j],self.mthreshHOD[j],log10mlow,log10mhigh)
            Ngals[:,j]=self.Nsatellite_binned(self.lnms,self.z[j],self.mthreshHOD[j],log10mlow,log10mhigh)
        nbargaltot=self.Nbargal(self.lnms,self.z,self.nfn,Ngalc+Ngals)
        #-------    
        twohalointegrand1 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        twohalointegrand2 = np.zeros([self.npts,z.shape[0],k.shape[0]])
        consistency=np.ones([z.shape[0]])

        #for j in range(0,z.shape[0]):
         #   consistency[j] = integrate.simps(np.exp(2.*lnms)*nfn[:,j]*halobias[:,j]/(self.omegam*self.rhocrit),lnms)

        if spec in ['mgas']:    
            lnmlow=np.log(10.**log10mlow)
            lnmhigh=np.log(10.**log10mhigh)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if (lnms[i]>=lnmlow) and (lnms[i]<lnmhigh):
                        twohalointegrand1[i,j,:] = uk1[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j]) #uk2 is the gas

        if spec in ['hgas']: 
            lnmlow=np.log(10.**log10mlow)
            lnmhigh=np.log(10.**log10mhigh)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    if (lnms[i]>=lnmlow) and (lnms[i]<lnmhigh):
                        twohalointegrand1[i,j,:] = uk1[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(lnms[i])/self.nbar_halo(z,log10mlow,log10mhigh)  #this equals the binned bias
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j]) #uk2 is the gas            
                    
        if spec in ['galgas']:
            Ntot=Ngalc+Ngals
            nbargaltot=self.Nbargal(lnms,z,nfn,Ntot)
            for j in range(0,z.shape[0]):
                for i in range(0,lnms.shape[0]):
                    twohalointegrand1[i,j,:] = halobias[i,j]*nfn[i,j]*np.exp(lnms[i])*(Ngalc[i,j]+Ngals[i,j]*uk1[i,j,:])/nbargaltot[j]
                    twohalointegrand2[i,j,:] = uk2[i,j,:]*halobias[i,j]*nfn[i,j]*np.exp(2.*lnms[i])/(rhomatter*consistency[j])

        Ptwohalo = np.zeros([z.shape[0],k.shape[0]])
        for j in range(0,z.shape[0]):
            for q in range(0,k.shape[0]):
                Ptwohalo[j,q] = pk[j,q]*integrate.simps(twohalointegrand1[:,j,q],lnms)*integrate.simps(twohalointegrand2[:,j,q],lnms)

        return Ptwohalo

    ################################## convinient wrappers of the previous functions

    def P_HIHI_1h(self,ks,zs,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukHI,self.ukHI,self.nfn,self.rhomatter,'HIHI')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukHI,self.ukHI,self.nfn,self.rhomatter,'HIHI',logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo

    def P_HIHI_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukHI,self.ukHI,self.nfn,self.rhomatter,self.halobias,'HIHI')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukHI,self.ukHI,self.nfn,self.rhomatter,self.halobias,'HIHI',logmlow,logmhigh,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo

    def P_HIe_1h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self.ukgas
        if (logmlow==None) or (logmhigh==None):
            Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,self.ukHI,ukgas,self.nfn,self.rhomatter,'HIgas')
        else:
            Ponehalo = self._Ponehalo_binned(self.z,self.lnms,self.k,self.ukHI,ukgas,self.nfn,self.rhomatter,'HIgas',logmlow,logmhigh) 
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    def P_HIe_2h(self,ks,zs,gasprofile='universal',logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)
        ukgas = self.ukgas
        if (logmlow==None) or (logmhigh==None):
            Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,self.ukHI,ukgas,self.nfn,self.rhomatter,self.halobias,'HIgas')
        else:
            Ptwohalo = self._Ptwohalo_binned(self.z,self.lnms,self.k,self.pk,self.ukHI,self.ukHI,self.nfn,self.rhomatter,self.halobias,'HIgas',logmlow,logmhigh,logmlow2,logmhigh2)

        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))  
        return Ptwohalo

    #this is just the shot noise. 
    def P_hh_1h(self,ks,zs,logmlow=None,logmhigh=None):
        Phh1h = np.ones( (self.pk.shape) ) * 1./self.nbar_halo(zs,logmlow,logmhigh)
        Pinterp = interp1d(np.log10(self.k),np.log10(Phh1h),bounds_error=False,fill_value=0)
        Phh1h = np.power(10.0, Pinterp(np.log10(ks)))
        return Phh1h

    #set second mass value for cross spectra
    def P_hh_2h(self,ks,zs,logmlow=None,logmhigh=None,logmlow2=None,logmhigh2=None):
        if logmlow2 == None:
            logmlow2 = logmlow
            logmhigh2 = logmhigh
        Phh2h = self.pk * self.bias_halo(zs,logmlow,logmhigh) * self.bias_halo(zs,logmlow2,logmhigh2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Phh2h),bounds_error=False,fill_value=0)
        Phh2h = np.power(10.0, Pinterp(np.log10(ks)))
        return Phh2h 
    def get_ukm(self,spec,gasprofile="universal",A=1):
        if spec in ["gal","CIB","m"]:
            return self.ukm
        if spec in ["tSZ"]:
            return self.ykm
        if spec in ["gas"]:
            ukgas = self.ukgas
            if A !=1:
                ukgas = self.rescale_electronprofile(A)
            return ukgas
    def P_1h(self,spec1,spec2,ks,zs,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault,frequency=None,frequency2=None,gasprofile='universal',A=1):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        
        ukm1 = self.get_ukm(spec1,gasprofile=gasprofile,A=A)

        ukm2 = self.get_ukm(spec2,gasprofile=gasprofile,A=A)
        Ponehalo = self._Ponehalo(self.z,self.lnms,self.k,ukm1,ukm2,self.nfn,self.rhomatter,spec1+spec2,frequency,frequency2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ponehalo),bounds_error=False,fill_value=0)
        Ponehalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ponehalo
    
    def P_2h(self,spec1,spec2,ks,zs,logmlow=None,logmhigh=None,mthreshHOD=mthreshHODdefault,frequency=None,frequency2=None,gasprofile='universal',A=1):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        
        ukm1 = self.get_ukm(spec1,gasprofile=gasprofile,A=A)
        ukm2 = self.get_ukm(spec2,gasprofile=gasprofile,A=A)
        Ptwohalo = self._Ptwohalo(self.z,self.lnms,self.k,self.pk,ukm1,ukm2,self.nfn,self.rhomatter,self.halobias,spec1+spec2,frequency,frequency2)
        Pinterp = interp1d(np.log10(self.k),np.log10(Ptwohalo),bounds_error=False,fill_value=0)
        Ptwohalo = np.power(10.0, Pinterp(np.log10(ks)))
        return Ptwohalo
    

    def bias_halo(self,zs,logmlow,logmhigh):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)            
        lnmlow=np.log(10.**logmlow)
        lnmhigh=np.log(10.**logmhigh)
        lnmsin=self.lnms[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh)]
        nfnin=self.nfn[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        halobiasin=self.halobias[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        bhalo = np.zeros([self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            bhalo[j]=integrate.simps((np.exp(lnmsin))*nfnin[:,j]*halobiasin[:,j],lnmsin)/integrate.simps((np.exp(lnmsin))*nfnin[:,j],lnmsin) #as eq 49 
        return bhalo
    
    def bias_galaxy(self,zs,logmlow,logmhigh,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        #-------           
        return self._galaxybias(self.lnms,zs,self.nfn,Ngalc,Ngals,self.halobias,logmlow,logmhigh)
    def nbar_halo(self,zs,logmlow,logmhigh):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs)        
        lnmlow=np.log(10.**logmlow)
        lnmhigh=np.log(10.**logmhigh)
        lnmsin=self.lnms[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh)]
        nfnin=self.nfn[(self.lnms>=lnmlow)&(self.lnms<=lnmhigh),:]
        nhalobinned=np.zeros(self.z.shape[0])
        for j in range(0,self.z.shape[0]):
            nhalobinned[j]=integrate.simps(np.exp(lnmsin)*nfnin[:,j],lnmsin)
        return nhalobinned
    def nbar_galaxy(self,zs,logmlow,logmhigh,mthreshHOD=mthreshHODdefault):
        zs = np.atleast_1d(zs)
        zs[zs<0.001] = 0.001
        self._setup_k_z_m_grid_functions(zs,mthreshHOD)
        #------- Number of satellite and halo galaxies
        Ngalc = np.zeros([self.npts,self.z.shape[0]])
        Ngals = np.zeros([self.npts,self.z.shape[0]])
        for j in range(0,self.z.shape[0]):
            Ngalc[:,j]=self.Ncentral(self.lnms,self.z[j],self.mthreshHOD[j])
            Ngals[:,j]=self.Nsatellite(self.lnms,self.z[j],self.mthreshHOD[j])
        #-------          
        return self.Nbargal_binned(self.lnms,zs,self.nfn,Ngalc+Ngals,logmlow,logmhigh)

    def convert_n_mpc3_arcmin2(self,n_mpc3,z):
        dz = 0.01
        zmax = z+dz/2.
        zmin = z-dz/2.
        chimax = self.csm.chi_from_z(zmax)
        chimin = self.csm.chi_from_z(zmin)
        dV_shell_comov = 4./3. * np.pi * ((chimax)**3. - (chimin)**3.)
        dV_dz = dV_shell_comov/dz
        dV_dZdOmega = dV_dz/allsky_arcmin2
        n_arcmin2 = n_mpc3 * dV_dZdOmega
        return n_arcmin2
    
    
    #convert dN/dOmega/dz to dN/dV. n_arcmin and z must be same length and matching z. 
    def convert_n_arcmin2_mpc3(self,n_arcmin2,z):
        dz = 0.01
        zmax = z+dz/2.
        zmin = z-dz/2.
        chimax = self.csm.chi_from_z(zmax)
        chimin = self.csm.chi_from_z(zmin)
        dV_shell_comov = 4./3. * np.pi * ((chimax)**3. - (chimin)**3.)
        dV_dz = dV_shell_comov/dz
        dV_dZdOmega = dV_dz/allsky_arcmin2
        n_mpc3 = n_arcmin2 / dV_dZdOmega
        return n_mpc3


    def CIB_Snu(self):
        
        return 1.


