import numpy as np
import healpy as hp
import config as conf

default_websky_folder = '/mnt/ceph/users/fmccarthy/kSZ/Websky_sims/'
default_astropaint_folder = default_websky_folder + 'AstroPaint/'
def ksz(nside,spec='websky',websky_maps_folder=default_websky_folder,astropaint_folder=default_astropaint_folder):
    if spec=='websky':
        return hp.ud_grade(hp.fitsfunc.read_map(websky_maps_folder + 'ksz.fits',verbose=False)/(conf.T_CMB),nside)
    if spec=='astropaint':
       chis = np.array([  43.98392527,  990.85202183, 1937.72011839, 2884.58821494,
       3831.4563115 , 4778.32440805, 5725.19250461, 6672.06060116,
       7618.92869772])
       kszmap = np.zeros(hp.nside2npix(4096))
       for x in range(0,8):
           chimin= chis[x]
           chimax = chis[x+1]
           kszmap += hp.fitsfunc.read_map(astropaint_folder+'ksz_chimin'+str(int(chimin))+'_chimax'+str(int(chimax))+'.fits',verbose=False)/(conf.T_CMB /1e6)
       return hp.ud_grade(kszmap,nside)
    
def CMB(nside,spec='websky',websky_maps_folder=default_websky_folder,alm=False):
    CMB_alm = hp.fitsfunc.read_alm(websky_maps_folder+'lensed_alm.fits')/(conf.T_CMB)
    if alm:
        return CMB_alm
    else:
        return hp.alm2map(CMB_alm,nside,verbose=False)

def Nhalos(nside,nbins,chiboundaries,websky_maps_folder=default_websky_folder):
    assert len(chiboundaries)==nbins+1
    Nhalos = np.zeros((nbins,hp.nside2npix(nside)))
    for x in range(0,nbins):   
        chimin = chiboundaries[x]
        chimax = chiboundaries[x+1]
        Nhalos[x] = hp.ud_grade(hp.fitsfunc.read_map(websky_maps_folder + 'N_chimin'+str(int(chimin))+'_chimax'+str(int(chimax))+'_NSIDE2048.fits',verbose=False),nside)
    return Nhalos

def tau(nside,nbins,chiboundaries,astropaint_folder = default_astropaint_folder):
    assert len(chiboundaries)==nbins+1
    tau = np.zeros((nbins,hp.nside2npix(nside)))
    for x in range(0,nbins):
        chimin = chiboundaries[x]
        chimax = chiboundaries[x+1]
        tau[x] = hp.ud_grade(hp.fitsfunc.read_map(astropaint_folder + 'tau_chimin'+str(int(chimin))+'_chimax'+str(int(chimax))+'.fits',verbose=False),nside)/(conf.T_CMB/1e6)
    return tau


def vmaps(nside,nbins,chiboundaries,websky_folder = default_websky_folder):
    assert nside<=32
    assert len(chiboundaries) == nbins+1
    vmaps = np.zeros((nbins,hp.nside2npix(nside)))
    halocounts = np.zeros((nbins,hp.nside2npix(nside)))
    for x in range(0,nbins):
         chimin = chiboundaries[x]
         chimax = chiboundaries[x+1]
         vmaps[x] = hp.fitsfunc.read_map(websky_folder + 'vrad_chimin'+str(int(chimin))+'_chimax'+str(int(chimax))+'_NSIDE'+str(32)+'.fits',verbose=False)
         halocounts[x] = hp.fitsfunc.read_map(websky_folder + 'N_chimin'+str(int(chimin))+'_chimax'+str(int(chimax))+'_NSIDE'+str(32)+'.fits',verbose=False)
         vmaps[x] /= halocounts[x]
         vmaps[x][halocounts[x]==0]=0

    return hp.ud_grade(vmaps,nside)
def get_power(fields1,fields2=None,diagonal=False,lmax=None):
    if fields2 is None:
        fields2 = fields1
    nside = hp.get_nside(fields1[0])
    if lmax is None:
         lmax = 3*nside
    power = np.zeros((lmax+1,len(fields1),len(fields2)))
    
    for x in range(len(fields1)):
        if diagonal:
            power[:,x,x] = hp.anafast(fields1[x],fields2[x],lmax=lmax)
        else:
            if len(fields1)==len(fields2):
                for y in range(x+1):
                    power[:,x,y]=power[:,y,x] =  hp.anafast(fields1[x],fields2[y],lmax=lmax)
                else:
                    for y in range(len(fields2)):
                        power[:,x,y] = hp.anafast(fields1[x],fields2[y],lmax=lmax)
        
    return power
        







