import numpy as np
import healpy as hp


import time


def filter_map(Map,Filter,nside):
    #returns map(alm(map)/filter) where filter is a filter in l-space
    alm = hp.map2alm(Map)
    almbar = hp.almxfl(alm,Filter)
    return hp.alm2map(almbar,nside)


def reconstruct_rml(Tfield,deltafield,clpT,nside ,nsideout,nbin,dobins='all'):
    #clpT should be primary CMB power
    t1=time.time()
    clpTinv = np.zeros(clpT.shape)
    clpTinv[101:] = 1./clpT[101:]
    Tbarfield = filter_map(Tfield,clpTinv,nside)
    print("filtered in",time.time()-t1,flush=True)
    npix = hp.nside2npix(nsideout)
    npix2 = hp.nside2npix(nside)

    tb = time.time()
    vestimates = np.zeros((nbin,npix))
    vdgraded = np.zeros((nbin,npix))

  #  deltav = np.zeros(vtrue.shape)

    numerator=np.zeros((nbin,npix))
    operator=np.zeros((nbin,nbin,npix))
    opinv=np.zeros((nbin,nbin,npix))

    deltabarfield=np.zeros((nbin,npix2))

    difference = np.zeros((nbin,npix))

    betaop=np.zeros((nbin,npix))
    t1=time.time()
    for binid in np.arange(nbin):
        t1=time.time()
        numerator[binid]=hp.ud_grade(Tbarfield*deltafield[binid],nsideout)
   # print("got numeraotr in",time.time()-t1)
    if 1==1:
        t1=time.time()
        for binid in np.arange(nbin):
           deltabarfield[binid] = filter_map(deltafield[binid],clpTinv,nside)
   #        deltav[binid] = hp.ud_grade(vdgraded[binid],nside)-vtrue[binid]
   #     print("got deltabarfield and deltav in",time.time()-t1)
        t1=time.time()
        for binid in np.arange(nbin):
            for binid2 in np.arange(nbin):

                operator[binid,binid2]=hp.ud_grade(deltafield[binid2]*deltabarfield[binid],nsideout)
    #            betaop[binid] += hp.ud_grade(deltafield[binid]*deltabarfield[binid2]*deltav[binid2],nsideout)
   #     print("got operator and betaop in",time.time()-t1)
        t1=time.time()
        for pix in np.arange(0,npix):

            opinv[:,:,pix] = np.linalg.inv(operator[:,:,pix])
     #       for binid in np.arange(nbin):

      #          difference[binid,pix] = np.matmul( opinv[binid,:,pix],betaop[:,pix])
 ###       print("got inverse operator and difference in",time.time()-t1)
    t1=time.time()
    if dobins=='all':
        bins = np.arange(nbin)
    else:
        bins = dobins
    for binid in bins:
        t1 = time.time()
        for pix in np.arange(0,npix):

            vestimates[binid,pix] = np.matmul(opinv[:,binid,pix],numerator[:,pix])
    print("estimate in",time.time()-tb)
    return vestimates



