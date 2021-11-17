
import numpy as np
from scipy.interpolate import interp1d

def log_interpolate(A_sample, ell_sparse):
    #log interpolate 
    
    ls = np.arange(0,ell_sparse[-1]+1)
    
    A = np.zeros(len(ls))
    
    if np.max(np.abs(A_sample)) != 0.0 : 
        
        inz = A_sample.nonzero()[0]
        lcut = ell_sparse[inz[-1]]
        cut = np.where(ls>lcut,0,1)
        
        m = np.min(A_sample)  
        if m > 0.0:
            Y = 0.0
        if m < 0.0:
            Y = np.abs(m)+np.abs(m)*1e-5
        if m == 0.0:
            Y = np.min(A_sample[inz])/1e10
            
        I = interp1d(np.log10(ell_sparse),np.log10(A_sample+Y), kind= 'linear')
        A[ell_sparse[0]:] = (np.power(10.0, I(np.log10(ls[ell_sparse[0]:])))-Y)*cut[ell_sparse[0]:]
                    
    return A
        

def log_interpolate_vector(A_sample, ell_sparse):
    #log interpolate 
    
    ls = np.arange(0,ell_sparse[-1]+1)
    N = len(A_sample[0,:])
    
    A_matrix = np.zeros((len(ls),N))
    
    for i in np.arange(N):
    
        A = np.zeros(len(ls))
        
        if np.max(np.abs(A_sample[:,i])) != 0.0 : 
            
            inz = A_sample[:,i].nonzero()[0]
            lcut = ell_sparse[inz[-1]]
            cut = np.where(ls>lcut,0,1)
            
            m = np.min(A_sample[:,i])  
            if m > 0.0:
                Y = 0.0
            if m < 0.0:
                Y = np.abs(m)+np.abs(m)*1e-5
            if m == 0.0:
                Y = np.min(A_sample[inz,i])/1e10
                
            I = interp1d(np.log10(ell_sparse),np.log10(A_sample[:,i]+Y), kind= 'linear')
            A[ell_sparse[0]:] = (np.power(10.0, I(np.log10(ls[ell_sparse[0]:])))-Y)*cut[ell_sparse[0]:]
            
            A_matrix[:,i] = A
                    
    return A_matrix

def log_interpolate_matrix(A_sample, ell_sparse):
    #log interpolate 
    
    ls = np.arange(0,ell_sparse[-1]+1)
    N1 = len(A_sample[0,:,0])
    N2 = len(A_sample[0,0,:])
    
    A_matrix = np.zeros((len(ls),N1,N2))
    
    for i in np.arange(N1):
        for j in np.arange(N2):
    
            A = np.zeros(len(ls))
            
            if np.max(np.abs(A_sample[:,i,j])) != 0.0 : 
                
                inz = A_sample[:,i,j].nonzero()[0]
                lcut = ell_sparse[inz[-1]]
                cut = np.where(ls>lcut,0,1)
                
                m = np.min(A_sample[:,i,j])  
                if m > 0.0:
                    Y = 0.0
                if m < 0.0:
                    Y = np.abs(m)+np.abs(m)*1e-5
                if m == 0.0:
                    Y = np.min(A_sample[inz,i,j])/1e10
                    
                I = interp1d(np.log10(ell_sparse),np.log10(A_sample[:,i,j]+Y), kind= 'linear')
                A[ell_sparse[0]:] = (np.power(10.0, I(np.log10(ls[ell_sparse[0]:])))-Y)*cut[ell_sparse[0]:]
                
                A_matrix[:,i,j] = A
                    
    return A_matrix