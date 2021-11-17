from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy import special
import scipy.integrate as integrate
import kszpsz_config as conf
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
import scipy.optimize as optimize
import master_lf_photometric as bs


######################################################################################################
################   HUBBLE PARAMETER, DENSITIES, COMOVING DISTANCE , ETC     ###################
######################################################################################################


# Auxiliary scale factor grid
ag = np.logspace(np.log10(conf.adec), 0, conf.transfer_integrand_sampling)

def az(z):
    """Scale factor at a given redshift"""
    return 1.0 / (1.0 + z)


def aeq(Omega_b, Omega_c, h, Omega_r_h2=conf.Omega_r_h2):
    """Scale factor at matter radiation equality"""
    return Omega_r_h2 / (Omega_b + Omega_c) / h**2


def k_sampling(config=conf):
    return np.logspace(config.k_min, config.k_max, config.k_res)

def L_sampling(config=conf):
    return np.arange(config.ksz_estim_signal_lmax)


# Modification to dark energy equation of state (only one model for now)
def fde(a):
    return 1.0 - a


def fde_da(a):
    return -1.0


def H0(h):
    """Hubble parameter today in Mpc**-1"""
    return 100.0 * h / (3.0 * 1.0e5)


def Omega_L(Omega_b, Omega_c, Omega_K, h, Omega_r_h2=conf.Omega_r_h2):
    """Omega_L in terms of Omega_b, Omega_c, and K imposed by consistency"""
    return 1 - Omega_K - (Omega_b + Omega_c) - Omega_r_h2/h**2


def E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=conf.Omega_r_h2):
    """Reduced Hubble parameter, H/H_0"""
    exp_DE = 3.0*(1.0 + w + wa*fde(a))
    E2 = (Omega_b + Omega_c) / a**3 + Omega_K / a**2 \
        + Omega_L(Omega_b, Omega_c, Omega_K, h) / a**exp_DE \
        + Omega_r_h2/h**2  / a**4
    return np.sqrt(E2)


def H(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Hubble parameter given a & cosmological parameters"""
    Ea = E(a, Omega_b, Omega_c, w, wa, Omega_K, h)
    return Ea * H0(h)


def H_config(a, config):
    """Hubble parameter given a & config settings"""
    return H(a, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def dEda(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=conf.Omega_r_h2) :
    """
    Derivative of the reduced Hubble parameter respect to the scale factor
    """
    exp_DE = 3.0*(1.0 + w + wa*fde(a))
    d = -3.0*(Omega_b + Omega_c) / a**4 - 2.0*Omega_K / a**3 \
        + (-3.0*wa*a*np.log(a) * fde_da(a) - exp_DE) * Omega_L(Omega_b, Omega_c, Omega_K, h) / a**(exp_DE + 1) \
        - 4.0*Omega_r_h2/h**2 / a**5
    derv_E = d / (2.0 * E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=Omega_r_h2))
    return derv_E


def dHdt(a, Omega_b, Omega_c, w, wa, Omega_K, h) :
    """(non-conformal) time-derivative of hubble parameter"""
    return a*H(a, Omega_b, Omega_c, w, wa, Omega_K, h)*H0(h) \
        * dEda(a, Omega_b, Omega_c, w, wa, Omega_K, h)


def dHdt_config(a, config) :
    """(non-conformal) time-derivative of hubble parameter"""
    return dHdt(a, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def sigma_nez(z, Omega_b, h):
    """Electron density as a function of redshift times the Thompson cross section"""
    mProton_SI = 1.673e-27
    G_SI = 6.674e-11
    H0 = 100 * h * 1e3  # Hubble constant in m s^-1 Mpc^-1
    MegaparsecTometer = 3.086e22
    thompson_SI = 6.6524e-29

    sigma_ne = thompson_SI * (3 * 0.88 / 8. / np.pi / MegaparsecTometer) * \
        (H0**2) * (1. / mProton_SI / G_SI) * Omega_b * (1 + z)**3

    return sigma_ne


def tau_z(z, Omega_b, h):
    """Optical depth at a given redshift"""
    chi = chifromz(z)
    chi_grid = np.linspace(0, chi, 100)
    z_grid = zfromchi(chi_grid)
    ae = az(z_grid)
    sigma_ne = sigma_nez(z_grid, Omega_b, h)

    integrand = ae * sigma_ne

    tau = integrate.simps(integrand, chi_grid)

    return tau


def tau_grid(Chi_grid, Z_grid, Omega_b, h):
    """
    Optical depth as a function of redshift
    Assumes Z_grid starts at z = 0.0!
    """
    ae = az(Z_grid)
    sigma_ne = sigma_nez(Z_grid, Omega_b, h)
    integrand = ae * sigma_ne
    tau_grid = integrate.cumtrapz(integrand, Chi_grid, initial=0.0)

    return tau_grid


def z_re(Omega_b, h, tau):
    """ redshift of recombination (?) """
    zguess = 6.0
    sol = optimize.root(root_tau2z, zguess, args=(Omega_b, h, tau))
    z_re = sol.x
    return z_re


def spherical_jn_pp( l, z ):
    """
    Second derivative of spherical Bessel function.
    """
    if l == 0:
        jn = special.spherical_jn(2, z) - special.spherical_jn(1, z)/z
    else:
        jn = (special.spherical_jn(l-1,z, True)
              - (l+1)/z * special.spherical_jn(l,z, True)
              + (l+1)/(z**2) * special.spherical_jn(l,z))
    return jn



##########################################################################
################   COMOVING DISTANCES AND THEIR DERIVATIVES     ##########
##########################################################################

def Integrand_chi(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integrand of the comoving distance defined below (chia, small c)"""
    int_chi = 1 / ((a**2) * H(a, Omega_b, Omega_c, w, wa, Omega_K, h))
    return int_chi


def chia(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Comoving distance to a (using integrate.quad)"""
    chi = integrate.quad(Integrand_chi, a, 1, args=(
        Omega_b, Omega_c, w, wa, Omega_K, h, ))[0]
    return chi


def Integrand_Chi(y, a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """ Function to integrate to find chi(a) defined below (Chia, capital C) """
    g = -((a**2) * H(a, Omega_b, Omega_c, w, wa, Omega_K, h))**-1
    return g

def Chia(Omega_b, Omega_c, w, wa, Omega_K, h):
    """Comoving distance to a (using integrate.odeint)"""
    Integral = (integrate.odeint(Integrand_Chi, chia(ag[0], Omega_b, Omega_c, w, wa, Omega_K, h),
                                 ag, args=(Omega_b, Omega_c, w, wa, Omega_K, h,))).reshape(len(ag),)

    if ag[-1] == 1.0:
        Integral[-1] = 0.0

    return Integral


def chifromz(z, config=conf):
    """Comoving distance at a redshift z"""
    chi = Chia_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)(az(z))
    return chi


def zfromchi(chi, config=conf):
    """Get redshift z given comoving distance chi"""
    afromchi = interp1d(
        Chia(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h), ag,
        kind="cubic", bounds_error=False, fill_value='extrapolate')

    aguess = afromchi(chi)

    sol = optimize.root(root_chi2a, aguess,
        args=(chi, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h))
    z = 1.0/sol.x - 1.0

    return z



#############################################################################
################                                          ###################
################          GROWTH FUNCTIONS                ###################
################                                          ###################
#############################################################################


def Integrand_GF(s, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integrand for the growth function (as defined below)
    """
    Integrand = 1 / (s * E(s, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0))**3
    return Integrand


def Integral_GF(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integral appearing in the growth factor GF
    """
    Integral = integrate.quad(Integrand_GF, 0, a, args=(
        Omega_b, Omega_c, w, wa, Omega_K, h, ))[0]
    return Integral


def Integrand_GF_ODE(y, a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Function for solving Integral_GF as an ODE (as defined below)
    """
    f = (a * E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0))**-3
    return f*1.0e13 # 10^13 changes scale so ODE has enough precision


def Integral_GF_ODE(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integral appearing in the growth function GF solved as an ODE
    """
    Integral = integrate.odeint(Integrand_GF_ODE,
                                1.0e13*Integral_GF(ag[0], Omega_b, Omega_c,
                                            w, wa, Omega_K, h),
                                ag, args=(Omega_b, Omega_c, w, wa, Omega_K, h,))

    return Integral * 1.0e-13 # 10^13 changes scale so ODE has enough precision


def GF(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Growth function using ODE, D_1(ag) / ag
    Dodelson Eq. 7.5, 7.77
    """
    l = 5.0/2.0 * (Omega_b + Omega_c) * \
        E(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0) / ag
    GF = l * Integral_GF_ODE(Omega_b, Omega_c, w, wa,
                             Omega_K, h).reshape(len(ag),)
    return GF


def Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    ~ Potential "growth function" from linear theory plus approximations
    Phi = Phi_prim * T(k) * (9/10 * D_1(a)/a) per Dodelson Eq. 7.5
    Dpsi is ("9/10" * D_1(a)/a)
    Dodelson Eq. 7.5, Eq. 7.32
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
    Dpsi = fancy_9_10 * GF(Omega_b, Omega_c, w, wa, Omega_K, h)
    return Dpsi


def derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Derivative of the growth function with respect to the scale factor
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    Eag = E(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0)
    dEdag = dEda(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0)
    fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
    P1 = ((8.0/np.sqrt(1.0 + y) + 27.0*y**2 + 4.0*y - 8.0) / (aeq(Omega_b,
      Omega_c, h)*10.0*y**3)) * GF(Omega_b, Omega_c, w, wa, Omega_K, h)
    P2 = Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h) * (-4/ag + dEdag/Eag)
    P3 = fancy_9_10 * (5.0/2.0) * (Omega_b + Omega_c) / (ag**4 * Eag**2)

    derv_Dpsi = P1 + P2 + P3
    return derv_Dpsi


def Dv(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Velocity growth function on superhorizon scales
    Dodelson 7.15 minus 7.16, v is 5.78
    v_i ~ - Dv d_i psi
    grad*v_i ~ k^2 * Dv * psi
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    Dv = 2.0 * (ag**2) * H(ag, Omega_b, Omega_c, w, wa, Omega_K, h) \
                / ((Omega_b + Omega_c)*H0(h)**2) * y / (4.0 + 3.0*y) \
            * (Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                + ag*derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h))
    return Dv


def T(k, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    'Best fitting' transfer function From Eq. 7.70 in Dodelson
    Assumes: no baryons, nonlinear effects, phi=psi always (9/10 -> 0.86), normal (no?) DE effects
    """
    fac = np.exp(-Omega_b*(1+np.sqrt(2*h)/(Omega_b+Omega_c)))
    keq = aeq(Omega_b, Omega_c, h) * H(aeq(Omega_b, Omega_c, h),
                                       Omega_b, Omega_c, w, wa, Omega_K, h)
    x = k / keq / fac
    x[np.where(x<1.0e-10)] = 1
    T = (np.log(1 + 0.171 * x) / (0.171 * x)) * (1 + 0.284 * x
      + (1.18 * x)**2 + (0.399 * x)**3 + (0.49 * x)**4)**(-0.25)
    return T


def T_config(k, config):
    return T(k, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


# TODO?
# Replace T*Dpsi (or other growth*transfer instances) with CAMB ones?
# Just to study the impact of BAO, other assumptions implicit in T(k)
# def TDpsi_CAMB(k, Omega_b, Omega_c, w, wa, Omega_K, h) :


def Ppsi(k, As, ns):
    """
    Power spectrum of primordial potential
    """
    P = (2.0/3.0)**2 * 2.0 * np.pi**2 / (k**3) \
        * As * 10**-9 * (k / conf.k0)**(ns - 1)
    return P


##########################################################################
################                                          ################
################ INTERPOLATING AND AUXILIARY FUNCTIONS    ################
################                                          ################
##########################################################################

def Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    Chia_inter = interp1d(ag, Chia(Omega_b, Omega_c, w, wa, Omega_K, h),
                          kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Chia_inter

def Chia_inter_config(config):
    return Chia_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    Dpsi_inter = interp1d(ag, Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h),
                          kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Dpsi_inter

def Dpsi_inter_config(config):
    return Dpsi_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Returns interpolating function
    of derivative of the growth function with respect to the scale factor
    """
    derv_Dpsi_inter = interp1d(ag, derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h),
                               kind="cubic", bounds_error=False, fill_value='extrapolate')
    return derv_Dpsi_inter

def derv_Dpsi_inter_config(config):
    return derv_Dpsi_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    """Returns interpolating funcntion of velocity growth function"""
    Dv_inter = interp1d(ag, Dv(Omega_b, Omega_c, w, wa, Omega_K, h),
                        kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Dv_inter

def Dv_inter_config(config):
    return Dv_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def root_chi2a(a, chis, Omega_b, Omega_c, w, wa, Omega_K, h):
    """ Needed to use the root function of scipy's optimize module. """
    return Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a) - chis


def root_tau2z(z, Omega_b, h, tau):
    """Needed to use the root function of scipy's optimize module below."""
    return tau_z(z, Omega_b, h) - tau



#######################################################################
################                                    ###################
################ KERNELS FOR SW, ISW AND DOPPLER    ###################
################                                    ###################
#######################################################################



def G_SW_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_SW_ksz = 3 * (2 * Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                    [0] - 3 / 2) * special.spherical_jn(1, k * (chidec - chie))

    return G_SW_ksz


def G_SW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """pSZ Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_SW_psz = -4 * np.pi * (2 * Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                             [0] - 3 / 2) * special.spherical_jn(2, k * (chidec - chie))

    return G_SW_psz


def G_SW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    G_SW_CMB = 4 * np.pi * ((1j)**l) * (2 * Dpsi(Omega_b, Omega_c, w,
        wa, Omega_K, h)[0] - 3 / 2) * special.spherical_jn(l, k * chidec)

    return G_SW_CMB


def G_Dopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_Dopp_ksz = k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[0] * (special.spherical_jn(0, k * (
        chidec - chie)) - 2 * special.spherical_jn(2, k * (chidec - chie))) - k * Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    return G_Dopp_ksz


def G_localDopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for the local Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_localDopp_ksz = -k * \
        Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    return G_localDopp_ksz


def G_Dopp_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """pSZ Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_Dopp_psz = (4 * np.pi / 5) * k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[0] * (
        3 * special.spherical_jn(3, k * (chidec - chie)) - 2 * special.spherical_jn(1, k * (chidec - chie)))

    return G_Dopp_psz


def G_Dopp_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    G_Dopp_CMB = (4 * np.pi / (2.0 * l + 1.0)) * (1j**l) * k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[
        0] * (l * special.spherical_jn(l - 1, k * chidec) - (l + 1) * special.spherical_jn(l + 1, k * chidec))

    return G_Dopp_CMB


def G_ISW_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the ksz ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(az(ze)), conf.transfer_integrand_sampling)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    Deltachi = Chia - Chie
    Deltachi[-1] = 0
    s2 = k[..., np.newaxis] * Deltachi

    integrand = special.spherical_jn(1, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_ksz = 6.0*integrate.simps(integrand, a)

    return g_isw_ksz


def G_ISW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the psz ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(az(ze)), 1000)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    Deltachi = Chia - Chie
    Deltachi[-1] = 0

    s2 = k[..., np.newaxis] * Deltachi
    integrand = special.spherical_jn(
        2, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_psz = -8 * np.pi * integrate.simps(integrand, a)

    return g_isw_psz


def G_ISW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the CMB ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(1.0), 1000)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Deltachi = Chia
    Deltachi[-1] = 0
    s2 = k[..., np.newaxis] * Deltachi

    integrand = special.spherical_jn(
        l, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_CMB = 8 * np.pi * \
        (1j**l) * integrate.simps(integrand, a)

    return g_isw_CMB


def G_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of kSZ integral kernels"""

    # Lower sampling for slow ISW term
    ks_isw = np.logspace(conf.k_min, conf.k_max, conf.k_res//10)
    Gs_isw = G_ISW_ksz( ks_isw, ze, Omega_b, Omega_c, w, wa, Omega_K, h )
    G_int_ISW = interp1d(
        ks_isw, Gs_isw, kind="cubic", bounds_error=False, fill_value='extrapolate')

    return G_int_ISW(k) \
        + G_SW_ksz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h ) \
        + G_Dopp_ksz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )


def G_ksz_localDopp(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ integral kernel including only local peculiar velocity"""
    G_s = G_localDopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    return G_s


def G_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of psz integral kernels"""
    G = G_SW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_Dopp_psz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_ISW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)

    return G


def G_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of CMB integral kernels"""
    G = G_SW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_Dopp_CMB( k, l, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_ISW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h)

    return G



##########################################################
################                       ###################
################ TRANSFER FUNCTIONS    ###################
################                       ###################
##########################################################


def Chi_bin_boundaries(z_min, z_max, N) :
    """
    Get comoving distances (chi) of boundaries of N bins from z_min to z_max,
    equally spaced in comoving distance
    """
    Chi_min = Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(z_min))
    Chi_max = Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(z_max))
    Chi_boundaries = np.linspace(Chi_min, Chi_max, N+1)
    return Chi_boundaries


def Chi_bin_centers(z_min, z_max, N) :
    """
    Get comoving distances at center of of bins from Chi_bin_boundaries()
    """
    Chi_boundaries = Chi_bin_boundaries(z_min, z_max, N)
    # Get center of bins in comoving distance, convert to redshift
    Chis = ( Chi_boundaries[:-1] + Chi_boundaries[1:] ) / 2.0
    return Chis


def Z_bin(N) :
    """
    Get redshifts corresponding to bin centers from Chi_bin_centers
    """
    Chis = Chi_bin_centers(conf.z_min, conf.z_max, N)
    return zfromchi(Chis)


def Z_bin_samples(N_bins, Bin_num, N_samples_in_bin):
    """
    Get redshifts of samples in a "bin" between conf.z_min and conf.z_max,
    uniformly distributed in chi, and at bin centers (so excluding boundaries.)

    N = number of bins between z_min and z_max
    B = bin number to get samples in
    N_samples = number of samples in bin
    """
    # Get boundaries of bins
    Chi_boundaries = Chi_bin_boundaries(conf.z_min, conf.z_max, N_bins)
    Z_boundaries = zfromchi(Chi_boundaries)

    # Generate redshift samples inside bin
    Chi_samples = np.linspace(Chi_boundaries[Bin_num], Chi_boundaries[Bin_num + 1], N_samples_in_bin)

    # Translate this to redshift boundaries
    z_samples = zfromchi(Chi_samples)
    return z_samples


def Z_bin_samples_conf(Bin_num, config=conf):
    return Z_bin_samples(config.N_bins, Bin_num, config.n_samples)


def Get_Windowed_Transfer(Transfer, redshifts, redshift_weights, L, *args, **kwargs) :
    """
    Generic function to compute windowed transfer functions; computes:

      Transfer(k, L, z, args, kwargs)

    at each redshift in `redshifts`, then integrates between
    redshifts[0] and redshifts[-1] using `redshift_weights` as weighting.
    Weights don't have to be normalized (they are below).
    Cosmology parameters are those provided in kszpsz_config (for now).

    *args and **kwargs are passed through to transfer.
    For a single redshift, no integration is performed.
    ks are determined by config passed in kwargs, otherwise by kszpsz_config sampling.
    """
    if 'config' in kwargs :
        k = k_sampling(config=kwargs['config'])
    else:
        k = k_sampling()
    N_samples = len(redshifts)
    T_samples = np.zeros((N_samples, len(k), len(L)), dtype=np.complex64)


    if N_samples == 1:
        z = redshifts[0]
        return Transfer(k, L, z, *args, **kwargs)
    else :
        Window_norm = integrate.simps(redshift_weights, redshifts)

        for n in np.arange(N_samples) :
            z = redshifts[n]
            T_samples[n] = redshift_weights[n] \
              * Transfer(k, L, z, *args, **kwargs)

        Transfer_windowed = integrate.simps(T_samples, redshifts, axis=0) / Window_norm

        return Transfer_windowed


def Transfer_ksz_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function at a given redshift `ze`
    """
    transfer_ksz = np.zeros((len(k), len(L)), dtype=np.complex64)
    Ker = G_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_ksz[:, l_id] = 0.0 + 1j * 0.0
        else:
            c = (4 * np.pi * (1j)**l) / (2 * l + 1)
            transfer_ksz[:, l_id] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    return transfer_ksz


def Transfer_ksz_bin(N, n, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given binning scheme
    """
    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(n):
        Ker = G_ksz(k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l == 0:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = (4 * np.pi * (1j)**l) / (2 * l + 1)
                T_samples[:, l, j] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                    l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    transfer_ksz = (1 / n) * np.sum(T_samples, axis=-1)

    return transfer_ksz


def Transfer_ksz_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    kSZ transfer function averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_ksz_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_ksz_localDopp_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given redshift using only local doppler
    """
    transfer_ksz = np.zeros((len(k), len(L)), dtype=np.complex64)
    Ker = G_ksz_localDopp(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_ksz[:, l_id] = 0.0 + 1j * 0.0
        else:
            c = (4 * np.pi * (1j)**l) / (2 * l + 1)
            transfer_ksz[:, l_id] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    return transfer_ksz


def Transfer_ksz_localDopp_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    kSZ transfer function, local doppler only, averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_ksz_localDopp_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_ksz_bin_localDopp(N, n, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given bin using only local doppler
    """
    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(0, n):
        Ker = G_ksz_localDopp(
            k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l == 0:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = (4 * np.pi * (1j)**l) / (2 * l + 1)
                T_samples[:, l, j] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                    l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    transfer_ksz = (1 / n) * np.sum(T_samples, axis=-1)

    return transfer_ksz


def Transfer_psz_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """PSZ transfer function for a given redshift"""
    if ze == 0.0:
        transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
        Ker = G_psz(k, 0.0, Omega_b, Omega_c, w, wa, Omega_K, h)
        Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

        for l_id, l in enumerate(L):
            if l == 2:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                transfer_psz[:, l_id] = c * (1. / 15.) * Tk * Ker
            else:
                transfer_psz[:, l_id] = 0.0 + 1j * 0.0

    else:
        transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
        Ker = G_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
        Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

        for l_id, l in enumerate(L):
            if l < 2:
                transfer_psz[:, l_id] = 0.0 + 1j * 0.0
            else:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                transfer_psz[
                    :, l] = c * (special.spherical_jn(l, k * Chie) / ((k * Chie)**2)) * Tk * Ker

        return transfer_psz


def Transfer_psz_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    pSZ transfer function averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_psz_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_psz_bin(N, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """PSZ transfer function for a given bin"""
    n = 10

    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)

    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(0, n):

        Ker = G_psz(k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l < 2:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                T_samples[
                    :, l, j] = c * (special.spherical_jn(l, k * Chie) / ((k * Chie)**2)) * Tk * Ker

    transfer_psz = (1.0 / n) * np.sum(T_samples, axis=-1)

    return transfer_psz


def Transfer_CMB(k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB transfer function for large scales at z = 0"""
    transfer_CMB = np.zeros((len(k), len(L)), dtype=np.complex64)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for l_id, l in enumerate(L):
        if l < 1:
            transfer_CMB[:, l_id] = 0.0 + 1j * 0.0
        else:
            Ker = G_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h)
            transfer_CMB[:, l_id] = Tk * Ker

    return transfer_CMB


def Transfer_E(k, L, Omega_b, Omega_c, w, wa, Omega_K, h, tau):
    """E transfer function for large scales at z = 0"""
    transfer_E = np.zeros((len(k), len(L)), dtype=np.complex64)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chi_grid = np.linspace(0.0, chifromz(z_re(Omega_b, h, tau)), 40)
    Z_grid = zfromchi(Chi_grid)
    a_grid = az(Z_grid)
    sigma_ne = sigma_nez(Z_grid, Omega_b, h)
    etau = np.exp(-tau_grid(Chi_grid, Z_grid, Omega_b, h))

    Integrand = np.zeros((len(k), len(L), len(Chi_grid)), dtype=np.complex64)
    for i in np.arange(len(Chi_grid)):
        Integrand[:, :, i] = (-np.sqrt(6.0) / 10.0) * Transfer_psz_redshift(k, L, Z_grid[
            i], Omega_b, Omega_c, w, wa, Omega_K, h) * a_grid[i] * sigma_ne[i] * etau[i]

    transfer_E = integrate.simps(Integrand, Chi_grid, axis=-1)

    return transfer_E


###############################################################################################
################                                                            ###################
################ Transfer function and signal covariance for number counts  ###################
################                                                            ###################
###############################################################################################


def f_evol(k, z, config=conf) :
    return np.full_like(k, bs.f_evo(z, config))

def s(k, z, config=conf) :
    return np.full_like(k, bs.s_magbias(z, config))

def bG(k, z, config=conf) :
    """ Standard/'usual galaxy bias """
    return np.full_like(k, bs.bias(z, config))

def bNG(k, z, config=conf) :
    """
    Scale-dependent bias from primordial non-Gaussianity
    a la Eq. 13 in https://arxiv.org/pdf/1507.03550.pdf
    """
    Trans = T_config(k, config)
    Dpsi = Dpsi_inter_config(config)(az(z))
    return 3.0*config.fNL*(bG(k, z) - 1.0)*(config.Omega_b+config.Omega_c)*H0(config.h)**2 * config.delta_collapse\
        / ( k**2 * Trans * Dpsi )

def b(k, z, config=conf) :
    """Galaxy bias plus fNL piece"""
    return bG(k, z, config) + bNG(k, z, config)


def Integral_Interpolation_jlDpsi(k, l, Chi_integrand, Dpsi_integrand):
    """
    cumulative integrated function of number counts term, (interpolated),
    """
    integrand = special.spherical_jn(l, k[:,None]*Chi_integrand) * Dpsi_integrand
    cum_int = integrate.cumtrapz(integrand, Chi_integrand, initial=0.0)
    return interp2d(k, Chi_integrand, np.transpose(cum_int))


def Integral_Interpolation_jlDpsi_chi(k, l, Chi_integrand, Dpsi_integrand):
    """
    cumulative integrated function of number counts term, (interpolated),
    """
    integrand = special.spherical_jn(l, k[:,None]*Chi_integrand) * Dpsi_integrand / Chi_integrand
    cum_int = integrate.cumtrapz(integrand, Chi_integrand, initial=0.0)
    return interp2d(k, Chi_integrand, np.transpose(cum_int))


def Integral_Interpolation_jlDpsi_prime(k, l, Chi_integrand, Dpsi_prime_integrand):
    """
    cumulative integrated function of number counts term, (interpolated)
    """
    integrand = special.spherical_jn(l, k[:,None]*Chi_integrand) * Dpsi_prime_integrand
    cum_int = integrate.cumtrapz(integrand, Chi_integrand, initial=0.0)
    return interp2d(k, Chi_integrand, np.transpose(cum_int))


def get_NGR_interps(l, config=conf):
    k = k_sampling(config)
    Ls = L_sampling(config)

    a_integrand = np.linspace(az(config.z_max), 1.0 - 1.0e-5, config.NGR_transfer_integrand_sampling)
    H_integrand = H_config(a_integrand, config)
    Chi_integrand = Chia_inter_config(config)(a_integrand)
    Dpsi_integrand = Dpsi_inter_config(config)(a_integrand)
    # d D_psi / da :
    dDpsida_integrand = derv_Dpsi_inter_config(config)(a_integrand)
    Dpsi_prime_integrand = a_integrand**2*H_integrand*dDpsida_integrand

    NGR_interps = {
        'jlDpsi': Integral_Interpolation_jlDpsi(k, l, Chi_integrand, Dpsi_integrand),
        'jlDpsi_chi': Integral_Interpolation_jlDpsi_chi(k, l, Chi_integrand, Dpsi_integrand),
        'jlDpsi_prime': Integral_Interpolation_jlDpsi_prime(k, l, Chi_integrand, Dpsi_prime_integrand),
    }
    return NGR_interps


def Transfer_NGR_redshift(k, L, ze, terms=('all',), config=conf, interps=None):
    """
    General relativistic number counts ("NGR") transfer function for a given redshift
    based on Eq. A1-A6 in https://arxiv.org/pdf/1710.02477.pdf
    fixed "f_evol" and "s"
    terms can contain: ('all', 'delta', 'delta_NG', 'rsd', 'v1v2', 'lens',
        'p1', 'p2', 'p3', 'p4', 'isw', 'dbG', 'dfNL', 'ds', 'dfevol')
    Where: dbg, dfnl, ds, df are derivatives wrt. biases.
    """
    transfer = np.zeros((len(k),len(L)), dtype = np.complex64)

    if 'all' in terms:
        transfer += Transfer_NGR_all_redshift(k, L, ze, config=config, interps=interps)
    elif 'dbG' in terms :
        transfer += Transfer_NGR_dbG_redshift(k, L, ze, config=config, interps=interps)
    elif 'dfNL' in terms :
        transfer += Transfer_NGR_dfNL_redshift(k, L, ze, config=config, interps=interps)
    elif 'ds' in terms :
        transfer += Transfer_NGR_ds_redshift(k, L, ze, config=config, interps=interps)
    elif 'dfevol' in terms :
        transfer += Transfer_NGR_dfevol_redshift(k, L, ze, config=config, interps=interps)
    else:
        if 'delta' in terms :
            transfer += Transfer_NGR_delta_redshift(k, L, ze, config=config)
        if 'delta_NG' in terms :
            transfer += Transfer_NGR_delta_NG_redshift(k, L, ze, config=config)
        if 'rsd' in terms :
            transfer += Transfer_NGR_RSD_redshift(k, L, ze, config=config)
        if 'v1v2' in terms :
            transfer += Transfer_NGR_V1V2_redshift(k, L, ze, config=config)
        if 'lens' in terms :
            transfer += Transfer_NGR_Lensing_redshift(k, L, ze, config=config, interps=interps)
        if 'p1' in terms :
            transfer += Transfer_NGR_P1_redshift(k, L, ze, config=config)
        if 'p2' in terms :
            transfer += Transfer_NGR_P2_redshift(k, L, ze, config=config)
        if 'p3' in terms :
            transfer += Transfer_NGR_P3_redshift(k, L, ze, config=config)
        if 'p4' in terms :
            transfer += Transfer_NGR_P4_redshift(k, L, ze, config=config, interps=interps)
        if 'isw' in terms :
            transfer += Transfer_NGR_ISW_redshift(k, L, ze, config=config, interps=interps)

    return transfer


def Delta(z, k, config=conf, use_bias=True) :
    a = az(z)
    Dpsi = Dpsi_inter_config(config)(a)
    Trans = T_config(k, config)
    bias = bG(k, z) if use_bias else 1.0

    return Dpsi * Trans * 2.0*k**2*a/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2)


def Transfer_NGR_delta_redshift(k, L, ze, config=conf, use_bias=True):
    """
    General relativistic number counts ("NGR") transfer function for a given redshift
    Density component
    Comoving CDM density perturbation is given in terms of newtonian potential via Poisson's equation,
    eg. as noted in Eq. 45 here: https://arxiv.org/pdf/astro-ph/0402060.pdf
    """
    transfer_NGR = np.zeros((len(k), len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Dpsi = Dpsi_inter_config(config)(ae)
    Trans = T_config(k, config)
    bias = bG(k, ze) if use_bias else 1.0

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = - bias * 4.0*np.pi*(1j)**l * Dpsi * Trans\
              * 2.0*k**2*ae/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2) \
              * special.spherical_jn(l, k*Chie)

    return transfer_NGR


def Transfer_NGR_delta_NG_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts ("NGR") transfer function for a given redshift
    fNL (primordial non-gaussianity) density component
    Same as Transfer_NGR_delta_redshift but different bias term
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Dpsi = Dpsi_inter_config(config)(ae)
    Trans = T_config(k, config)
    bias = bNG(k, ze, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = -bias * 4.0*np.pi*(1j)**l * Dpsi * Trans\
              * 2.0*k**2*ae/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2) \
              * special.spherical_jn(l, k*Chie)
    
    return transfer_NGR


def Transfer_NGR_RSD_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts transfer function for a given redshift
    RSD component
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    H = H_config(ae, config)
    Dv = Dv_inter_config(config)(ae)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else :
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * Dv/(ae*H) * k**2 \
                * spherical_jn_pp(l, k*Chie)

    return transfer_NGR


def Transfer_NGR_Lensing_redshift(k, L, ze, config=conf, interps=None):
    """
    General relativistic number counts transfer function for a given redshift
    Lensing component
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral = (2.0 - 5.0*s(k, ze)) * (
                1.0/Chie * interps[l]['jlDpsi'](k, Chie)
                - interps[l]['jlDpsi_chi'](k, Chie)
            )
            transfer_NGR[:, l_id] = -l*(l+1)*4.0*np.pi*(1j)**l * Trans * integral

    return transfer_NGR


def Transfer_NGR_V1V2_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts transfer function for a given redshift
    Peculiar velocity terms
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Dv = Dv_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    Trans = T_config(k, config)

    prefactor = (1.0 + Hprime/ae/Hubble**2 + (2.0-5.0*s(k, ze))/(Chie*ae*Hubble) + 5.0*s(k, ze) - f_evol(k, ze) )

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * (Dv*k**2) * (
                    (f_evol(k, ze) - 3.0) * ae*Hubble / k**2 * special.spherical_jn(l, k*Chie)
                    + prefactor / k * special.spherical_jn(l, k*Chie, derivative=True)
                )

    return transfer_NGR


def Transfer_NGR_P1_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts transfer function for a given redshift
    P1 term
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    Dpsi = Dpsi_inter_config(config)(ae)
    dDpsida = derv_Dpsi_inter_config(config)(ae)
    Dpsi_prime = ae**2*Hubble*dDpsida
    p1_prefactor = 2.0 + Hprime/ae/Hubble**2 + (2.0-5.0*s(k, ze))/(Chie*ae*Hubble) + 5.0*s(k, ze) - f_evol(k, ze)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * special.spherical_jn(l, k*Chie) * (
                    p1_prefactor * Dpsi # P1
                )

    return transfer_NGR


def Transfer_NGR_P2_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts transfer function for a given redshift
    P2 term
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    Dpsi = Dpsi_inter_config(config)(ae)
    dDpsida = derv_Dpsi_inter_config(config)(ae)
    Dpsi_prime = ae**2*Hubble*dDpsida
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * special.spherical_jn(l, k*Chie) * (
                    (5.0*s(k, ze) - 2.0) * Dpsi # P2
                )

    return transfer_NGR


def Transfer_NGR_P3_redshift(k, L, ze, config=conf):
    """
    General relativistic number counts transfer function for a given redshift
    P3 term
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    Dpsi = Dpsi_inter_config(config)(ae)
    dDpsida = derv_Dpsi_inter_config(config)(ae)
    Dpsi_prime = ae**2*Hubble*dDpsida
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * special.spherical_jn(l, k*Chie) * (
                    2.0*Dpsi_prime / (ae*Hubble) # P3
                )

    return transfer_NGR


def Transfer_NGR_P4_redshift(k, L, ze, config=conf, interps=None):
    """
    General relativistic number counts transfer function for a given redshift
    P4 term
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral = 2.0 * (2.0 - 5.0*s(k, ze)) / Chie * interps[l]['jlDpsi'](k, Chie)
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * integral

    return transfer_NGR


def Transfer_NGR_ISW_redshift(k, L, ze, config=conf, interps=None):
    """
    General relativistic number counts transfer function for a given redshift
    ISW term, ie. last line of Eq. A.20 in https://arxiv.org/pdf/1307.1459.pdf
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Trans = T_config(k, config)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    prefactor = (1.0 + Hprime/ae/Hubble**2 + (2.0-5.0*s(k,ze))/(Chie*ae*Hubble) + 5.0*s(k,ze) - f_evol(k,ze) )

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral = 2.0*interps[l]['jlDpsi_prime'](k, Chie)
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * prefactor * integral

    return transfer_NGR


def Transfer_NGR_all_redshift(k, L, ze, config=conf, interps=None):
    """
    General relativistic number counts ("NGR") transfer function for a given redshift
    Including all terms, ie. Eq. A.20 from here:
    https://arxiv.org/pdf/1307.1459.pdf
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    Hprime = ae*dHdt_config(ae, config)
    prefactor = 1.0 + Hprime/ae/Hubble**2 + (2.0-5.0*s(k,ze))/(Chie*ae*Hubble) + 5.0*s(k,ze) - f_evol(k,ze)
    bias = b(k, ze, config)

    Dpsi = Dpsi_inter_config(config)(ae)
    dDpsida = derv_Dpsi_inter_config(config)(ae)
    Dv = Dv_inter_config(config)(ae)
    Dpsi_prime = ae*ae*Hubble*dDpsida
    Ddelta_m = Dpsi * 2.0*k**2*ae/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2)

    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral_term = (
                2.0* (2.0 - 5.0*s(k,ze))/2.0 * (
                    l*(l+1.0) * interps[l]['jlDpsi_chi'](k, Chie)
                    - l*(l+1.0) * interps[l]['jlDpsi'](k, Chie)/Chie
                    + 2.0*interps[l]['jlDpsi'](k, Chie)/Chie
                )
                + 2.0*prefactor*interps[l]['jlDpsi_prime'](k, Chie)
            )

            jl = special.spherical_jn(l, k*Chie)
            jlp = special.spherical_jn(l, k*Chie, derivative=True)
            jlpp = spherical_jn_pp(l, k*Chie)

            # transfer function
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * (
                jl * (
                        -bias*Ddelta_m + (prefactor+1.0)*Dpsi
                        + (5.0*s(k,ze) - 2.0)*Dpsi + Dpsi_prime/(ae*Hubble)
                ) + 1.0*(
                    prefactor*jlp
                    + jlpp*k/(ae*Hubble)
                    + (f_evol(k,ze) - 3.0)*jl*ae*Hubble/k
                )*(k**2*Dv)/k
                + integral_term
            )

    return transfer_NGR


def Transfer_NGR_dbG_redshift(k, L, ze, config=conf, interps=None):
    """
    Analytic derivative wrt. galaxy bias bG of NGR transfer function
    (All terms included.)
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Dpsi = Dpsi_inter_config(config)(ae)
    Ddelta_m = Dpsi * 2.0*k**2*ae/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            jl = special.spherical_jn(l, k*Chie)
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * jl * (-Ddelta_m)

    return transfer_NGR


def Transfer_NGR_dfNL_redshift(k, L, ze, config=conf, interps=None):
    """
    Analytic derivative wrt. f_NL of NGR transfer function
    (All terms included.)
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Dpsi = Dpsi_inter_config(config)(ae)
    Ddelta_m = Dpsi * 2.0*k**2*ae/(3.0*(config.Omega_b+config.Omega_c)*H0(config.h)**2)
    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            jl = special.spherical_jn(l, k*Chie)
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * jl * (-Ddelta_m) \
                * 3.0*(bG(k, ze) - 1.0)*(config.Omega_b+config.Omega_c)*H0(config.h)**2 * config.delta_collapse\
                / ( k**2 * Trans * Dpsi )

    return transfer_NGR


def Transfer_NGR_ds_redshift(k, L, ze, config=conf, interps=None):
    """
    Analytic derivative wrt. magnification bias s of NGR transfer function
    (All terms included.)
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    ds_prefactor = -5.0/(Chie*ae*Hubble) + 5.0

    Dpsi = Dpsi_inter_config(config)(ae)
    Dv = Dv_inter_config(config)(ae)

    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral_term = (
                -5.0*s(k,ze) * (
                    l*(l+1.0) * interps[l]['jlDpsi_chi'](k, Chie)
                    - l*(l+1.0) * interps[l]['jlDpsi'](k, Chie)/Chie
                    + 2.0*interps[l]['jlDpsi'](k, Chie)/Chie
                )
                + 2.0*ds_prefactor*interps[l]['jlDpsi_prime'](k, Chie)
            )

            jl = special.spherical_jn(l, k*Chie)
            jlp = special.spherical_jn(l, k*Chie, derivative=True)

            # transfer function
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * (
                jl * ( ds_prefactor*Dpsi + 5.0*Dpsi )
                + 1.0*ds_prefactor*jlp*k*Dv
                + integral_term
            )

    return transfer_NGR


def Transfer_NGR_dfevol_redshift(k, L, ze, config=conf, interps=None):
    """
    Analytic derivative wrt. evolution bias f_evol of NGR transfer function
    (All terms included.)
    """
    transfer_NGR = np.zeros((len(k),len(L)), dtype=np.complex64)

    ae = az(ze)
    Chie = Chia_inter_config(config)(ae)
    Hubble = H_config(ae, config)
    df_prefactor = -1.0

    Dpsi = Dpsi_inter_config(config)(ae)
    Dv = Dv_inter_config(config)(ae)

    Trans = T_config(k, config)

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_NGR[:, l_id] = 0.0 + 1j * 0.0
        else:
            integral_term = 2.0*df_prefactor*interps[l]['jlDpsi_prime'](k, Chie)

            jl = special.spherical_jn(l, k*Chie)
            jlp = special.spherical_jn(l, k*Chie, derivative=True)

            # transfer function
            transfer_NGR[:, l_id] = 4.0*np.pi*(1j)**l * Trans * (
                jl * (
                        df_prefactor*Dpsi
                ) + 1.0*(
                    df_prefactor*jlp + 1.0*jl*ae*Hubble/k
                )*(k**2*Dv)/k
                + integral_term
            )

    return transfer_NGR


def Transfer_NGR_windowed(L, z_samples, z_sample_weights,
    terms=('all',), config=conf, interps=None):
    """
    Number counts transfer function averaged in a bin
    N = number of samples between z_min and z_max
    """
    return Get_Windowed_Transfer(Transfer_NGR_redshift,
        z_samples, z_sample_weights, L, terms=terms, config=config, interps=interps)


def get_PkDeltaDelta(ze):
    k = k_sampling()
    a = az(ze)
    Dpsi = Dpsi_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(a)
    Trans = T(k, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)

    return {
        'k': k,
        'Pk': Dpsi**2 * Trans**2 * (2.0*k**2*a/(3.0*(config.Omega_b+config.Omega_c)*H0(conf.h)**2))**2 \
                * Ppsi(k, conf.As, conf.ns)
    }


def get_CLDeltaDelta(ze):
    """Delta_l(k) Transfer function for synchronous gauge density perutrbations"""
    print("Calculating Cl_DeltaDelta...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)
    T_list = [ Transfer_NGR_delta_redshift(k, Lv, ze,
            conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h) ]

    return {
        'Cl': CL_bins(T_list, T_list, k, Lv),
        'l': Lv
    }


def get_CLNGRNGR_windowed(z_samples, z_sample_weights,
            terms_1=('all',), terms_2=('all',) ):
    """
    Number counts (cross-) correlations,
    allowing coefficients for specific terms, averaged in a redshift range.
    """
    print("Calculating Cl_NGRNGR...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)
    NGR_interps = {}
    for l in Lv:
        NGR_interps[l] = get_NGR_interps(l, config=conf)
    T_list_1 = [ Transfer_NGR_windowed(Lv, z_samples, z_sample_weights, terms=terms_1, interps=NGR_interps) ]
    T_list_2 = [ Transfer_NGR_windowed(Lv, z_samples, z_sample_weights, terms=terms_2, interps=NGR_interps) ]

    return {
        'Cl': CL_bins(T_list_1, T_list_2, k, Lv),
        'l': Lv
    }


def get_CLNGRNGR(ze, terms_1=('all',), terms_2=('all',) ):
    """
    Number counts (cross-) correlations,
    allowing coefficients for specific terms at a fixed redshift.
    Defaults to using all terms, but specific terms can be requested (see Transfer_NGR_redshift).
    """
    print("Calculating Cl_NGRNGR...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)
    NGR_interps = {}
    for l in Lv:
        NGR_interps[l] = get_NGR_interps(l, config=conf)
    T_list_1 = [ Transfer_NGR_redshift(k, Lv, ze, terms=terms_1, interps=NGR_interps) ]
    T_list_2 = [ Transfer_NGR_redshift(k, Lv, ze, terms=terms_2, interps=NGR_interps) ]

    return {
        'Cl': CL_bins(T_list_1, T_list_2, k, Lv),
        'l': Lv
    }


#############################################################################################
################                                                          ###################
################ Transfer function and signal covariance for moving lens  ###################
################                                                          ###################
#############################################################################################



def Transfer_ML_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Moving lens v_perp transfer function for a given redshift
    """
    transfer_ML = np.zeros((len(k),len(L)), dtype = np.complex64)
    Dv = Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h )
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_ML[:, l_id] = 0.0 +1j*0.0
        else:
            c = -(4*np.pi*(1j)**l)
            transfer_ML[:, l_id]= c*special.spherical_jn(l, k*Chie)*Tk*Dv/Chie

    return transfer_ML


def Transfer_ML_bin(N, n, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Moving lens v_perp transfer function for a given bin
    """
    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(0, n):
        Dv = Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(z_samples[j]))
        Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(z_samples[j]))

        for l in L:
            if l == 0:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = -(4 * np.pi * (1j)**l)
                T_samples[:, l, j] = c * \
                    special.spherical_jn(l, k * Chie) * Tk * Dv / Chie

    transfer_ML = (1 / n) * np.sum(T_samples, axis=-1)
    return transfer_ML


def Transfer_ML_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    Moving lens transfer function averaged using a window
    """
    return Get_Windowed_Transfer(Transfer_ML_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def get_CLvtvt():
    print("Calculating Cl_vtvt...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)

    T_list = []
    for i in np.arange(conf.N_bins):
        T_list.append(Transfer_ML_bin(conf.N_bins, conf.n_samples, i, k, Lv,
            conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin " + str(i))

    CLvtvt = CL_bins(T_list, T_list, k, Lv, conf.As, conf.ns)

    return CLvtvt



###############################################################################
################                                            ###################
################ Transfer functions for tensors             ###################
################                                            ###################
###############################################################################

# Tensor power spectrum

def PT(k, r, As, nt):

    P = (r*As*(10**-9))*((k/conf.k0)**(nt))

    return P/2.

# Derivative of tensor power spectrum with respect to n_t"

def dPTdnt(k, r, As, nt):

    P = (r*As*(10**-9))*((k/conf.k0)**(nt))*np.log(k/conf.k0)

    return P/2.

# A few functions relevant for tensors

# Function A_l(k,chi).

def Aell( l, k, chi ):

    if chi < 10**(-6):
        Al = 0.0
    else:
        Al = 2 * special.spherical_jn(l,k*chi) / ( k*chi ) + special.spherical_jn(l, k*chi, True)

    return Al/2

#  Function B_l(k,chi).

def Bell( l, k, chi ):
    if chi==0.0:
        Bell=(-1./5)*np.ones(k.shape[0])
    else:
        Bl1 = -.25 * spherical_jn_pp(l, k*chi)
        Bl2 = - 1 / ( k*chi ) * special.spherical_jn(l, k*chi, True)
        Bl3 = special.spherical_jn(l,k*chi) * (.25 - 1 / (2 * (k*chi)**2))
        Bell=Bl1+Bl2+Bl3
    return Bell


# Integral kernel for pSZ from tensors

def I_psz(k,ze,l, Omega_b, Omega_c, w, wa, Omega_K, h):

    a = np.logspace(np.log10(conf.adec),np.log10(az(ze)),1000)

    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)

    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    Deltachi = Chia-Chie

    Deltachi[-1] = 0

    tau0=chia( 0.0, Omega_b, Omega_c, w, wa, Omega_K, h)

    tau = tau0-Chia

    s2= k[...,np.newaxis]*Deltachi

    ktau = k[...,np.newaxis]*tau

    integrand1temp = special.spherical_jn(2, s2[:,0:-1])/(s2[:,0:-1]*s2[:,0:-1])

    integrand1=np.ones(s2.shape)

    integrand1[:,0:-1]=integrand1temp[:,:]

    integrand1[:,-1]=1./15

    integrand2 = 3.0*((special.spherical_jn(1, ktau,True)/ktau)-special.spherical_jn(1, ktau)/(ktau*ktau))

    integrand=integrand1*integrand2

    I_psz = 2.*np.pi*np.sqrt(6)*integrate.simps(integrand, tau)

    return k*I_psz


# PSZ E-mode Tensor transfer function for a given redshift

def Transfer_pszEtens_redshift(k, L , ze, Omega_b, Omega_c, w, wa, Omega_K, h):

    chie=chifromz(ze)

    transfer_pszEtens = np.zeros((len(k),len(L)))
    for l in L:
        if l < 2 :
            transfer_pszEtens[:,l] = 0.0

        else:

            Ker = I_psz(k,ze,l, Omega_b, Omega_c, w, wa, Omega_K, h)
            transfer_pszEtens[:,l] = 5.*Ker*Bell( l, k, chie)/(np.sqrt(2.*np.pi))


    return transfer_pszEtens


def Transfer_pszBtens_redshift(k, L , ze, Omega_b, Omega_c, w, wa, Omega_K, h):

    chie=chifromz(ze)

    transfer_pszBtens = np.zeros((len(k),len(L)))
    for l in L:
        if l < 2 :

            transfer_pszBtens[:,l] = 0.0

        else:

            Ker = I_psz(k,ze,l, Omega_b, Omega_c, w, wa, Omega_K, h)
            transfer_pszBtens[:,l] = 5.*Ker*Aell( l, k, chie)/(np.sqrt(2.*np.pi))


    return transfer_pszBtens


def Transfer_pszEtens_bin(N, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):

    n = 10

    T_samples = np.zeros((len(k),len(L),) +(n,) )

    z_samples = Z_bin_samples(N, B, n)

    for j in np.arange(0,n):

        chie=chifromz(z_samples[j])

        for l in L:
            if l < 2 :

                T_samples[:,l,j] = 0.0

            else:

                Ker = I_psz(k,z_samples[j],l, Omega_b, Omega_c, w, wa, Omega_K, h)
                T_samples[:,l,j] = 5.*Ker*Bell( l, k, chie)/(np.sqrt(2.*np.pi))

    transfer_pszEtens = (1.0/n)*np.sum(T_samples, axis=-1)

    return transfer_pszEtens

def Transfer_pszBtens_bin(N, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):

    n = 10

    T_samples = np.zeros((len(k),len(L),) +(n,) )

    z_samples = Z_bin_samples(N, B, n)

    for j in np.arange(0,n):

        chie=chifromz(z_samples[j])

        for l in L:
            if l < 2 :

                T_samples[:,l,j] = 0.0

            else:

                Ker = I_psz(k,z_samples[j],l, Omega_b, Omega_c, w, wa, Omega_K, h)
                T_samples[:,l,j] = 5.*Ker*Aell( l, k, chie)/(np.sqrt(2.*np.pi))

    transfer_pszBtens = (1.0/n)*np.sum(T_samples,axis= -1)

    return transfer_pszBtens


def I_Ttens(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):

    ze=0.0

    a = np.logspace(np.log10(conf.adec),np.log10(az(ze)),1000)

    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)

    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    Deltachi = Chia-Chie

    Deltachi[-1] = 0

    tau0=chia( 0.0, Omega_b, Omega_c, w, wa, Omega_K, h)

    tau = tau0-Chia

    s2= k[...,np.newaxis]*Deltachi

    ktau = k[...,np.newaxis]*tau

    integrand1temp = special.spherical_jn(l, s2[:,0:-1])/(s2[:,0:-1]*s2[:,0:-1])

    integrand1=np.ones(s2.shape)

    integrand1[:,0:-1]=integrand1temp[:,:]

    if l==2:
        integrand1[:,-1]=1./15
    else:
        integrand1[:,-1]=0.0

    integrand2 = 3.0*((special.spherical_jn(1, ktau,True)/ktau)-special.spherical_jn(1, ktau)/(ktau*ktau))

    integrand=integrand1*integrand2

    I_Ttens = integrate.simps(integrand, tau)

    return k*I_Ttens


def Transfer_Ttens(k, L , Omega_b, Omega_c, w, wa, Omega_K, h):

    ze=0.0

    chie=chifromz(ze)

    transfer_Ttens = np.zeros((len(k),len(L)))
    for l in L:
        if l < 2 :
            transfer_Ttens[:,l] = 0.0

        else:
            Ker = I_Ttens(k,l, Omega_b, Omega_c, w, wa, Omega_K, h)
            transfer_Ttens[:,l] = - Ker*np.sqrt(np.pi/2.)*np.sqrt(l*(l-1)*(l+1)*(l+2))


    return transfer_Ttens


def Transfer_Btens_novis(k, L , Omega_b, Omega_c, w, wa, Omega_K, h, tau):

    transfer_B = np.zeros((len(k),len(L)))

    Chi_grid = np.linspace(0.0, chifromz(z_re(Omega_b, h, tau)), 40)

    Z_grid = zfromchi(Chi_grid)

    a_grid = az(Z_grid)

    sigma_ne = sigma_nez(Z_grid, Omega_b, h)

    etau = np.exp(-tau_grid(Chi_grid,Z_grid, Omega_b, h))

    Integrand = np.zeros((len(k),len(L),len(Chi_grid)))

    for i in np.arange(len(Chi_grid)):

        Integrand[:,:,i] = (-np.sqrt(6.0)/10.0)*Transfer_pszBtens_redshift(k, L , Z_grid[i], Omega_b, Omega_c, w, wa, Omega_K, h)*a_grid[i]*sigma_ne[i]*etau[i]

    transfer_B = integrate.simps(Integrand, Chi_grid)

    return transfer_B



def Transfer_Btens(k, L , Omega_b, Omega_c, w, wa, Omega_K, h, tau):

    import camb

    transfer_B = np.zeros((len(k),len(L)))

    Z_grid = np.linspace(0.000001,15, 40)

    Chi_grid = chifromz(Z_grid)

    a_grid = az(Z_grid)

    Integrand = np.zeros((len(k),len(L),len(Chi_grid)))

    pars = camb.set_params(H0=conf.H0, ombh2=conf.ombh2, omch2=conf.omch2,tau=conf.tau)
    data= camb.get_background(pars)
    visibility = data.get_background_redshift_evolution(Z_grid, ['visibility'], format='array')

    for i in np.arange(len(Chi_grid)):

        Integrand[:,:,i] = (-np.sqrt(6.0)/10.0)*Transfer_pszBtens_redshift(k, L , Z_grid[i], Omega_b, Omega_c, w, wa, Omega_K, h)*visibility[i]

    transfer_B = integrate.simps(Integrand, Chi_grid)

    return transfer_B




def Transfer_Etens_novis(k, L , Omega_b, Omega_c, w, wa, Omega_K, h, tau):

    transfer_E = np.zeros((len(k),len(L)))

    Chi_grid = np.linspace(0.0, chifromz(z_re(Omega_b, h, tau)), 40)

    Z_grid = zfromchi(Chi_grid)

    a_grid = az(Z_grid)

    sigma_ne = sigma_nez(Z_grid, Omega_b, h)

    etau = np.exp(-tau_grid(Chi_grid,Z_grid, Omega_b, h))

    Integrand = np.zeros((len(k),len(L),len(Chi_grid)))

    for i in np.arange(len(Chi_grid)):

        Integrand[:,:,i] = (-np.sqrt(6.0)/10.0)*Transfer_pszEtens_redshift(k, L , Z_grid[i], Omega_b, Omega_c, w, wa, Omega_K, h)*a_grid[i]*sigma_ne[i]*etau[i]

    transfer_E = integrate.simps(Integrand, Chi_grid)

    return transfer_E


def Transfer_Etens(k, L , Omega_b, Omega_c, w, wa, Omega_K, h, tau):

    import camb

    transfer_B = np.zeros((len(k),len(L)))

    Z_grid = np.linspace(0.000001,15, 40)

    Chi_grid = chifromz(Z_grid)

    a_grid = az(Z_grid)

    Integrand = np.zeros((len(k),len(L),len(Chi_grid)))

    pars = camb.set_params(H0=conf.H0, ombh2=conf.ombh2, omch2=conf.omch2,tau=conf.tau)
    data= camb.get_background(pars)
    visibility = data.get_background_redshift_evolution(Z_grid, ['visibility'], format='array')

    for i in np.arange(len(Chi_grid)):

        Integrand[:,:,i] = (-np.sqrt(6.0)/10.0)*Transfer_pszEtens_redshift(k, L , Z_grid[i], Omega_b, Omega_c, w, wa, Omega_K, h)*visibility[i]

    transfer_E = integrate.simps(Integrand, Chi_grid)

    return transfer_E




######################################################################################################
################             Covariance matrix for tensor forecast          ###################
######################################################################################################

# CL at each bin

def CL_binstens(T_list1, T_list2, k, L, r, As, nt):

    CL= np.zeros((len(T_list1),len(L)))

    for i in np.arange(len(T_list1)):

        for l in L:

            T1= T_list1[i]

            T2= T_list2[i]

            I= (1./k)*PT(k, r, As, nt)*np.conj(T1[:,l])*T2[:,l]

            CL[i,l] = np.real(integrate.simps(I, k))

    return CL


def CL_covtens(T_list, k, L, r, As, nt):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= (1./k)*PT(k, r, As, nt)*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL

def dntCL_covtens(T_list, k, L, r, As, nt):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= (1./k)*dPTdnt(k, r, As, nt)*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL


def CL_covtens_bump(T_list, k, L, r, As, sigma, kp):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    Pbump = PT(k, r, As, 0.0)*np.exp(-(np.log(k/kp)**2)/(2.*sigma**2))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= (1./k)*Pbump*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL


def dsigmaCL_covtens_bump(T_list, k, L, r, As, sigma, kp):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    dsigmaPbump = PT(k, r, As, 0.0)*(np.log(k/kp)**2/(sigma**3))*np.exp(-(np.log(k/kp)**2)/(2.*sigma**2))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= (1./k)*dsigmaPbump*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL


def dkpCL_covtens_bump(T_list, k, L, r, As, sigma, kp):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    dkpPbump = PT(k, r, As, 0.0)*(np.log(k/kp)/(kp*sigma**2))*np.exp(-(np.log(k/kp)**2)/(2.*sigma**2))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= (1./k)*dkpPbump*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL


def get_CLtens_transfers():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list = []

    T_list.append(Transfer_Ttens(k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

    print("Done tensor T")

    T_list.append(Transfer_Etens(k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.tau))

    print("Done tensor E")

    T_list.append(Transfer_Btens(k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.tau))

    print("Done tensor B")

    for i in np.arange(conf.N_bins):

        T_list.append(Transfer_pszEtens_bin(conf.N_bins, i, k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

        print("Done pSZE bin "+str(i))

    for i in np.arange(conf.N_bins):

        T_list.append(Transfer_pszBtens_bin(conf.N_bins, i, k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

        print("Done pSZB bin "+str(i))

    np.save('T_list_tensors.npy',T_list)

    return T_list


def get_CLscalqCMB_transfers():

    k = k_sampling()

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list = []

    T_list.append(Transfer_CMB(k, Lq , conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

    print("Done scalar T")

    T_list.append(Transfer_E(k, Lq , conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.tau))

    print("Done scalar E")

    for i in np.arange(conf.N_bins):

        T_list.append(Transfer_psz_bin(conf.N_bins, i, k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

        print("Done pSZ bin "+str(i))

    np.save('T_list_scalqCMB.npy',T_list)

    return T_list



def get_CLscalCMB_cov():

    import camb
    import clprimary

    k = k_sampling()

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_scalqCMB.npy')

    CLscalCMB_cov_temp = CL_cov(T_list, k, Lq, conf.As, conf.ns)

    CLscalCMB_cov = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))

    CAMBCLs = clprimary.Cl_scalar_primary(ellmax=1000,muK=False)

    CLscalCMB_cov[0,0,:] = CLscalCMB_cov_temp[0,0,:] # TT
    CLscalCMB_cov[1,1,:] = CLscalCMB_cov_temp[1,1,:] # EE
    CLscalCMB_cov[0,1,:] = CLscalCMB_cov_temp[0,1,:] # TE
    CLscalCMB_cov[1,0,:] = CLscalCMB_cov_temp[1,0,:]
    CLscalCMB_cov[2,2,:] = CAMBCLs[0:conf.psz_estim_signal_lmax,2] # Lensing BB
    CLscalCMB_cov[3:conf.N_bins+3,3:conf.N_bins+3,:] = CLscalCMB_cov_temp[2:conf.N_bins+2,2:conf.N_bins+2,:] # pSZ auto
    CLscalCMB_cov[0,3:conf.N_bins+3,:] = CLscalCMB_cov_temp[0,2:conf.N_bins+2,:] # pSZ T
    CLscalCMB_cov[3:conf.N_bins+3,0,:] = CLscalCMB_cov_temp[2:conf.N_bins+2,0,:]
    CLscalCMB_cov[1,3:conf.N_bins+3,:] = CLscalCMB_cov_temp[1,2:conf.N_bins+2,:] # pSZ E
    CLscalCMB_cov[3:conf.N_bins+3,1,:] = CLscalCMB_cov_temp[2:conf.N_bins+2,1,:]

    return CLscalCMB_cov


def get_CLtens_cov():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = CL_covtens(T_list, k, Lq, conf.r, conf.As, conf.nt)

    CLtens_cov[0,2,:] = conf.Delta_c*CLtens_cov[0,2,:]    # TB
    CLtens_cov[2,0,:] = conf.Delta_c*CLtens_cov[2,0,:]
    CLtens_cov[1,2,:] = conf.Delta_c*CLtens_cov[1,2,:]    # EB
    CLtens_cov[2,1,:] = conf.Delta_c*CLtens_cov[2,1,:]
    CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:]   # TqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:]
    CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:]   # EqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:]
    CLtens_cov[2,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[2,3:3+conf.N_bins,:]   # BqE
    CLtens_cov[3:3+conf.N_bins,2,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,2,:]
    CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] # qEqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] # qEqB

    return CLtens_cov

def get_CLtens_cov_dr():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = CL_covtens(T_list, k, Lq, conf.r, conf.As, conf.nt)

    CLtens_cov[0,2,:] = conf.Delta_c*CLtens_cov[0,2,:]    # TB
    CLtens_cov[2,0,:] = conf.Delta_c*CLtens_cov[2,0,:]
    CLtens_cov[1,2,:] = conf.Delta_c*CLtens_cov[1,2,:]    # EB
    CLtens_cov[2,1,:] = conf.Delta_c*CLtens_cov[2,1,:]
    CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:]   # TqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:]
    CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:]   # EqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:]
    CLtens_cov[2,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[2,3:3+conf.N_bins,:]   # BqE
    CLtens_cov[3:3+conf.N_bins,2,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,2,:]
    CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] # qEqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] # qEqB

    return CLtens_cov/conf.r


def get_CLtens_cov_dDc():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = CL_covtens(T_list, k, Lq, conf.r, conf.As, conf.nt)

    CLtens_cov_dDc = np.zeros(CLtens_cov.shape)

    CLtens_cov_dDc[0,2,:] = CLtens_cov[0,2,:]    # TB
    CLtens_cov_dDc[2,0,:] = CLtens_cov[2,0,:]
    CLtens_cov_dDc[1,2,:] = CLtens_cov[1,2,:]    # EB
    CLtens_cov_dDc[2,1,:] = CLtens_cov[2,1,:]
    CLtens_cov_dDc[0,3+conf.N_bins:3+2*conf.N_bins,:] = CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:]   # TqB
    CLtens_cov_dDc[3+conf.N_bins:3+2*conf.N_bins,0,:] = CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:]
    CLtens_cov_dDc[1,3+conf.N_bins:3+2*conf.N_bins,:] = CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:]   # EqB
    CLtens_cov_dDc[3+conf.N_bins:3+2*conf.N_bins,1,:] = CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:]
    CLtens_cov_dDc[2,3:3+conf.N_bins,:] = CLtens_cov[2,3:3+conf.N_bins,:]   # BqE
    CLtens_cov_dDc[3:3+conf.N_bins,2,:] = CLtens_cov[3:3+conf.N_bins,2,:]
    CLtens_cov_dDc[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] = CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] # qEqB
    CLtens_cov_dDc[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] = CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] # qEqB

    return CLtens_cov_dDc

def get_CLtens_cov_dnt():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = dntCL_covtens(T_list, k, Lq, conf.r, conf.As, conf.nt)

    CLtens_cov[0,2,:] = conf.Delta_c*CLtens_cov[0,2,:]    # TB
    CLtens_cov[2,0,:] = conf.Delta_c*CLtens_cov[2,0,:]
    CLtens_cov[1,2,:] = conf.Delta_c*CLtens_cov[1,2,:]    # EB
    CLtens_cov[2,1,:] = conf.Delta_c*CLtens_cov[2,1,:]
    CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[0,3+conf.N_bins:3+2*conf.N_bins,:]   # TqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,0,:]
    CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[1,3+conf.N_bins:3+2*conf.N_bins,:]   # EqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,1,:]
    CLtens_cov[2,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[2,3:3+conf.N_bins,:]   # BqE
    CLtens_cov[3:3+conf.N_bins,2,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,2,:]
    CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] = conf.Delta_c*CLtens_cov[3:3+conf.N_bins,3+conf.N_bins:3+2*conf.N_bins,:] # qEqB
    CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] = conf.Delta_c*CLtens_cov[3+conf.N_bins:3+2*conf.N_bins,3:3+conf.N_bins,:] # qEqB

    return CLtens_cov


def get_CLtens_bump_cov():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = CL_covtens_bump(T_list, k, Lq, conf.r, conf.As, conf.sigma, conf.kp)

    return CLtens_cov

def get_CLtens_bump_cov_dr():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = CL_covtens_bump(T_list, k, Lq, 1.0, conf.As, conf.sigma, conf.kp)

    return CLtens_cov

def get_CLtens_bump_cov_dsigma():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = dsigmaCL_covtens_bump(T_list, k, Lq, conf.r, conf.As, conf.sigma, conf.kp)

    return CLtens_cov

def get_CLtens_bump_cov_dkp():

    k = np.logspace(conf.k_min_tens, conf.k_max_tens, conf.k_res_tens)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list=np.load('T_list_tensors.npy')

    CLtens_cov = dkpCL_covtens_bump(T_list, k, Lq, conf.r, conf.As, conf.sigma, conf.kp)

    return CLtens_cov


def get_NLtens():
    import cmblss
    import camb
    import kszpsz_estimator
    import clprimary

    estim_smallscale_lmax = conf.estim_smallscale_lmax
    estim_smallscale_lmin = conf.estim_smallscale_lmin
    psz_estim_signal_lmin = conf.psz_estim_signal_lmin
    psz_estim_signal_lmax = conf.psz_estim_signal_lmax
    zbins_nr = conf.zbins_nr

    # Calculate scalar signal covariance matrix.
    scal_CL = get_CLscalCMB_cov()

    # Calculate tensor signal covariance matrix.

    tens_CL = get_CLtens_cov()

    # Calculate reconstruction noise.
    Nl_qq_E, Nl_qq_B = kszpsz_estimator.calc_Nl_qq()

    Nl_qq_EE_diag = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))
    Nl_qq_BB_diag = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))

    for ell_id in range(psz_estim_signal_lmax):
        for zbins_id in range(zbins_nr):
            Nl_qq_EE_diag[3+zbins_id, 3+zbins_id, ell_id] = Nl_qq_E[zbins_id,ell_id]
            Nl_qq_BB_diag[3+conf.N_bins+zbins_id, 3+conf.N_bins+zbins_id, ell_id] = Nl_qq_B[zbins_id,ell_id]

    # Calculate CMB instrumental noise.

    Nl_CMB_inst = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))

    Nl_CMB_inst[0,0,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_T,noiseTuKArcmin=conf.noiseTuKArcmin_T,muK=False)
    Nl_CMB_inst[1,1,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_pol,noiseTuKArcmin=conf.noiseTuKArcmin_pol,muK=False)
    Nl_CMB_inst[2,2,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_pol,noiseTuKArcmin=conf.noiseTuKArcmin_pol,muK=False)

    NL = scal_CL + tens_CL + Nl_qq_EE_diag + Nl_qq_BB_diag + Nl_CMB_inst

    return NL


def get_NLtens_cvlimit():
    #import cmblss
    #import camb
    #import kszpsz_estimator
    #import clprimary

    estim_smallscale_lmax = conf.estim_smallscale_lmax
    estim_smallscale_lmin = conf.estim_smallscale_lmin
    psz_estim_signal_lmin = conf.psz_estim_signal_lmin
    psz_estim_signal_lmax = conf.psz_estim_signal_lmax
    zbins_nr = conf.zbins_nr

    # Calculate scalar signal covariance matrix.
    scal_CL = get_CLscalCMB_cov()

    # Calculate tensor signal covariance matrix.

    tens_CL = get_CLtens_cov()

    # Calculate reconstruction noise.
    #Nl_qq_E, Nl_qq_B = kszpsz_estimator.calc_Nl_qq()

    #Nl_qq_EE_diag = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))
    #Nl_qq_BB_diag = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))
    #
    #for ell_id in range(psz_estim_signal_lmax):
    #    for zbins_id in range(zbins_nr):
    #        Nl_qq_EE_diag[3+zbins_id, 3+zbins_id, ell_id] = Nl_qq_E[zbins_id,ell_id]
    #        Nl_qq_BB_diag[3+conf.N_bins+zbins_id, 3+conf.N_bins+zbins_id, ell_id] = Nl_qq_B[zbins_id,ell_id]

    # Calculate CMB instrumental noise.

    #Nl_CMB_inst = np.zeros((2*conf.N_bins+3,2*conf.N_bins+3,conf.psz_estim_signal_lmax))

    #Nl_CMB_inst[0,0,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_T,noiseTuKArcmin=conf.noiseTuKArcmin_T,muK=False)
    #Nl_CMB_inst[1,1,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_pol,noiseTuKArcmin=conf.noiseTuKArcmin_pol,muK=False)
    #Nl_CMB_inst[2,2,:] = clprimary.Nl_CMB(ellmax=psz_estim_signal_lmax-1,beamArcmin=conf.beamArcmin_pol,noiseTuKArcmin=conf.noiseTuKArcmin_pol,muK=False)

    NL = scal_CL + tens_CL

    return NL




###############################################################################
################                                            ###################
################ BIN CORRELATIONS AND CORRELATION MATRIX    ###################
################                                            ###################
###############################################################################

def CL_bins(T_list1, T_list2, k, L, As=conf.As, ns=conf.ns):
    """
    Compute correlation C_l's using transfer function pairs in T_list1 and T_list2
    assumes T[l_idx, k], not T[k,l].
    """
    CL = np.zeros((len(T_list1),len(L)))

    for i in np.arange(len(T_list1)):
        for l in L:
            T1 = T_list1[i]
            T2 = T_list2[i]

            I = (k**2)/(2*np.pi)**3 * Ppsi(k, As, ns) * np.conj(T1[:,l]) * T2[:,l]

            CL[i,l] = np.real(integrate.simps(I, k))

    return CL

def CL(T1, T2, k=None, L=None, P_psi=None, config=conf):
    """
    Compute correlation C_l's using transfer function pairs in T_list1 and T_list2
    assumes T[l_idx, k], not T[k,l].
    """
    if not L :
        L = L_sampling(config)
    if not k :
        k = k_sampling(config)
    if not P_psi :
        As=config.As
        ns=config.ns
        P_psi = Ppsi(k, As, ns)

    CL = np.zeros(len(L))

    for l in L:
        I = (k**2)/(2*np.pi)**3 * P_psi * np.conj(T1[l]) * T2[l]
        CL[l] = np.real(integrate.simps(I, k))

    return CL

# Bin Correlation matrix

def CL_cov(T_list, k, L, As, ns):

    CL= np.zeros((len(T_list),len(T_list),len(L)))

    for i in np.arange(len(T_list)):

        for j in np.arange(i,len(T_list)):

            for l in L:

                T1= T_list[i]

                T2= T_list[j]

                I= ((k**2)/((2*np.pi)**3))*Ppsi(k, As, ns)*np.conj(T1[:,l])*T2[:,l]

                CL[i,j,l] = np.real(integrate.simps(I, k))

                CL[j,i,l] = CL[i,j,l]

    return CL

######################################################################################################
################                                         ###################
################ CORRELATION MATRIX FOR FISHER FORECAST  ###################
################                                         ###################
######################################################################################################


# Correlation matrix built for fisher forecast

def CL_fisher(N, n, Omega_b, Omega_c, w, wa, Omega_K, h, As, ns, tau):

    CL_fisher = []

    k = k_sampling()

    Lv = np.arange(conf.ksz_estim_signal_lmax)

    Lq = np.arange(conf.psz_estim_signal_lmax)

    LT = np.arange(conf.T_estim_signal_lmax)

    LE = np.arange(conf.E_estim_signal_lmax)

    l_min = np.min([conf.ksz_estim_signal_lmin,conf.psz_estim_signal_lmin, conf.T_estim_signal_lmin, conf.E_estim_signal_lmin])
    l_max = np.max([conf.ksz_estim_signal_lmax,conf.psz_estim_signal_lmax, conf.T_estim_signal_lmax, conf.E_estim_signal_lmax])

    L = np.arange(l_max)


    T_list = []

    for l in L:

        T_list.append([])

    #print("Getting dipole transfer functions ... ")

    for i in np.arange(N):

        Tksz = Transfer_ksz_bin(N, n, i, k, Lv, Omega_b, Omega_c, w, wa, Omega_K, h)

        for u1 in Lv:

            if u1 >= conf.ksz_estim_signal_lmin:

                (T_list[u1]).append(Tksz[:,u1])

    #print("Getting quadrupole transfer functions ... ")

    for i in np.arange(N):

        Tpsz = Transfer_psz_bin(N, i, k, Lq, Omega_b, Omega_c, w, wa, Omega_K, h)

        for u2 in Lq:

            if u2 >= conf.psz_estim_signal_lmin:

                (T_list[u2]).append(Tpsz[:,u2])

    #print("Getting CMB transfer functions ... ")

    TCMB = Transfer_CMB(k, LT , Omega_b, Omega_c, w, wa, Omega_K, h)

    for u3 in LT:

        if u3 >= conf.T_estim_signal_lmin:

            (T_list[u3]).append(TCMB[:,u3])

    #print("Getting E polarization transfer functions ... ")

    TE = Transfer_E(k, LE , Omega_b, Omega_c, w, wa, Omega_K, h, tau)

    for u4 in LE:

        if u4 >= conf.E_estim_signal_lmin:

            (T_list[u4]).append(TE[:,u4])



    for l in L:

        if l >= l_min:

            Transfers = T_list[l]

            CL= np.zeros((len(Transfers),len(Transfers)))

            for i in np.arange(len(Transfers)):

                for j in np.arange(i,len(Transfers)):

                    T1= Transfers[i]

                    T2= Transfers[j]

                    I= ((k**2)/((2*np.pi)**3))*Ppsi(k, As, ns)*np.conj(T1)*T2

                    CL[i,j] = np.real(integrate.simps(I, k))

                    CL[j,i] = CL[i,j]

            CL_fisher.append(CL)

        else:

            CL_fisher.append([])


    return CL_fisher

# Derivatives of the correlation matrix


def CL_fisher_Di(i, N, n, Omega_b, Omega_c, w, wa, Omega_K, h, As, ns, tau):

    step = np.array([Omega_b/100.0, Omega_c/100.0 , w/100.0, 1e-3 ,1e-3, h/100.0, As/100.0, ns/100.0, tau/10.0])

    step_Ob     = np.array(  [step[0],    0     ,   0     ,     0     ,    0     ,    0    ,   0     ,   0    ,    0   ])
    step_Oc     = np.array(  [    0  ,  step[1] ,   0     ,     0     ,    0     ,    0    ,   0     ,   0    ,    0   ])
    step_w      = np.array(  [    0  ,    0     , step[2] ,     0     ,    0     ,    0    ,   0     ,   0    ,    0   ])
    step_wa     = np.array(  [    0  ,    0     ,   0     ,  step[3]  ,    0     ,    0    ,   0     ,   0    ,    0   ])
    step_OK     = np.array(  [    0  ,    0     ,   0     ,     0     , step[4]  ,    0    ,   0     ,   0    ,    0   ])
    step_h      = np.array(  [    0  ,    0     ,   0     ,     0     ,    0     , step[5] ,   0     ,   0    ,    0   ])
    step_As     = np.array(  [    0  ,    0     ,   0     ,     0     ,    0     ,    0    , step[6] ,   0    ,    0   ])
    step_ns     = np.array(  [    0  ,    0     ,   0     ,     0     ,    0     ,    0    ,   0     , step[7],    0   ])
    step_tau    = np.array(  [    0  ,    0     ,   0     ,     0     ,    0     ,    0    ,   0     ,   0    , step[8]])

    CL_fisher1 = CL_fisher(N, n, Omega_b + step_Ob[i], Omega_c + step_Oc[i], w +step_w[i], wa+step_wa[i], Omega_K + step_OK[i], h+step_h[i], As+step_As[i], ns+step_ns[i], tau + step_tau[i])
    CL_fisher2 = CL_fisher(N, n, Omega_b - step_Ob[i], Omega_c - step_Oc[i], w -step_w[i], wa-step_wa[i], Omega_K - step_OK[i], h-step_h[i], As-step_As[i], ns-step_ns[i], tau - step_tau[i])

    nl = len(CL_fisher1)

    CL_fisher_Di = []

    for u in np.arange(nl):

        C_plus  = np.asarray(CL_fisher1[u])
        C_minus = np.asarray(CL_fisher2[u])

        CL_fisher_Di.append((C_plus-C_minus)/(2.0*step[i]))

    return CL_fisher_Di





###############################################################################################
################                      WRAPPER FUNCTIONS                     ###################
###############################################################################################


#
# These are functions the user can call when using fiducial parameters
#

def zbins_z_func():
    # "Get boundaries of bins in comoving distance"
    Chis_bound = np.linspace(
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_min)),
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_max)),
        conf.N_bins+1)

    # "Translate this to redshift boundaries"
    a_guess = np.linspace(az(conf.z_min),az(conf.z_max),conf.N_bins+1)
    sol = optimize.root(root_chi2a, a_guess, args=(Chis_bound, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
    Z = (1/sol.x)-1

    return Z


def zbins_zcentral_func():
    return Z_bin(conf.N_bins)


def zbins_chi_func():
    """Get boundaries of bins in comoving distance"""
    Chis_bound = np.linspace(
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_min)),
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_max)),
        conf.N_bins+1)

    return Chis_bound


def zbins_chicentral_func():
    """Get boundaries of bins in comoving distance"""
    Chis_bound = np.linspace(
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_min)),
        Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(conf.z_max)),
        conf.N_bins+1)

    # "Get positions of bins in comoving distance"
    Chis = np.zeros(conf.N_bins)
    for i in np.arange(0,conf.N_bins):
        Chis[i] = (Chis_bound[i]+Chis_bound[i+1])/2

    return Chis


def get_CLvv():
    """
    Returns the diagonal part of the vv covariance
    matrix as a (N_bin,N_bin,L) array
    """
    print ("Calculating Cl_vv...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)
    T_list = []

    for i in np.arange(conf.N_bins):
        T_list.append(Transfer_ksz_bin(conf.N_bins, conf.n_samples, i, k, Lv,
            conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    CLvv = CL_bins(T_list, T_list, k, Lv, conf.As, conf.ns)
    return CLvv


def get_CLvv_localdopp():
    """
    Returns the diagonal part of the vv covariance matrix as a
    (N_bin,N_bin,L) array using only the local doppler contribution
    """
    print ("Calculating Cl_vv...")
    k = k_sampling()
    Lv = np.arange(conf.ksz_estim_signal_lmax)
    T_list = []

    for i in np.arange(conf.N_bins):
        T_list.append(Transfer_ksz_bin_localDopp(conf.N_bins, conf.n_samples, i, k, Lv,
            conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    CLvv = CL_bins(T_list, T_list, k, Lv, conf.As, conf.ns)
    return CLvv


def get_CLqq():
    """
    Returns the diagonal part of the qq covariance matrix as a (N_bin,N_bin,L) array
    """
    print ("Calculating Cl_qq...")
    k = k_sampling()
    Lq = np.arange(conf.psz_estim_signal_lmax)
    T_list = []

    for i in np.arange(conf.N_bins):
        T_list.append(Transfer_psz_bin(conf.N_bins, i, k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    CLqq = CL_bins(T_list, T_list, k, Lq, conf.As, conf.ns)
    return CLqq


def get_CLvq():
    print ("Calculating Cl_vq...")
    k = k_sampling()
    l_max = np.min([conf.ksz_estim_signal_lmax,conf.psz_estim_signal_lmax])
    Lvq = np.arange(l_max)
    T_list1 = []
    T_list2 = []

    for i in np.arange(conf.N_bins):
        T_list1.append(Transfer_psz_bin(conf.N_bins, i, k, Lvq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    for i in np.arange(conf.N_bins):
        T_list2.append(Transfer_ksz_bin(conf.N_bins, conf.n_samples, i, k, Lvq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    CL_vq = CL_bins(T_list2, T_list1, k, Lvq, conf.As, conf.ns)
    return CL_vq


def get_CLTT():
    print ("Calculating Cl_TT...")
    k = k_sampling()
    LT = np.arange(conf.T_estim_signal_lmax)
    T_list = [Transfer_CMB(k, LT , conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)]

    CL_TT = CL_bins(T_list, T_list, k, LT, conf.As, conf.ns)
    return CL_TT


def get_CLEE():
    print ("Calculating Cl_EE...")
    k = k_sampling()
    LE = np.arange(conf.E_estim_signal_lmax)
    T_list = [Transfer_E(k, LE , conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.tau)]

    CL_EE = CL_bins(T_list, T_list, k, LE, conf.As, conf.ns)
    return CL_EE


def get_CLTv():
    print ("Calculating Cl_Tv...")
    k = k_sampling()
    l_max = np.min([conf.ksz_estim_signal_lmax,conf.T_estim_signal_lmax])
    LTv = np.arange(l_max)

    T_list1 = []
    T_list2 = []

    TCMB = Transfer_CMB(k, LTv , conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)

    for i in np.arange(conf.N_bins):
        T_list1.append(TCMB)
        print("Done bin "+str(i))


    for i in np.arange(conf.N_bins):
        T_list2.append(Transfer_ksz_bin(conf.N_bins, conf.n_samples, i, k, LTv, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))
        print("Done bin "+str(i))

    CL_Tv = CL_bins(T_list2, T_list1, k, LTv, conf.As, conf.ns)
    return CL_Tv


def get_CLvv_cov():

    k = k_sampling()

    Lv = np.arange(conf.ksz_estim_signal_lmax)

    T_list = []

    for i in np.arange(conf.N_bins):

        T_list.append(Transfer_ksz_bin(conf.N_bins, conf.n_samples, i, k, Lv, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

        print("Done bin "+str(i))

    CLvv_cov = CL_cov(T_list, k, Lv, conf.As, conf.ns)

    return CLvv_cov

def get_CLqq_cov():

    k = k_sampling()

    Lq = np.arange(conf.psz_estim_signal_lmax)

    T_list = []

    for i in np.arange(conf.N_bins):

        T_list.append(Transfer_psz_bin(conf.N_bins, i, k, Lq, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h))

        print("Done bin "+str(i))

    CLqq_cov = CL_cov(T_list, k, Lq, conf.As, conf.ns)

    return CLqq_cov



def get_CL_fisher(N, n):

    return CL_fisher(N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_Ob(N, n):

    return CL_fisher_Di(0, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_Oc(N, n):

    return CL_fisher_Di(1, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_w(N, n):

    return CL_fisher_Di(2, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_wa(N, n):

    return CL_fisher_Di(3, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_OK(N, n):

    return CL_fisher_Di(4, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_h(N, n):

    return CL_fisher_Di(5, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_As(N, n):

    CL = CL_fisher(N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

    l_min = np.min([conf.ksz_estim_signal_lmin,conf.psz_estim_signal_lmin, conf.T_estim_signal_lmin, conf.E_estim_signal_lmin])

    L = np.arange(len(CL))

    for l in L:

        if l >= l_min:

            CL[l] = CL[l]/conf.As

    return CL

def get_CL_fisher_ns(N, n):

    return CL_fisher_Di(7, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)

def get_CL_fisher_tau(N, n):

    return CL_fisher_Di(8, N, n, conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h, conf.As, conf.ns, conf.tau)



#Globals for use in other modules:
zbins_nr = conf.N_bins
zbins_z = zbins_z_func()
zbins_zcentral = zbins_zcentral_func()
zbins_chi = zbins_chi_func()
zbins_chicentral = zbins_chicentral_func()
