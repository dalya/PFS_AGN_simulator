import numpy
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.io import fits


from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.interpolate import interp1d,make_interp_spline, BSpline
from scipy.stats import truncnorm
from scipy.integrate import simps
from scipy.stats import median_abs_deviation

import astropy.units as u
from pyphot import Filter
from pyphot import unit
from sedpy.smoothing import smoothspec

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

h=0.7
H0=70.0 # km s-1 Mpc-1
c=299792 #km s-1
gama=1.86
Omega_m=0.3
Omega_v=0.7
sigma_8=0.84
def E(z):
    return np.sqrt(Omega_m*pow(1+z,3)+Omega_v)
def V_ele(z):
    return c/H0*cosmo.luminosity_distance(z).value*cosmo.luminosity_distance(z).value/E(z)/(1+z)/(1+z) #unit Mpc^3

#####################################################################################################################

###################### BH mass from stellar mass

def return_BH_mass(log_M_stars):
    """
    Function takes as an input stellar mass and returns the BH mass
    The function is based on the BH mass - stellar mass relation by Mcconel & Ma and includes the scatter as well
    """
    log_M_BH = 8.46 + 1.05*(log_M_stars - 11.) + np.random.normal(loc=0, scale=0.34, size=len(log_M_stars))
    return log_M_BH

###################### Eddington ratio using the ERDF

aird17=np.loadtxt('data/pledd_all.dat')

def ERDF(zmin, zmax, msmin, msmax):
    edd=[aird17[ee,4] for ee in range(len(aird17)) if aird17[ee,0]==zmin \
         and aird17[ee,1]==zmax and aird17[ee,2]==msmin and aird17[ee,3]==msmax]
    pEdd=[aird17[ee,5] for ee in range(len(aird17)) if aird17[ee,0]==zmin \
         and aird17[ee,1]==zmax and aird17[ee,2]==msmin and aird17[ee,3]==msmax]

    return interpolate.interp1d(edd,pEdd,kind='linear')


def return_logEdd_ratio(z_vector, logM_vector):
    """
    Function assigns the Eddington ratio of each galaxy given its redshift and stellar mass.
    The function uses the ERDF given by Aird et al. 2017, which is divided into different redshift and stellar mass bins.
    Everything outside the redshift range 0.5 - 1.5 and stellar mass range 9 - 11.5 gets clipped to this range, and 
    the assignment is according to the closest redshift and stellar mass bin.
    The assignment of the Eddington ratios results in populations that obey the ERDFs of different bins.
    """
    x = np.arange(-4, 1.55, 0.1) 
    # start by clipping the arrays to the ranges I am going to consider
    offset = 0.01
    z_copy = np.copy(z_vector)
    z_copy = np.clip(z_copy, 0.5 + offset, 1.5 - offset)
    
    logM_copy = np.copy(logM_vector)
    logM_copy = np.clip(logM_copy, 9 + offset, 11.5 - offset)
    
    # array to be filled with values
    indices_all = np.arange(len(z_copy))
    gal_logEdd = np.zeros(len(z_copy))
    
    # limits of the ERDF
    z_limits = [0.5, 1, 1.5]
    logM_limits = [9, 9.5, 10, 10.5, 11, 11.5]

    for i in range(len(z_limits)-1):
        for j in range(len(logM_limits)-1):
            z_min = z_limits[i]
            z_max = z_limits[i+1]
            logM_min = logM_limits[j]
            logM_max = logM_limits[j+1]
            
            # ERDF for this array
            interp_func = ERDF(z_min, z_max, logM_min, logM_max)
            y = interp_func(x)
            y[x > 1] = 0
            y_norm = y / np.sum(y)
            
            # identify the galaxies within this range
            indices_in_range = indices_all[(z_copy > z_min) & (z_copy <= z_max) & (logM_copy > logM_min) & (logM_copy <= logM_max)]
            k_range = len(indices_in_range)
            
            # randomly-draw from the ERDF that corresponds to this redshift and mass range
            gal_logEdd[indices_in_range] = choices(x, y_norm, k=k_range)
            
    return gal_logEdd

###################### AGN bolometric luminosity and X-ray luminosity

def return_logLbol(log_M_BH, log_Edd):
    """
    function takes as an input the BH mass and the Eddington ratio (both in logarithmic scale) and 
    returns the AGN bolometric luminosity.
    """
    L_bol = (10**log_Edd)*1.26e38*(10**log_M_BH)
    return np.log10(L_bol)

def return_logLx(log_L_bol):
    """
    Function converts the bolometric luminosity to X-ray luminosity according 
    to the bolometric corrections by Duras et al. (2020)
    """
    a = np.random.normal(10.85, 0.08)
    b = np.random.normal(11.90, 0.01)
    c = np.random.normal(19.93, 0.29)
    
    Kx = a * (1 + ((log_L_bol - np.log10(3.846e33))/b)**c)
    
    log_L_x = log_L_bol - np.log10(Kx)
    return log_L_x

###################### Duty cycle

xxx=[4.12638,4.63472,6.09508,7.00679,7.18427,7.30527,7.45851,7.59561,7.72460,7.84553,\
     7.99062,8.10344,8.23238,8.36133,8.49838,8.63546,8.75644,8.88553,9.05498,9.17603,9.33744,\
     9.46652,9.59552,9.70830,9.78871,9.86105,9.89304,10.49348,10.78155,11.5]
yyy=[-0.52009,-0.52991,-0.60841,-0.66729,-0.68692,-0.70654,-0.74579,-0.78505,-0.84393,-0.90280,\
     -0.98131,-1.05981,-1.14813,-1.22664,-1.29533,-1.34439,-1.37383,-1.38364,-1.38364,-1.37383,-1.36402,\
     -1.37383,-1.43271,-1.53084,-1.67804,-1.82523,-1.97243,-4.44533,-5.70140,-10]
duty_AS=interpolate.interp1d(xxx,yyy,kind='linear')

def dduty(log_M_BH):
    if log_M_BH>=5 and log_M_BH<=11:
        return 10**duty_AS(log_M_BH)
    else: 
        return 0

def return_duty_cycle(log_M_BH, log_L_x):
    """
    Function returns the AGN duty cycle within the galaxy, given the BH mass and the X-ray luminosity
    """
    gal_duty=np.zeros_like(log_M_BH)

    indices_all = np.arange(len(log_M_BH))

    for log_mbh in np.arange(5,10.5,0.2):
        indices_tot = indices_all[(log_M_BH >= log_mbh) & (log_M_BH < log_mbh + 0.2)]
        indices_can = indices_all[(log_M_BH >= log_mbh) & (log_M_BH < log_mbh + 0.2) & (log_L_x > 40)]
        #print(type(indices_can), len(indices_can))

        duty_tem = round(dduty(log_mbh)*len(indices_tot))

        if len(indices_can) >= duty_tem:
            indexx = np.random.choice(indices_can, duty_tem, replace=False)
            gal_duty[indexx] = 1

        elif (len(indices_can) < duty_tem) and (len(indices_can) >= 0):
            gal_duty[indices_can] = 1

    return gal_duty

###################### Obscuration and AGN type

def obsfrac(L2_10):
    """
    function returns the obscuring fraction for the given X-ray luminosity
    """
    frac = 0.56 + np.arctan((43.89-L2_10)/0.46)/np.pi
    if frac>0.985:frac=0.985
    #if L2_10<43: frac=0.985
    return frac

def return_AGN_type(gal_duty, log_L_x):
    """
    function returns the type of the AGN given the duty cycle and given the X-ray luminosity
    """
    gal_type=[]
    for ii in range(len(log_L_x)):
        if gal_duty[ii]==0:
            gal_type.append(0)
            continue

        rand_num=np.random.uniform(0,1,1)
        if rand_num < obsfrac(log_L_x[ii]):
            gal_type.append(2)
        else:
            gal_type.append(1)

    gal_type = np.array(gal_type)
    return gal_type