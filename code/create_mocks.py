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

###################################################################################################################
############################################## Type I mocks  ######################################################
###################################################################################################################

def return_MgII_to_FeII_conversion_factor(log_edd_ratio):
    """
    Function returns the conversion factor between MgII and FeII flux according to
    the relation found by Shin et al. 2021. The input is log(L/LEdd)
    """
    a, b = 0.35, 0.68
    if log_edd_ratio < -3:
        return (a * -3) + b
    elif log_edd_ratio > 0:
        return (a * 0) + b
    else:
        return (a * log_edd_ratio) + b

def return_alpha_opt_distribution():
    """
    Function returns the distribution of alpha_nu_opt, where F_nu ~ nu^alpha_nu_opt from the Shen et al
    quasar catalog. The array will be used to later sample the continuum slope in optical wavelengths
    in the type I mock construction.

    NOTE: it has been decided that we will not use this function and will only simulate blue quasars, 
    because our goal is to add them to the host galaxy spectrum, which will redden them.
    """
    path = "/Users/dalyabaron/Documents/PFS/code/data/shen_alpha_opt.fit"
    h = fits.open(path)
    data = h[1].data
    alpha_lambda = data['alpHb']
    indices = numpy.arange(len(alpha_lambda))
    indices_clean = indices[~numpy.isnan(alpha_lambda)]

    alpha_nu = -1 * (alpha_lambda[indices_clean] + 2)
    alpha_sample = alpha_nu[(alpha_nu > -5) & (alpha_nu < 1)]
    return alpha_sample

def bol2L_iso(L_bol):
    """
    Function takes as an input the AGN bolometric luminosity (linear units) and returns
    several monochromatic luminosities in certain wavelengths. 
    """
    L_1350=L_bol/3.81/1350. #richards +06
    L_1450=L_bol/4.2/1450. #runnoe+12
    L_3000=L_bol/5.2/3000. #runnoe+12
    L_5100=L_bol/8.1/5100. #runnoe+12
    
    cons=4. * np.pi * pow((10. * u.pc).to(u.cm),2)
    M_1450=-2.5*np.log10(1450*1450/3e18*L_1450/cons.value)-48.6

    return L_1450, M_1450, L_1350, L_3000, L_5100

def continuum(Lbol, cons):
    """
    Function constructs the continuum emission of the type I mock. It returns the wavelength in Ang and the 
    flux density in units of erg/sec/cm^2/Ang. The input to the function is the AGN bolometric luminosity 
    (in linear units) and the conversion factor between luminosity and distance (which depends on the redshift)
    """
    L_1350=bol2L_iso(Lbol)[2]
    f1350=L_1350/cons
    
    wave=np.arange(1,8000,1)
    alpha1=np.random.normal(1.76,0.1) # Telfer+2002
    #alpha2=-1*np.random.choice(alpha_sample, 1)[0] # alpha_opt from Shen et al SDSS quasar catalog, includes red quasars
    alpha2=-1*np.random.normal(0.44,0.3) # only blue quasars from Vanden Berk+2001
    alpha3=2.45 # slope of the geometric mean from Vanden Berk+2001
    ratio1=pow(3e18/1100.,-alpha1)/pow(3e18/1100.,-alpha2)
    ratio2=pow(3e18/5000.,-alpha2)*ratio1/pow(3e18/5000.,-alpha3)
    Fv=[pow(3e18/aa,-alpha1) for aa in wave if aa<1100]+[pow(3e18/aa,-alpha2)*ratio1 for aa in wave if aa>=1100 and aa<5000]+\
       [pow(3e18/aa,-alpha3)*ratio2 for aa in wave if aa>=5000]
    Flam=np.array([3e18*Fv[oo]/wave[oo]/wave[oo] for oo in range(len(wave))])
    Flam_normed=f1350/Flam[1349]*Flam
    
    return wave, Flam_normed

def objective(x, a, b):
    return a * x + b

def Fline(Lbol,flux_):
    """
    Function takes as an input the AGN bolometric luminosity and the flux density of the continuum.
    It uses the Baldwin effect to estimate EW(CIV) using Niida+2016. Using the continuum flux density,
    the function converts EW(CIV) to flux. It then uses the emission lines measured in Vanden Berk+2001
    and scales their flux to show similar flux ratios with respect to the CIV. 
    """
    M1450 = bol2L_iso(Lbol)[1]
    Mag_i_z2=M1450-1.486
    EW_CIV=-1
    ew_mean=pow(10, 0.074*Mag_i_z2+3.503)
    ew_std=a_ew*Mag_i_z2+b_ew
    
    while EW_CIV<0:
        EW_CIV=np.random.normal(ew_mean,ew_std,1)[0]
    
    F_civ=EW_CIV*flux_[1545]
    F_line=[['CIV', 1546.15, float(van01[0,8]), F_civ]]
    
    for i in range(1,len(van01)):
        Lratio=float(van01[i,4])/float(van01[0,4])
        if Lratio > 0.001:        
            F_line.append([van01[i,-2], float(van01[i,-1]), float(van01[i,8]), Lratio*F_civ])
            
    return F_line

def F_FeII(F_line,logLedd):
    """
    Function takes as an input the list of the different emission lines and log(L/LEdd).
    It extracts the flux of the MgII line, and then uses the correlation between log(FeII/MgII) and log(L/LEdd)
    to estimate the coversion factor between the flux of MgII and the flux of the FeII in the wavelength
    range 2200-3090. It scales the Iron template so that its integral between 2200 and 3090 will result 
    in the correct FeII flux, given the assumed MgII flux.
    """
    f_mg2=[item[3] for item in F_line if item[0]=="MgII"]
    conv_factor = return_MgII_to_FeII_conversion_factor(logLedd)
    F_ratio=f_mg2[0]*pow(10,conv_factor) 
    return flam_Fe2*F_ratio

def mbh2fwhm(mbh,lll,typ):
    """
    Function takes as an input the BH mass (in linear scale), a luminosity value (also in linear scale),
    and a flag corresponding to the line of interest, and calculate the FWHM of the line of interest. 
    """
    if typ=='civ':
        fwhm=1000*pow(10, 0.5*(np.log10(mbh) - 6.66 - 0.53*np.log10(1350*lll/1e44) )) #km/s
        return fwhm/3e5*1546.15 #AA
    
    if typ=='mg2':
        fwhm=1000*pow(10, 0.5*( np.log10(mbh) - 6.748 - 0.62*np.log10(3000*lll/1e44) )) #km/s
        return fwhm/3e5*2798.75 #AA
    
    if typ=='Hb':
        fwhm=1000*pow(10, 0.5*( np.log10(mbh) - 6.91 - 0.5*np.log10(5100*lll/1e44) )) #km/s
        return fwhm/3e5*4862.68 #AA
    
    if typ=='Ha':
        fwhm=1000*pow(10, 1/2.12*( np.log10(mbh) - 6.71 - 0.48*np.log10(lll/1e42) )) #km/s
        return fwhm/3e5*6564.61 #AA

def Ffwhm(mbh, flux_, Fline, cons):
    """
    Function takes as an input the BH mass, the continuum flux density, the list of line fluxes, and 
    the conversion factor between luminosity and flux (which depends on the redshift). It returns the 
    FWHM of the emission lines CIV, MgII, Hbeta, and Halpha. These values are consistent with scaling
    relations between the BH mass, line FWHM, and continuum flux (single epoch BH mass estimation).
    """
    L1350_=flux_[1349]*cons
    L3000_=flux_[2999]*cons
    L5100_=flux_[5099]*cons
    f_ha=[item[3] for item in Fline if item[0]=="Hα"]
    Lha=f_ha[0]*cons
    fwhm_civ = mbh2fwhm(mbh, L1350_, 'civ')
    fwhm_mg2 = mbh2fwhm(mbh, L3000_, 'mg2')
    fwhm_hb = mbh2fwhm(mbh, L5100_, 'Hb')
    fwhm_ha = mbh2fwhm(mbh, Lha, 'Ha')
    return fwhm_civ, fwhm_mg2, fwhm_hb, fwhm_ha

#add gaussian
def gaussian(x, mu, sig):
    norm=integrate.quad(lambda xx:np.exp(-pow(xx - mu, 2.) / (2 * pow(sig, 2.))),mu-200,mu+200)[0]
    return np.exp(-pow(x - mu, 2.) / (2 * pow(sig, 2.)))/norm 

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def aymGau(mean,std_up,std_low):
    temIGM=[]
    for uu in range(len(mean)):
        tem=np.random.uniform(-1,1)
        if tem>0:temIGM.append(get_truncated_normal(mean[uu],std_up[uu],mean[uu],1).rvs(1))
        else: temIGM.append(get_truncated_normal(mean[uu],std_low[uu],0,mean[uu]).rvs(1))
    return np.concatenate(temIGM)

def myFilter(filterName,w,f):
    b=np.loadtxt('/Users/dalyabaron/Documents/PFS/code/data/Filter/'+str(filterName)+'.dat')
    fil=Filter(b[:,0], b[:,1], name='newfilt', dtype='photon', unit='Angstrom')
    flux=fil.get_flux(np.array(w)*unit['AA'], np.array(f))
    ABmag=-2.5*np.log10(flux)-fil.AB_zero_mag
    return ABmag

####################################################################################################################
############################### Final functions to produce type I mock #############################################
####################################################################################################################

# Baldwin effect to convert M_i to EW(CIV)
Mi=np.arange(-30.5,-23,1)
EW_err=[6.67, 10.8, 11.1 ,13.6 ,18.0 ,23.2 ,24.5, 34.8]
popt, _ = curve_fit(objective, Mi,EW_err)
a_ew, b_ew = popt

# Load the Vanden-Berk+ 2001 line list
van01=np.loadtxt('/Users/dalyabaron/Documents/PFS/code/data/Vanden01_lineRatio', dtype='str')

# Load Iron template
FeII=np.loadtxt('/Users/dalyabaron/Documents/PFS/code/data/FeII_Mix_Int.dat')
Fe2=interp1d(FeII[:,0], FeII[:,1], kind='linear')
F_fe2=integrate.quad(lambda xx:Fe2(xx),2200,3090)[0]
flam_Fe2=np.array([Fe2(ww)/F_fe2 for ww in np.arange(1,8000,1)])

# Load distirbution of continuum slope in optical wavelengths
alpha_sample = return_alpha_opt_distribution()

def makeQSO(mbh, Lbol, z, logLedd, return_redshifted=True):
    """
    This is the main function producing the type I mock. The inputs are: the BH mass in linear scale, 
    the AGN bolometric luminosity in linear scale, the redshift, and log(L/LEdd).
    """
    # conversion of luminosity to flux given redshift
    cons = 4. * np.pi * pow((cosmo.luminosity_distance(z)).to(u.cm).value,2)
    # construct continuum emission
    w_r, f_r = continuum(Lbol,cons)
    # get line list
    f_line = Fline(Lbol,f_r)
    # get FWHM of strongest broad lines
    F_civmg2hbha = Ffwhm(mbh,f_r,f_line,cons)
    
    # normalize and convolve the Iron template, then add it to the continuum
    sigma_MgII_km_s = ((F_civmg2hbha[1]/2.355)/2798.75)*3e5
    FeII_template = F_FeII(f_line,logLedd)
    FeII_template_conv = smoothspec(w_r, FeII_template, resolution=sigma_MgII_km_s, smoothtype="vel")
    f_r += FeII_template_conv
    
    # update the FWHM of the CIV, MgII, Hbeta, and Halpha, according to the BH mass
    for uu in range(len(f_line)):
        if f_line[uu][0]=='CIV':
            f_line[uu][2]=F_civmg2hbha[0]/2.355  #fwhm to sigma
            
        if f_line[uu][0]=='MgII':
            f_line[uu][2]=F_civmg2hbha[1]/2.355
            
        if f_line[uu][0]=='Hβ':
            f_line[uu][2]=F_civmg2hbha[2]/2.355
            
        if f_line[uu][0]=='Hα':
            f_line[uu][2]=F_civmg2hbha[3]/2.355
            
    # add Gaussians corresponding to all the emission lines
    for uu in range(len(f_line)):
        low=int(f_line[uu][1])-4*int(f_line[uu][2])
        high=int(f_line[uu][1])+4*int(f_line[uu][2])
        if low<min(w_r):low=min(w_r)
        if high>max(w_r):high=max(w_r)

        f_r[low-1:high-1] += [f_line[uu][3]*gaussian(xx, f_line[uu][1], f_line[uu][2]) for xx in np.arange(low, high, 1)]

    # add IGM attenuation according to Inoue+14
    igm_inoue=np.loadtxt('/Users/dalyabaron/Documents/PFS/code/data/IGM/z_'+str(int(z*10)))
    temp=aymGau(igm_inoue[:,1],igm_inoue[:,2],igm_inoue[:,3])
    model_IGM=interp1d(igm_inoue[:,0],temp,kind='linear')
    loww=round(min(igm_inoue[:,0]))+1
    highh=round(max(igm_inoue[:,0]))-1
    f_r[:loww]=np.nan
    for i in range(loww,highh):
        f_r[i]=f_r[i]*model_IGM(w_r[i])

    # redshift the wavelength
    w_o=w_r*(1+z)
    f_o=f_r/(1+z)
    
    if not return_redshifted:
        return w_r, f_r, f_line
    else:
        return w_o, f_o, f_line

###################################################################################################################
############################################## Type II mocks  #####################################################
###################################################################################################################

def return_typeII_template(wavelength_interp):
    """
    Function returns the type II AGN template by Yuan et al. (2017), interpolated
    to the wavelength given in the input (presumabely the wavelength of the galaxy templates).
    """
    h = fits.open("Type2s_composite.fits")
    data = h[1].data

    wavelength = data['wv']
    flux_norm = data['comp_norm'] - 1 # the flux is normalized to have a continuum of 1
    flux_final = numpy.copy(flux_norm)
    flux_final[numpy.abs(flux_final) < 0.05] = 0
    flux_final[flux_final < 0] = 0
    flux_final[wavelength < 2000] = 0
    flux_final[wavelength > 8000] = 0
    flux_interp = numpy.interp(wavelength_interp, wavelength, flux_final)

    
    # normalize the template so that the integral over [OIII] will be 1
    wavelength_small = wavelength_interp[(wavelength_interp >= 4980) & (wavelength_interp <= 5045)]
    flux_small = flux_interp[(wavelength_interp >= 4980) & (wavelength_interp <= 5045)]
    OIII_integ = simps(flux_small, wavelength_small)
    
    flux_norm = flux_interp / OIII_integ
    
    return flux_norm

def return_logLx(logLbol):
    """
    Function converts the bolometric luminosity to X-ray luminosity according 
    to the bolometric corrections by Duras et al. (2020)
    """
    a = numpy.random.normal(10.85, 0.08)
    b = numpy.random.normal(11.90, 0.01)
    c = numpy.random.normal(19.93, 0.29)
    
    Kx = a * (1 + ((logLbol - numpy.log10(3.846e33))/b)**c)
    
    logLx = logLbol - numpy.log10(Kx)
    return logLx

def return_logLOIII(logLx):
    """
    Function converts the X-ray luminosity into [OIII] luminosity using the relation
    given by Lamastra et al. (2009).
    """
    b1 = numpy.random.normal(1.11, 0.10)
    b2 = numpy.random.normal(1.02, 0.06)
    logLOIII = 42 + ((logLx - 42 - b1)/b2)
    return logLOIII


####################################################################################################################
############################### Final functions to produce type II mock ############################################
####################################################################################################################

wavelength_interp = numpy.linspace(1800, 10000, 10000)
flux_norm_ = return_typeII_template(wavelength_interp)

def return_typeII_emission_lines(logLbol, cons):
    """
    Function returns the type II emission line spectrum (flux density), normalized
    so that the [OIII] luminosity is consistent with the AGN luminosity
    """
    logLx = return_logLx(logLbol)
    logLOIII = return_logLOIII(logLx)
    OIII_flux = 10**logLOIII / cons
    
    emission_spec = flux_norm_ * OIII_flux
    
    return emission_spec
