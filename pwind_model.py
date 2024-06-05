#######################################################################################################################################

## Module holding the python version of Manuel Lopez-Puertas model
## Coding : utf-8
## Author : Adrien Masson (adrien.masson@obspm.fr)
## Date   : February 2023

# equivalent to manuel_model class but for pwinds
# this model uses an already computed abundance/velocity profile (computed either with Lampon's model or pwinds hydrodynamique code)
# and do the transmission spectrum computation

#######################################################################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------
#<f Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import astropy.units as u
import datetime
import time
from astropy import constants as const
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.functional_models import Voigt1D
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from astropy import constants as const
import os
from p_winds import parker, hydrogen, helium, transit, lines, tools
from PyAstronomy import pyasl
from scipy import optimize

# some global variables
cpu_count = os.cpu_count()-1 # will use all available cpu except 1 by default, change here if you want to use less/more
#f>

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : pwind_model.py has been loaded')

def find_nearest(array, value):
    ''' find index of the closest element to the given value in an array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def lsf_carmenes(lbda,x):
    '''
    Compute CARMENES' LSF depending on the wavelength (CARMENES has a wavelength dependent LSF)
    Supplied to Manuel Lopez-Puertas by Evangelos
    '''
    # lbda in micron, x in microns around lambda (-> nb of points used for the convolution ?)
    # output : ils_vgt
    # x -> les longueurs d'ondes (en microns) auxquelles échantillonner la réponse instrumentale autour du lbda central

    # NIR FWHM (DeltaL/L) supplied by Evangelos
    res_gauss=1.18e-5 # constant for Doppler part in Voigt
    res_lorentz=1.7e-6 # constant for Lorentz part in Voigt

    # ad checked with Evangelos
    ad=res_gauss*lbda/ (2.*np.sqrt(np.log(2)))
    al=res_lorentz*lbda/2.
    a= (al/ad)
    u = (x/ad)
    v1 = Voigt1D(a)
    ils_vgt=(1./(ad*np.sqrt(np.pi)))*v1(u)
    ils_vgt=ils_vgt/np.max(ils_vgt)

    return(ils_vgt)

class Pw_He_model:
    #<f Documentation
    '''
    TO DO !
    '''
    #f>

    def __init__(self, params, transit, verbose=True):
        # verbose
        self.verbose            = verbose # print log ?

        # Set up the simulation: keep same grid parameters as Granada's model

        # Fixed parameters for the model
        self.params = params

        self.dnu = params['dnu']
        # same as manuel_model
        if self.params['air']:
            self.xmin=1.08270
            self.xmax=1.08330
            self.xminn=1.0828
            self.xmaxx=1.08320
        #Vacuum
        else:
            self.xmin= 1.0828
            self.xmax= 1.0838
            self.xminn=1.0830 # xmin for plotting
            self.xmaxx=1.0835 # xmax for plotting
        self.tr_np = int((self.xmax-self.xmin)/self.dnu)      # nb of wavelength points
        self.wn = self.xmin+np.arange(self.tr_np)*self.dnu # wavelength vector corresponding to the transmission model

        self.R_pl = params['r_planet']*1e3 /const.R_jup.value # Planetary radius (jup radii)
        R_s  = params['r_star']*1e3 / const.R_jup.value # stellar radii (jup radii)
        M_pl = transit['Mp'] / const.M_jup.value  # Planetary mass (Jupiter masses)
        a_pl = transit['a'] / const.au.value      # Semi-major axis (au)
        self.planet_to_star_ratio = self.R_pl / R_s
        self.impact_parameter = params['b']              # impact parameter

        self.transit_dic = transit

        h_fraction = params['hfrac']/100                      # H number fraction
        he_fraction = 1 - h_fraction                # He number fraction
        he_h_fraction = he_fraction / h_fraction
        mean_f_ion = 0.90                           # Initially assumed, but the model relaxes for it
        mu_0 = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + mean_f_ion)  # mu_0 is the constant mean molecular weight (assumed for now, will be updated later)

        # Physical constants
        m_h = const.m_p.to(u.g).value  # Hydrogen atom mass in g
        self.m_He = 4 * 1.67262192369e-27  # Helium atomic mass in kg
        k_B = 1.380649e-23  # Boltzmann's constant in kg / (m / s) ** 2 / K

        # Model settings
        relax_solution = True  # This will iteratively relax the solutions until convergence
        exact_phi = True  # Exact calculation of H photoionization

        w0, w1, w2, f0, f1, f2, a_ij = lines.he_3_properties()
        self.w_array = np.array([w0, w1, w2])  # Central wavelengths of the triplet
        self.f_array = np.array([f0, f1, f2])  # Oscillator strengths of the triplet
        self.a_array = np.array([a_ij, a_ij, a_ij])  # This is the same for all lines in then triplet

        # First guesses of fractions (not to be fit, but necessary for the calculation)
        initial_f_ion = 0.0  # Fraction of ionized hydrogen
        initial_f_he = np.array([1.0, 0.0])  # Fraction of singlet, triplet helium

        if self.verbose: print('Model succesfully parametrised')

    def compute_kepler_orbit(self):
        '''
        Uses pyasl.KeplerEllipse to compute the planetary elliptical orbit and store it as an attribute
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py)
        more information on pyasl.KeplerEllipse definition : https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/keplerOrbitAPI.html

        position and velocity can then be access with respect to time using self.ke.xyzPos(time) & self.xyzVel(time) for a given time.
        The unity corresponds to those provided in the transit dictionnary, e.g if transit['a'] is in m and transit['Porb'] in s, coordinates will be in m & velocity in m/s

        In order to align the origin of time for the KeplerEllipse orbit, we compute the time of mid transit corresponding to the model and store it in self.model_midpoint.
        Thus, time from mid transit can be convert from data_set time to keplerEllipse time using self.time_from_mid+self.model_midpoint
        '''

        # Set the model
        a = self.transit_dic['a'] # m
        per = self.transit_dic['Porb'] # BJDTBD
        e = self.transit_dic['e']
        Omega = self.transit_dic['lbda'] # °
        w = self.transit_dic['w'] # ° , in case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see above documentation)
        i = self.transit_dic['i'] # °
        # the KeplerEllipse object is stored as an attribute
        self.ke = pyasl.KeplerEllipse(a=a, per=per, e=e, Omega=Omega, w=w, i=i, ks=pyasl.MarkleyKESolver)
        # find the mid-transit time corresponding to the models, defined as the time at which the modeled position is the closest to the stellar center. This time is found usnig scipy.minimize
        def f(time):
            pos = self.ke.xyzPos(time)
            r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.transit_dic['Rs']
            return(r)
        # time in BJDTBD
        time = np.linspace(0,per,100)
        pos = self.ke.xyzPos(time)
        # find model midpoint i.e corresponding to the minimal distance to the stellar surface in the xy plan
        # only select point during primary transit
        r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.transit_dic['Rs'] # distance from center of star during primary transit in stellar radii
        m = np.logical_and(pos[:,2]<=0, r<=1.0) # z negative = planet in front of star, sqrt(x**2+y**2)<Rs = planet crossing stellar surface
        result = optimize.minimize(f, x0=time[m][m.sum()//2])
        self.model_midpoint = result.x # BJDTBD

    def custom_draw_transit(self,grid_size,time_from_mid,plot=False):
        '''
        Simulate the stellar surface in a grid with pixels set as 0. behind the transiting planetary disk for a given time from mid transit.
        The pixels are normalized such that the sum of all pixels equals 1. when off-transit
        The planet trajectory is directly read in the DataSet object provided as argument: it has been precomputed by pyasl.KeplerEllipse method, taking into account eccentricity & spin-orbit alignement.
        The stellar grid also takes into account non-linear limb darkening if it has been provided to the transit_info dictionnary during class initialization

        Takes:
        - grid_size : the size of the stellar grid, in pixels.
        - time_from_mid : time from mid transit, in days (BJD-TBD), at which computing the planet position
        - plot : bool, set to True to show the grid

        Returns:
        - flux_map : a grid of size (grid_size,grid_size), containing the normalized flux (sum of all pixels equals 1. when off transit, pixels are set to 0. outside stellar disk & behind planetary disk)
        - transit_depth : the transit_depth due to the opaque planetary disk (at 1Rp) at given time
        - r_map : a grid for which each pixels contain the distance, in m, of the pixel to the planetary center
        '''

        def limb_dark(mu):
            # see if the limb dark law is uniform or not by reading the transit_dic
            if len(self.transit_dic['c'])==4:
                c1,c2,c3,c4 = self.transit_dic['c']
                limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4
                limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2))
            elif len(self.transit_dic['c'])==0:
                limb = np.zeros(mu.shape)
                limb[np.isfinite(mu)] = 1.
            elif len(self.transit_dic['c'])==2:
                c1, c2 = self.transit_dic['c']
                limb = 1-c1*(1-mu)-c2*(1-mu)**2
            else:
                raise NameError(f'Limb dark function with coeff {self.transit_dic["c"]} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')
            return(limb)

        self.compute_kepler_orbit()

        position = self.ke.xyzPos(time_from_mid+self.model_midpoint) # position array from keplerEll computation
        x_planet = position[:,0]/self.transit_dic['Rs'] # x axis position, in stellar radii, of the planet in the grid
        y_planet = position[:,1]/self.transit_dic['Rs'] # y axis position, in stellar raddi, of the planet in the grid

        # set the grid: we define everything in stellar radii, so the grid extend to -1 to +1 stella radii in both x & y axis
        X = np.linspace(-1.0, 1.0, grid_size)
        Y = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(X, Y)

        r_map = np.sqrt((X-x_planet)**2 + (Y-y_planet)**2)*self.transit_dic['Rs'] # distance of each cell from planet center, in m

        r_s = np.sqrt(X**2+Y**2) # grid containing the distance of each pixel to the stellar center (at (0,0)). Used for building the stellar disk
        alpha = np.arcsin(r_s)
        mu = np.cos(alpha) # limb dark angle
        limb = limb_dark(mu) # the limb-darkening intensity map of the star

        flux_map = np.copy(limb) # fill stellar disk with limb darkening as read in the transit dictionnary
        flux_map[~np.isfinite(flux_map)] = 0. # fill the outside of stellar surface with 0.

        flux_map /= flux_map.sum() # pwinds requires stellar flux map normalized by off transit total intensity of the star
        intensity_off = flux_map.sum() # total off-transit sum of the cell's grid. should be very close to 1.

        dx = self.transit_dic['Rs']/grid_size # size of a pixel, in m
        flux_map[r_map<=(self.transit_dic['Rp']+dx)] = 0. # we add a "dx", which is the size of a pixel, to also incorporate pixels which are not fully covered by the planetary disk

        intensity_on = flux_map.sum() # compute the total intensity during the transit

        transit_depth = intensity_off-intensity_on # compute transit depth due to planet surface (at 1Rp) at this given time of the transit

        # show transit map
        if plot:
            plt.figure()
            plt.imshow(flux_map,origin='lower')
            plt.colorbar()
            plt.plot(x_planet,y_planet,'r+',label=f'planet center at T = {time_from_mid}')

        return flux_map, transit_depth, r_map

    def compute_model(self,zlev,n_he_3_distribution, v_array,plot=False):
        '''
        Compute the transmission spectra of He metastable line at given wavelength_array and for given profiles
        wavelength_array -> the wavelength (in m) at which computing the spectra
        - zlev : the layers level (in planet radii) used to sample the altitude grid. The grid must start at the planet surface (at 1Rp) and with the orgine at planet "center" (i.e the lowest layer in the grid is 1Rp and correspond to an altitude of 1Rp)
        - v_wind -> a velocity (km/s) to doppler shift the spectra
        - n_he_3_distribution -> vertical abundance profile of the metastable He triplet (cm^-3)
        - Vp -> array of planet orbital velocity in stellar RF, projected along line of sight, in m/s
        - time_samples -> an array containing the time from mid transit, in days (BJD-TBD) at which the transit sampling will be computed and averaged.

        '''

        if self.verbose:print('\n----------- Computing Model -----------')
        t0 = time.time()
        n = 1.00027394 # air refractive index at 1 micron (from https://refractiveindex.info/)
        if self.params['air']:
            wl_transm = self.wn/1e6 # convert to m
        else:
            wl_transm = self.wn/1e6/n # convert to m & air

        # Set up the transit configuration. We use SI units to avoid too many
        # headaches with unit conversion
        R_pl_physical = self.R_pl * const.R_jup.value # Planet radius in m
        r_SI = zlev * self.params['r_planet'] * 1e3      # Array of altitudes in m, define with respect to planet center (thus start at 1Rp)
        v_SI = v_array * 1000      # Velocity of the outflow in m / s
        v_wind = self.params['vshift']*1e3 # convert to m/s
        n_he_3_SI = n_he_3_distribution * 1E6  # Volumetric densities in 1 / m ** 3

        # Set up the ray tracing
        f_maps = []
        t_depths = []
        r_maps = []

        n_samples = len(self.params['time_samples'])

        for i in range(n_samples):
            flux_map, transit_depth, r_map = self.custom_draw_transit(self.params['transit_grid_size'],self.params['time_samples'][i],plot=plot)

            f_maps.append(flux_map)
            t_depths.append(transit_depth)
            r_maps.append(r_map)

        # Do the radiative transfer
        spectra = []

        # whether to compute wind broadening using average wind velocity profile or altitude dependent wind velocity value
        if self.params['rad_vel']:
            wind_broadening_method = 'formal'# + lent mais + rigoureux
        else:
            wind_broadening_method = 'average'

        turb_broad = self.params['fac_d'] > 1. # if a turbulence broadening factor is defined in paramters: let pwinds compute the broadening factor and apply it


        for i in range(n_samples):
            spec = transit.radiative_transfer_2d(f_maps[i], r_maps[i],
                                            r_SI, n_he_3_SI, v_SI, self.w_array, self.f_array, self.a_array,
                                            wl_transm, self.params['t_d'], self.m_He, bulk_los_velocity=v_wind,# wavelength must be in air for the transmission computation
                                                wind_broadening_method=wind_broadening_method,
                                                turbulence_broadening = turb_broad)

            # We add the transit depth because ground-based observations
            # lose the continuum information and they are not sensitive to
            # the loss of light by the opaque disk of the planet, only
            # by the atmosphere
            spectra.append(spec + t_depths[i])



        spectra = np.array(spectra)
        # Finally we take the mean of the spectra we calculated for each phase
        self.trm = np.mean(spectra, axis=0)
        # if the spectra is full of nan, replace by a zero spectrum to avoid error in chi2 maps
        if np.all(np.isnan(self.trm)): self.trm = np.zeros(self.tr_np)
        t1 = time.time()
        if self.verbose:
            print(f'Done in {(t1-t0)//60:.0f}min {(t1-t0)%60:.0f}sec')
            print(f'---------------------------------------')

    def convolve_CARMENES(self, plot=True):
        '''
        convolve the computed model with the CARMENES instrumental function
        '''

        # Convolve with CARMENES spectra
        grid=self.dnu
        xmaxx0=8e-5
        xminn0=-xmaxx0
        nk=np.round(2*xmaxx0/grid)+1
        x = xminn0+np.arange(nk)*grid

        # output grid
        wmin=self.wn[0]+1e-4 #Reduced range to avoid problems in the convolution at the edges
        wmax=self.wn[-1]-1e-4

        grido=1.0e-8      # microns: Output grip
        nwo=int(np.round((wmax-wmin)/grido)+1.)
        self.wno=wmin+np.arange(nwo)*grido
        self.tro=np.zeros(nwo)

        # Loop over all output wns
        for i in range(nwo):
            i0,a=find_nearest(self.wn,self.wno[i])
            if np.abs(self.wn[i0] - self.wno[i]) > np.abs(self.wn[i0+1] - self.wno[i]) : i0=i0+1
            i1=np.max([0,i0-nk/2])
            i2=np.min([i0+nk/2,len(self.wn)-1])
            trr=self.trm[int(i1):int(i2)] # int() ??? see L376 & 377 of manuel's IDL code
            # Calculate the ILS at the given wavelength
            kernel=lsf_carmenes(self.wno[i],x)

            trx=convolve(trr,kernel,normalize_kernel=True)
            self.tro[i]=trx[int(nk/2)] # int() ???

        if plot:
            plt.figure()
            plt.plot(self.wn,self.trm,label='Transmission')
            plt.plot(self.wno,self.tro,label='After CARMENES convolution')
            plt.legend()

        if self.verbose: print(f'Convolution with CARMENES instrumental function done !')

    def convolve_SPIRou(self, plot=True):
        '''
        convolve the computed model with the SPIRou instrumental function
        '''
        pixel_size = 2*2.28e3 # SPIRou element resolution in m/s
        nb_points = 11 # size, in pixel, of the door function used for the convolution

        half_size = pixel_size / 2
        pixel = np.linspace(-half_size,half_size,nb_points)

        convolved_spec = np.zeros(self.trm.size)

        f = interp1d(self.wn,self.trm,fill_value=np.nan)

        for v in pixel:
            # mask wavelength shifted outside the interpolation domain
            mask_down = (self.wn / (1 + v/const.c.value)) < self.wn.min()
            mask_up   = (self.wn / (1 + v/const.c.value)) > self.wn.max()
            mask = np.logical_or(mask_down,mask_up) # contains True where shifted wavelength are outside the valid interpolation domain
            convolved_spec[~mask] += f(self.wn[~mask] / (1 + v/const.c.value))
            # replace values outside range by nan
            convolved_spec[mask] = np.nan

        # normalise
        convolved_spec /= len(pixel)

        # cut invalid values and store
        mask = np.isfinite(convolved_spec)
        self.wno = self.wn[mask]
        self.tro = convolved_spec[mask]

        if plot:
            plt.figure()
            plt.plot(self.wn,self.trm,label='Transmission')
            plt.plot(self.wno,self.tro,label='After SPIRou convolution')
            plt.legend()

    def chi2(self,data,error,use_convolved=True,plot=True):
        '''
        return the chi2 value computed between this model and some data
        if use_convolved=True, chi2 use computed using the convolved model (self.tro)
        TO DO : let user specify the mask used for chi2 ?
        data must have:
            - wavelength in first column in same units as model
            - normalised transmission in second column
        error must have same shape as data and contains the error on each data point

        if plot=True: plot data Vs model
        '''
        data_wave = data[:,0] # wavelength of the data
        # mask for chi2 computation
        if self.params['air']: chi2_mask = (data_wave > 1.0828)*(data_wave < 1.08315)
        else: chi2_mask = (data_wave > 1.0831)*(data_wave < 1.08345)
        self.chi2_mask = chi2_mask

        x_data = data[chi2_mask,1]
        x_error = error[chi2_mask]
        # interpolate model on data wavelength
        if use_convolved: self.x_model = interp1d(self.wno,self.tro)(data_wave)
        else: self.x_model = interp1d(self.wn,self.trm)(data_wave)
        x_model = self.x_model[chi2_mask]
        # compute chi2
        Chi2 = np.sum((x_data-x_model)**2/x_error**2)

        if plot:
            plt.figure()
            plt.errorbar(data[:,0],data[:,1],yerr=error,label='data',fmt='.-k')
            if use_convolved:plt.plot(self.wno,self.tro,label='model convolved',color='r')
            else: plt.plot(self.wn,self.trm,label='model (not convolved)',color='r')
            if self.params['air']:
                for w in [10829.0911,10830.2501,10830.3398]:
                    plt.vlines(w*1e-4,0.9854,1.005,color='k',ls='dotted')
            else:
                for w in [10832.057 ,10833.217, 10833.307]:
                    plt.vlines(w*1e-4,0.9854,1.005,color='k',ls='dotted')
            plt.hlines(1,self.xminn,self.xmaxx,ls='dotted',color='k')
            # plt.legend()
            plt.xlim(self.xminn,self.xmaxx)
            plt.xlabel("Wavelength [microns]")
            plt.ylabel('Normalized flux')

        return Chi2















































#
