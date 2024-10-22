"""
VERSION 1.0.2, released 10/22/2024

PREVIOUS VERSION: 1.0.1, released 10/07/2024

NEW FEATURES:

BUG FIXES:
    fixed error in align() method where whichorders variable was
    called before it was defined. Thanks Krishna for finding this.

"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.constants as const
import pandas as pd
import copy
import pickle
from scipy import interpolate
from scipy.interpolate import griddata
from scipy import constants
from scipy.optimize import curve_fit, minimize
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as query
import pdb
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.ndimage import gaussian_filter1d


class Planet:
    '''
    IGRINS cubifying. I'm thinking about this now so you don't have to.
    Steps: 1. get_params
           2. cubify
           3. plot
           4. trim
           5. align
           6. do_pca
           4. save

    If your planet is unpublished and/or there isn't a good reference on
    the exoplanet archive, you can set your own params as attributes as
    the Planet object. They are:
    period - period in days
    T0 - time of mid transit in BJD. Can be any or most recent
    RV - radial velocity of star in km/s
    RA - system right ascension
    DEC - system declination
    '''

    def __init__(self, planetname, path):
        self.planetname = planetname
        self.path = path

    def get_params(self, ref=None, get_RV=False, source='tepcat'):
        """
        Get orbital and positional parameters necessary for calculating
        the orbtial phase and radial velocity of planet-star system.
        Will either read this from TEPCat or NEA. TEPCat is default, and
        if you want the NEA set source = 'nea'. Getting the stellar RV is 
        optional and by default is set to 0.
        """
        name = self.planetname

        #if course if TEPCat, which is the default
        if source == 'tepcat':
            #read tables from URL
            RA, DEC, Tdur, T0, period, RV = self.read_TEPCAT(ref, get_RV)

        elif source == 'nea':
            period, T0, RV, RA, DEC, Tdur = self.read_NEA(ref, get_RV)

        self.period = period
        self.T0 = T0
        self.RV = RV
        self.RA = RA 
        self.DEC = DEC
        self.Tdur = Tdur

        return period, T0, RV, RA, DEC, Tdur

    def read_NEA(self, ref, get_RV):
        '''
        Use Astroquery to query the NASA Exoplanet Archive so you don't have to look it up yourself.
        If there is a specific reference you want to use, put the first author's last name as the optional
        ref kwarg. Gives the period, T0, RV, RA, and DEC attributes to your planet object. Alternatively, if you already
        have these params, you can manually give these attributes to your planet object and not run this function.
        '''
        name = self.planetname #name of the planet. Make sure it is in the proper format recognized by the Archive.
        params = query.query_criteria(table='ps',select="pl_orbper, pl_refname, pl_tranmid, st_radv, st_refname, rastr, decstr, pl_trandur", where="pl_name like '{}'".format(name))
        if ref:
            # mask = (ref in params['pl_refname']) ## FIX THIS
            mask = [i for i, refs in enumerate(params['pl_refname']) if ref in refs]
        else:
            mask = np.isfinite(params['st_radv']) #take sources that had a system RV.
            print('Taking values from',params['pl_refname'][mask][0])
            if True not in mask:
                print(params['pl_refname'])
                ref = input('Please specify reference:')
                mask = (ref in params['pl_refname']) ## FIX THIS
                
        period = params['pl_orbper'][mask][0].value #orbital period in days
        T0 = params['pl_tranmid'][mask][0].value #time of mid transit in BJD
        RV = params['st_radv'][mask][0].value #system velocity in km/s
        RA = params['rastr'][mask][0] #right ascension. string
        DEC = params['decstr'][mask][0] #declincation. string
        Tdur = params['pl_trandur'][mask][0].value #transit duration in hour

        if get_RV == False:
            RV = 0.

        return period, T0, RV, RA, DEC, Tdur

    def read_TEPCAT(self, ref, get_RV):
        """
        Read in the params we want from TEPCat. TEPCat doesn't list
        stellar RV so if you want to read it in from NEA, set get_RV=True.
        It will take the first RV it will find, or you can specific a source.
        Also, you can mannually set it. Otherwise, it will be set to 0 by
        default, in which case the Rvel.pic file will just be the barycentric
        velocity.
        """
        name = self.planetname

        if 'WASP' in name:
            wasp, letter = name.split()
            wasp, number = wasp.split('-')
            if len(number) == 2:
                number = '0'+number
            name = wasp+'-'+number +' ' + letter
            # if len(name) == 9:
            #     name = name[:-4] +'0' + name[-4:]
            # elif len(name) == 10:
            #     name = name[:-5] + '0' + name[-5:]
        if 'HD' in name:
            name = name[:2] + '_' + name[3:]

        if 'A b' in name:
            url = 'https://www.astro.keele.ac.uk/jkt/tepcat/planets/' + name[:-3] +'.html'
        else:
            url = 'https://www.astro.keele.ac.uk/jkt/tepcat/planets/' + name[:-2] +'.html'
        try:
            dfs = pd.read_html(url)
        except: 
            print('URL not found. Make sure the planet name is in the right format. \nOtherwise, set source="nea" or set attributes manually.')
            return None

        RA, DEC, Tdur, T0, period = dfs[0].loc[[3,4,9,11,12]]['Value']

        Tdur = Tdur.split('(')[1]
        Tdur = float(Tdur.split(')')[0])
        
        T0 = float(T0.split('±')[0])
        period = float(period.split('±')[0])

        a, Rp, Mp, Rstar = dfs[1].loc[[7,9,8,3]]['Value']
        a = self.split_string(a)
        Rp = self.split_string(Rp)
        Mp = self.split_string(Mp)
        Rstar = self.split_string(Rstar)
        print('Rp:',Rp, 'Mp:',Mp, 'Rstar:',Rstar)
        
        Kp = 2. * np.pi * a * u.au / period / u.day
        Kp = Kp.to(u.km / u.s)
        
        print('Kp:', Kp)

        #if you want to read in an RV, we'll have to use astroquery to get if
        #from NEA bc TEPCAT doesn't list it 
        if get_RV == True:
            params = query.query_criteria(table='ps',select="st_refname, st_radv", where="pl_name like '{}'".format(self.planetname))
            if ref:
                mask = [i for i, refs in enumerate(params['st_refname']) if ref in refs]
            else:
                mask = np.isfinite(params['st_radv']) #take sources that had a system RV.
                print('Taking values from',params['st_refname'][mask][0])
            RV = params['st_radv'][mask][0].value
        else:
            RV = 0.

        return RA, DEC, Tdur, T0, period, RV

    def split_string(self, string):
        string = string.split('±')
        if len(string) == 1:
            string = string[0].split('+')
        return float(string[0])

    def initialize(self):
        '''
        Initialize some orbital params and stuff for a planet object
        Inputs: 
            planet: object

        Outputs:
            wlgrid: wavelength grid
            T0: time of mid transit in BJD
            sc: SkyCoord object for the target
            gems: EarthLocation object of the observatory
        '''
        filesH = sorted(glob.glob(self.path+'SDCH*.spec.fits'))
        filesK = sorted(glob.glob(self.path+'SDCK*.spec.fits'))

        '''Construct Wavelength grid'''
        last_frameH =  filesH[-1]
        last_frameK = filesK[-1]

        wlfits = fits.open(last_frameH)
        wlgridH = wlfits[1].data
        wlfits.close()

        wlfits = fits.open(last_frameK)
        wlgridK = wlfits[1].data
        wlfits.close()

        wlgrid = np.concatenate([wlgridK, wlgridH])

        '''Initialize Orbital params'''
        first_frame = filesH[0]
        start = fits.open(first_frame)
        start_time = Time(start[0].header['JD-OBS'], format='jd')
        location = start[0].header['TELESCOP']

        # radec = start[0].header['USERRA'] + '\t' + start[0].header['USERDEC']
        radec = self.RA + '\t' + self.DEC 

        gems = EarthLocation.of_site(location)
        sc = SkyCoord([radec], unit=(u.hourangle, u.deg))

#         T0 = Time(self.T0, format='isot', scale='utc')
        T0 = Time(self.T0, format='jd') #time of mid transit in BJD
        # timecorrection = start_time.light_travel_time(sc,'barycentric',gems)
        # 12/16 COMMENTING THIS OUT FOR TESTING - KEEP T0 IN BJD
        # T0 -= timecorrection #time of mid transit in JD
        start.close()

        return wlgrid, T0, sc, gems

    def cubify(self):
        '''
        Inputs:
            Planet object
        Outputs:
            wlgrid: array of dim Ndet x Npix
            cube: array of dim Ndet x Nphi x Npix
            ph_arr: orbital phases
            Rvel: radial velocity
        '''
        wlgrid, T0, sc, gems = self.initialize()

        '''Lets start cubin'''
        filesH = sorted(glob.glob(self.path+'SDCH*.spec.fits'))
        filesK = sorted(glob.glob(self.path+'SDCK*.spec.fits'))

        '''SNR'''
        snrfilesH = sorted(glob.glob(self.path+'SDCH*.sn.fits'))
        snrfilesK = sorted(glob.glob(self.path+'SDCK*.sn.fits'))

        '''Variance'''
        varfilesH = sorted(glob.glob(self.path+'SDCH*.variance.fits'))
        varfilesK = sorted(glob.glob(self.path+'SDCK*.variance.fits'))

        Nphi = len(filesH) #number of frames
        Ndet, Npix = wlgrid.shape #orders, wavelength channels

        '''Initialize arrays'''
        data_raw = np.zeros((Ndet, Nphi, Npix))
        snr_raw = np.zeros((Ndet, Nphi, Npix))
        var_raw = np.zeros((Ndet, Nphi, Npix))
        ph_arr = []
        rvel_arr = []
        time_arr = []
        am_arr = [] #air mass
        hum_arr = [] #humidity
        Texp_arr = []
        for i in range(Nphi):
            #### SNR ####
            hdu = fits.open(snrfilesH[i])
            snr_H = hdu[0].data
            hdu.close()

            hdu = fits.open(snrfilesK[i])
            snr_K = hdu[0].data
            hdu.close()

            snr = np.concatenate([snr_K, snr_H])
            snr_raw[:,i] = snr

            #### VARIANCE ####
            hdu = fits.open(varfilesH[i])
            var_H = hdu[0].data 
            hdu.close()

            hdu = fits.open(varfilesK[i])
            var_K = hdu[0].data 
            hdu.close()

            var = np.concatenate([var_K, var_H])
            var_raw[:,i] = var

            #### DATA ####
            hdu = fits.open(filesH[i])
            Texp = hdu[0].header['EXPTIME']# * u.s #exposure time seconds
            Texp_arr.append(Texp)
            image_dataH = hdu[0].data
            hdu.close()

            hdu = fits.open(filesK[i])
            image_dataK = hdu[0].data

            am_start = hdu[0].header['AMSTART']
            am_end = hdu[0].header['AMEND']
            am = np.mean([am_start, am_end])
            am_arr.append(am)

            hum = hdu[0].header['HUMIDITY']
            hum_arr.append(hum)

            frame_start = hdu[0].header['JD-OBS']
            frame_end = hdu[0].header['JD-END']
            
            t1 = Time(frame_start, format='jd')
            t2 = Time(frame_end, format='jd')

            # frame_time = t1 + 0.5*Texp #"average" time of the A frame
            frame_time = Time(0.5 * (t1.jd + t2.jd), format='jd') #average time of AB frame

            #convert to BJD to compare to mid transit time
            timecorrection = frame_time.light_travel_time(sc,'barycentric', gems)
            frame_time_bjd = frame_time.tdb + timecorrection
            time_arr.append(frame_time_bjd.value[0])
            # ph = (frame_time_bjd.tdb.value - T0.tdb.value)/self.period % 1
            ph = (frame_time_bjd.value - T0.value)/self.period % 1
            baryvelcorr = sc.radial_velocity_correction(obstime=frame_time,location=gems)
            ph_arr.append(ph)
            rvel_arr.append(baryvelcorr)

            hdu.close()

            data = np.concatenate([image_dataK,image_dataH])
            data_raw[:,i,:] = data

        ph_arr = np.asarray(ph_arr)
        ph_arr[ph_arr > 0.8] -= 1.
        rvel_arr = np.asarray(rvel_arr)
        rvel_arr *= -1e-3 #km/s
        rvel_arr += self.RV
        ph_arr = ph_arr.reshape(ph_arr.shape[0])
        rvel_arr = rvel_arr.reshape(rvel_arr.shape[0])
        time_arr = np.asarray(time_arr)
        time_arr = time_arr.reshape(time_arr.shape[0])
        am_arr = np.asarray(am_arr)
        am_arr = am_arr.reshape(am_arr.shape[0])
        hum_arr = np.asarray(hum_arr)
        hum_arr = hum_arr.reshape(hum_arr.shape[0])
        Texp_arr = np.asarray(Texp_arr)
        Texp_arr = Texp_arr.reshape(Texp_arr.shape[0])

        self.wlgrid = wlgrid
        self.data_raw = data_raw
        self.snr_raw = snr_raw
        self.var_raw = var_raw
        self.ph_arr = ph_arr
        self.rvel_arr = rvel_arr
        self.time_arr = time_arr
        self.am_arr = am_arr
        self.hum_arr = hum_arr
        self.Texp_arr = Texp_arr


        # check if any frames are during eclipse #
        ph_transit = self.Tdur / self.period / 24.
        in_eclipse = (  abs(ph_arr  - 0.5) < 0.5*ph_transit ) #frames in eclipse
        if in_eclipse.sum() > 0:
            print('WARNING: '+str(in_eclipse.sum()) +' frames are during eclipse!')

        return wlgrid, data_raw, snr_raw, var_raw, ph_arr, rvel_arr, time_arr, am_arr, hum_arr, Texp_arr
    
    def plot(self, order=20):
        plt.imshow(self.data_raw[order], 
            extent=[self.wlgrid[order][0], self.wlgrid[order][-1], self.ph_arr[0], self.ph_arr[-1]],origin='lower',aspect='auto')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('Phase')
        plt.show()
       
    def trim(self, transit_only=False, low_index = None, high_index=None):
        '''
        Trim off some frames from the beginning or end of the sequence.
        Set transit_only = True to remove out-of-transit frames.
        '''
        if transit_only == True:
            # discard the out of transit frames
            ph_transit = self.Tdur / self.period / 24. #fraction of orbit in transit
            in_transit = (abs(self.ph_arr) <= 0.5*ph_transit) #mask for in transit
            self.ph_arr = self.ph_arr[in_transit]
            self.rvel_arr = self.rvel_arr[in_transit]
            self.time_arr = self.time_arr[in_transit]
            self.data_raw = self.data_raw[:,in_transit]
            self.snr_raw = self.snr_raw[:,in_transit]
            self.var_raw = self.var_raw[:, in_transit]
            self.am_arr = self.am_arr[in_transit]
            self.hum_arr = self.hum_arr[in_transit]
            self.Texp_arr = self.Texp_arr[in_transit]

        if low_index == None:
            low_index = int(input('Number frames to discard from beginning:'))
        if high_index == None:
            high_index = int(input('Number of frames to discard from end:'))

        if high_index != 0:
            data_raw = self.data_raw[:,low_index:-high_index]
            snr_raw = self.snr_raw[:, low_index:-high_index]
            var_raw = self.var_raw[:, low_index:-high_index]
            ph_arr = self.ph_arr[low_index:-high_index]
            rvel_arr = self.rvel_arr[low_index:-high_index]
            time_arr = self.time_arr[low_index:-high_index]
            am_arr = self.am_arr[low_index:-high_index]
            hum_arr = self.hum_arr[low_index:-high_index]
            Texp_arr = self.Texp_arr[low_index:-high_index]
        else:
            data_raw = self.data_raw[:,low_index:]
            snr_raw = self.snr_raw[:, low_index:]
            var_raw = self.var_raw[:, low_index:]
            ph_arr = self.ph_arr[low_index:]
            rvel_arr = self.rvel_arr[low_index:]
            time_arr = self.time_arr[low_index:]
            am_arr = self.am_arr[low_index:]
            hum_arr = self.hum_arr[low_index:]
            Texp_arr = self.Texp_arr[low_index:]
        
        self.data_raw = data_raw
        self.snr_raw = snr_raw
        self.var_raw = var_raw
        self.ph_arr = ph_arr
        self.rvel_arr = rvel_arr
        self.time_arr = time_arr
        self.am_arr = am_arr 
        self.hum_arr = hum_arr
        self.Texp_arr = Texp_arr

        print('N frames:', str(data_raw.shape[1]))


    def stretched(self, wl, wl0, shift, stretch):
        '''
        Given a raw wavelength array, calculates the shifted and stretched
        version given shift and stretch nuisance parameters. Then calculates
        the new flux array by spline interpolating the data (global variable) onto the
        new stretched wavelength array
        Inputs: self - planet object
                wl - raw wavelength array
                wl0 - center wavelength of stretching (assuming it's symmetric)
                shift - additive term to shift wavelength 
                stretch - multiplication term to stretch wavelength
        Output: data_int - raw data interpolated onto shifted and stretched wavelength grid
        '''
        wl1=shift+(wl-wl0)*stretch
        data_int=interpolate.splev(wl1,cs_data,der=0)
    
        return data_int

    def correct(self, wl_arr, data_arr, normed=False, which='last'):
        '''
        Wavelength correct raw data to the last frame in the sequence, which is wavelength
        calibrated to telluric lines.
        Inputs: self - planet object
                wl_arr - raw wavelength grid, shape Ndet x Npix
                data_arr - raw data, shape Ndet x Nphi x Npix
        Output: wl_arr - shifted wavelength grid
                data_corrected - data interpolated onto the shifted wavelength grid
        '''
        data_corrected = np.zeros(data_arr.shape)
        Ndet, Nphi, Npix = data_arr.shape
        #correct to last or first frame. Look at recipes
        if which == 'last':
            ii = -1
        elif which == 'first':
            ii = 0

        corrections = np.zeros((Ndet, Nphi)) #corrections in wl solution

        for order in range(Ndet):
            control = data_arr[order,ii,:].copy() #last frame
            control /= control.max() #normalize the control
            wl_raw = wl_arr[order,]
            for frame in range(Nphi):
                data_to_correct = data_arr[order,frame,]
                maximum = data_to_correct.max()
                data_to_correct /= maximum #normalize
                global cs_data
                cs_data = interpolate.splrep(wl_raw, data_to_correct,s=0.0)
                try:
                    popt, pconv = curve_fit(self.stretched, wl_raw, control, p0=np.array([np.median(wl_raw),np.median(wl_raw),1.]), maxfev = 5000)
                    data_stretched = self.stretched(wl_raw, *popt)
                    corrections[order, frame] = popt[0]
                except RuntimeError:
                    data_stretched = data_to_correct
                    corrections[order, frame] = 0.
                if normed == False:
                    data_stretched *= maximum #un-normalize
                data_corrected[order, frame,] = data_stretched

        return wl_arr, data_corrected, corrections

    def align(self, whichorders='snr', normed=False, which='last', interpolation=False, min_snr=50.):
        '''
        Trim and wavelength calibrate the raw data.
        Inputs: self - planet object
                whichorders - list or array of orders to keep
        Outputs: wlgrid - corrected wavelength grid
                 data - corrected data
        '''

        data = self.data_raw 
        wlgrid = self.wlgrid
        med_snr_per_order = self.med_snr_per_order

        if whichorders == None:
            whichorders=[ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18,19,20,27,28, 29, 30, 31, 32, 33,34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,48,49,50,51]

        else:
            whichorders = (med_snr_per_order > min_snr) #which orders have a median SNR above some set minimum

        data = data[whichorders, :, 100:-100]
        wlgrid = wlgrid[whichorders, 100:-100]
        var_raw = self.var_raw[whichorders, : , 100:-100]

        if interpolation == True:
            print('Doing wavelength realignment. This takes a minute because we are interpolating over nans...')
            data[data < 0.] = np.nan #set negative values to nan
            nanmask = np.isnan(data) #where the nans at

            #interpolate over nans
            Ndet, Nphi, Npix = data.shape 
            x = np.arange(Npix)
            y = np.arange(Nphi)
            X, Y =  np.meshgrid(x,y)

            for order in range(Ndet):
                if nanmask[order].sum() > 0:
                    data[order] = griddata((X[~nanmask[order]], Y[~nanmask[order]]), data[order, ~nanmask[order]], (X,Y), method='cubic')
                    #if something is still nan, set it to a small value
                    stillnan = np.isnan(data[order])
                    data[order][stillnan] = 1e-3

            ## do the same for the variance ##
            var_raw[var_raw < 0.] = np.nan 
            var_nanmask = np.isnan(var_raw)

            Ndet, Nphi, Npix = var_raw.shape 
            x = np.arange(Npix)
            y = np.arange(Nphi)
            X, Y = np.meshgrid(x, y)

            for order in range(Ndet):
                if var_nanmask[order].sum() > 0.:
                    var_raw[order] = griddata((X[~var_nanmask[order]], Y[~var_nanmask[order]]), var_raw[order, ~var_nanmask[order]], (X,Y), method='cubic')
                    stillnan = np.isnan(var_raw[order])
                    #poisson estimate if still nan
                    var_raw[order][stillnan] = data[order][stillnan] / 4. #np.sqrt( data[order][stillnan] ) / 2.

        else:
            data[np.isnan(data)] = 0.
            data[data < 0.] = 0.
            nanmask = np.isnan(data)

            var_raw[np.isnan(var_raw)] = 0.
            var_raw[var_raw < 0.] = 0.
            var_nanmask = np.isnan(var_raw)
        # data[nanmask] = 1e-3 #set to small value for fitting, etc.
        
        wlgrid, corrected_data, corrections = self.correct(wlgrid, data, normed, which)

        corrected_data[corrected_data < 0.] = 1e-3#np.nan #idk why anything would be negative, but it shouldn't...

        # corrected_data[nanmask] = np.nan #add the nans back in

        self.nanmask = nanmask #save this for later
        self.var_nanmask = var_nanmask

        self.wlgrid = wlgrid
        self.data = corrected_data

        self.snr = self.snr_raw[whichorders, :, 100:-100]

        self.variance = var_raw

        self.corrections = corrections #you can plot this later if you want...

        return wlgrid, data


    def do_svd(self, NPC = 4, interpolation=False):
        '''
        Inputs: cube (arg) - a numpy array of dimension 3
                NPC (kwarg) - integer

        Outputs: pca_data - a numpy array of dimension 3
                 pca_scale - a numpy array of dimension 3

        Perform PCA on a data cube. Returns two numpy arrays
        of the same shape as the input cube. The first will
        be the input cube with the first N PCs removed. The
        second will be the input cube with the first N PCs 
        remaining and all others removed.
        '''
        cube = self.data 

        Ndet, Nphi, Npix = cube.shape
        pca_data =  np.zeros(cube.shape)
        pca_scale = np.zeros(cube.shape)
        singular_values = np.zeros((Ndet, Nphi))

        if interpolation == True:
            print('Doing SVD. This takes a minute because we are interpolating over nans...')
            nanmask = np.isnan(cube) #where the nans at
            x = np.arange(Npix)
            y = np.arange(Nphi)
            X, Y = np.meshgrid(x,y) #meshgrid for interpolation purposes
            for order in range(Ndet):
                #if there are nans, interpolate over them so can do SVD
                if nanmask[order].sum() > 0:
                    cube[order] = interpolate.griddata((X[~nanmask[order]], Y[~nanmask[order]]), cube[order,~nanmask[order]],(X,Y), method='cubic')
                    stillnan = np.isnan(cube[order])
                    cube[order][stillnan] = 1e-3
                    # cube[order,nanmask[order]] = np.nanmedian(cube[order, :,nanmask[order][1]]) #median of column
        for order in range(Ndet):

            u,s,vh = np.linalg.svd(cube[order,],full_matrices=False) #decompose
            idx = np.argsort(s, axis=0)[::-1]
            singular_values[order] = s[idx]
            s1 = s.copy()
            s1[NPC:] = 0. 
            W1 = np.diag(s1)
            A1 = np.dot(u, np.dot(W1, vh)) #recompose
            pca_scale[order,] = A1 

            s[0:NPC] = 0. #remove first N PCs
            W = np.diag(s)
            A = np.dot(u, np.dot(W,vh))
            #sigma clip
            sig = np.nanstd(A)
            med = np.nanmedian(A)
            loc = (A > 5. * sig+med)
            A[loc] = np.nan 
            loc = (A < -5. * sig+med)
            A[loc] = np.nan

            A -= np.nanmean(A, axis=1)[:, np.newaxis] #mean subtract each frame
            # A[np.isnan(A)] = 0.
            A[nanmask[order]] = np.nan #add all the other nans back in

            pca_data[order,] = A

        new_nanmask = np.isnan(pca_data) #where the nans at
        old_nanmask = self.nanmask

        nanmask = old_nanmask + new_nanmask

        self.pca_data = pca_data
        self.pca_scale = pca_scale
        self.NPC = NPC 
        self.singular_values = singular_values
        self.nanmask = nanmask

        return pca_data, pca_scale


    def do_sysrem(self, niter=4):
        """
        Do sysrem per Tamuz 2005.
        """
        cube = self.data 
        variance = self.variance
        am = self.am_arr
        Ndet, Nphi, Npix = cube.shape
        pca_data =  np.zeros(cube.shape)
        pca_scale = np.zeros(cube.shape)

        for i in range(Ndet):
            dat = cube[i].copy()
            med = np.nanmedian(dat, axis=1) #median in time

            rij = dat / med[:,np.newaxis] #divide by median spectrum
            invvar = 1/variance[i] #inverse variance

            #save the basis vectors and weights to create a scale matrix
            ccube = np.zeros((Nphi, niter))
            acube = np.zeros((Npix, niter))

            #first pass, use air mass as the initial guess of "systematic"
            ci = am.copy()
            aj = np.ones(Npix)

            ccube[:,0] = ci 
            acube[:,0] = aj 

            rij -= np.outer(ci, aj)

            for j in range(niter - 1):
                ci = np.dot(rij * invvar, aj) / np.dot(invvar, aj**2)
                aj = np.dot(ci, rij * invvar) / np.dot(ci**2, invvar)

                ccube[:, j+1] = ci 
                acube[:,j + 1] = aj

                rij -= np.outer(ci, aj)

            pca_data[i] = rij
            pca_scale[i] = (ccube @ acube.T) * med[:, np.newaxis]

        self.pca_data = pca_data
        self.pca_scale = pca_scale
        self.NPC = niter 

        return pca_data, pca_scale


    def do_mlr_pca(self, standardize=False, NPC=None, coefdeter=0.68, operation='divide', intercept=False, log=False):
        dat = self.data 
        if log == True:
            dat = np.log(dat)
        airmass = self.am_arr 
        humidity = self.hum_arr
        time = self.time_arr 
        variance = self.variance

        wl_data = self.wlgrid 

        ### taking this out for now ###
        # variance_per_res_element = np.zeros_like(variance)

        # for i in range(variance.shape[0]):
        #     factor = wl_data[i] / np.gradient(wl_data[i]) / 45000
        #     variance_per_res_element[i]= variance[i] / factor


        nanvar = np.isnan(variance)
        variance[nanvar] = dat[nanvar] / 4. #just assume poisson noise where there are nans

        mlrcorr = AutoMLRPCA(dat, airmass, humidity, time, variance, standardize=standardize, NPC=NPC, operation=operation, intercept=intercept)

        if 'mask_arr' in vars(self):
            mlrcorr.mask = self.mask_arr

        scale_matrix, post_pca_data, where_corr, sigma, basis, vectors = mlrcorr()

        if log == True:
            scale_matrix = np.exp(scale_matrix)
            post_pca_data = np.exp(post_pca_data)
            sigma = np.exp(sigma)
            vectors = np.exp(vectors)

        self.pca_data = post_pca_data 
        self.pca_scale = scale_matrix
        self.std_arr = np.sqrt(sigma)
        if NPC == None:
            self.NPC = 'auto'
        self.where_corr = where_corr
        self.vectors = vectors

        return scale_matrix, post_pca_data, where_corr, sigma, basis, vectors


    def plot_scree(self):
        """
        Plot scree and variance plots for each principal component.
        """
        Ndet, Nphi, Npix = self.data.shape 

        tot_variance = np.cumsum(self.singular_values, axis=1)
        tot_variance = np.sum(tot_variance, axis=0) / np.sum(tot_variance, axis=0)[-1]

        plt.plot(np.arange(20)+1, np.mean(self.singular_values, axis=0)[:20])
        plt.xlabel('Principal Component')
        plt.ylabel('Singular Value')
        plt.xticks(np.arange(20)+1)
        plt.yscale('log')
        plt.axhline(1, ls='--')
        plt.show()

        plt.plot(np.arange(20)+1, tot_variance[:20])
        plt.xlabel('Principal Component')
        plt.ylabel('Variance')
        plt.xticks(np.arange(20)+1)
        plt.show()


    def lnlikelihood(self,params, F_i, R_i):
        '''
        log-Likelihood for estimating per-pixel uncertainty via Gibson+ 2020

        Inputs: params - tuple with a and b
                F_i - flux at a given time and wavelength
                R_i - post-PCA residuals at a given time and wavelength

        Outputs: -lnlike - negative of log-likelihood for a given a, b

        This function was originally written by Krishna.
        '''
        a, b = params 

        sigma_i = np.sqrt(a * F_i + b)

        lnlike = -0.5*np.sum((R_i / sigma_i)**2) - np.sum(np.log(sigma_i))

        return -lnlike #returning negative because we want to optimize the likelihood, so minimize the negative likelihood

    def lnlikelihood_2(self, a, Var0, R):
        """
        a - variance scale factor
        Var0 - variance
        R - post-SVD residuals
        """
        Var1 = a * Var0 #scale variance

        lnlike = -0.5 * np.sum(   R**2 / Var1  ) - np.sum(np.log(  np.sqrt(Var1)  ))

        return -lnlike


    def get_error_from_var(self):
        """
        Scale PLP variance to match the post-SVD residuals. Motivated by
        the fact that for some reason the PLP variance is overestimated.
        """
        dat = self.data
        post_pca_data = self.pca_data 
        var = self.variance

        nanvar = np.isnan(var)
        var[nanvar] = dat[nanvar] / 4. #just assume poisson noise where there are nans

        std_arr = np.zeros_like(post_pca_data)
        Ndet, Nphi, Npix = post_pca_data.shape 

        for order in range(Ndet):
            for frame in range(Nphi):
                a_init = 1.

                bf_opt = minimize(self.lnlikelihood_2, a_init, bounds = [ (0., 10  )], method='Nelder-Mead', args = (var[order, frame], post_pca_data[order,frame]))
                a_opt = bf_opt['x']

                sigma_opt = np.sqrt( a_opt * var[order,frame]  )

                std_arr[order,frame] = sigma_opt

        return std_arr


    def get_error_gibson(self, NPC=5):
        '''
        Estimate the post-PCA per-pixel uncertainty via Gibson+ 2020.
        Inputs: self - planet object
                NPC - number of principal components to keep
        Outputs: std_arr_clean - array of 1 sigma uncertainty estimates

        This function was originally written by Krishna.
        '''
        cube = self.data #wavelength-aligned data, pre-PCA
        post_pca_data = self.pca_data

        std_arr = np.zeros_like(cube)
        Ndet, Nphi, Npix = cube.shape 

        for i in range(Ndet):
            for j in range(Nphi):
                ab_init = np.array([1, 1]) #initial guess for a, b

                #optimize the likelihood
                bf_opt = minimize(self.lnlikelihood, ab_init, bounds = [(0.0, 10.0), (-np.inf, np.inf)], method='Nelder-Mead', args=(cube[i,j], post_pca_data[i,j]))

                a_opt, b_opt = bf_opt['x']

                sigma_i_opt = np.sqrt(a_opt * cube[i,j] + b_opt)

                std_arr[i,j] = sigma_i_opt

        # std_arr[np.isnan(std_arr)] = 1e10

        ## de-noise the noise...
        #do a singular value decomposition and recompose with only the first few components
        std_arr_clean = np.zeros_like(std_arr)

        for order in range(Ndet):
            u, s, vh = np.linalg.svd(std_arr[order], full_matrices=False)

            s[NPC:] = 0.
            W = np.diag(s)
            A = np.dot(u, np.dot(W, vh))
            std_arr_clean[order] = A 

        # self.std_arr = std_arr_clean

        return std_arr_clean


    def get_error_boucher(self, filt_width=None):
        """
        Estimate error on post-PCA data as described in Boucher et al. 2021
        """
        snr = self.snr 
        med_snr = np.median(snr, axis=2) #median SNR per order per frame

        Ndet, Nphi, Npix = snr.shape
        for i in range(Ndet):med_snr[i] /= med_snr[i].max() ## normalize to the median SNR across all frames (in that order)

        cube = self.data

        std_arr = np.zeros_like(cube)

        #remove just one component
        single_pc_removed = np.zeros_like(cube)
        for i in range(Ndet):
            u, s, vh = np.linalg.svd(cube[i], full_matrices=False)
            s[0] = 0.
            W = np.diag(s)
            A = np.dot(u, np.dot(W, vh))

            A -= np.mean(A, axis=1)[:, np.newaxis]

            single_pc_removed[i] = A 

        # estimate the standard deviation in time per pixel #
        per_pixel_stds = np.sqrt(np.var(single_pc_removed, axis=1, ddof=1))
        for i in range(Ndet):
            for j in range(Nphi):
                std_arr[i,j] = per_pixel_stds[i] / med_snr[i,j] #inverse weighted by the SNR

        # smooth with a gaussian filter #
        if filt_width != None:
            for i in range(Ndet):
                for j in range(Nphi):
                    std_arr[i,j] = gaussian_filter1d(std_arr[i,j], file_width)


        return std_arr


    def get_error(self, method='PLP', NPC=5, filt_width=None):
        """
        Get per-pixel error estimates.
        """
        # if method == 'PLP':
        #     std_arr = self.get_error_PLP ### ???
        if method == 'PLP':
            std_arr = self.get_error_from_var()

        elif method == 'gibson':
            std_arr = self.get_error_gibson(NPC)

        elif method == 'boucher':
            std_arr = self.get_error_boucher(filt_width)

        self.std_arr = std_arr 

        return std_arr


    def get_mask(self, threshold=0.7, niter=10):
        """
        This method identifies wavelength channels in which the transmittance
        is less than a specified threshold. It does this by iteratively
        fitting for the blaze function continnum and dividing it out. A mask
        of the channels with signal above the threshold will be saved.
        Inputs: threshold - number between 0 and 1
                niter - number of iterations. I've found 10 is fine

        Outputs: mask_arr - array of booleans size (Ndet, Npix)
        """
        Ndet, Nphi, Npix = self.data.shape 

        mask_arr = np.zeros((Ndet, Npix))

        for i in range(Ndet):
            wl = self.wlgrid[i]
            dat = self.data[i,0]

            xx = dat.copy()
            for j in range(niter):
                z = np.polyfit(wl, xx, 4)
                cont = np.poly1d(z)(wl)
                divided = dat / cont 
                mean, med, sc = sigma_clipped_stats(divided, sigma_lower=3., sigma_upper=3.)
                divided[divided < 1. - 1.*sc] = 1.
                xx = divided * cont 

            z = np.polyfit(wl, xx, 4)
            cont = np.poly1d(z)(wl)
            divided = dat/cont 

            mask = (divided > threshold)

            mask_arr[i] = mask 

        mask_arr = mask_arr.astype(bool)

        self.mask_arr = mask_arr
        self.threshold = threshold

        return mask_arr


    def read_mask(self, threshold):
        # currently supports 90, 95, and 98
        wl_data, mask_arr = pickle.load(open('IGRINS_tell_mask_'+str(threshold)+'pc.pic','rb'), encoding='bytes')

        self.mask_arr = mask_arr
        self.threshold = threshold

    def plot_snr(self):
        '''
        Plot the SNR per order.
        '''
        snr= self.snr_raw
        wlgrid = self.wlgrid

        med_snr = np.nanmedian(snr, axis=1)
        med_per_order = np.nanmedian(med_snr, axis=1)
        med_snr[med_snr < 0.] = 0.
        med_per_order[med_per_order < 0.] = 0.

        medH = np.nanmedian(np.rollaxis(snr, 1)[:,wlgrid < 1.85])
        medK = np.nanmedian(np.rollaxis(snr, 1)[:,wlgrid > 1.85])

        print('Median H:', str(medH))
        print('Median K:', str(medK))

        for i in range(wlgrid.shape[0]):
            plt.plot(wlgrid[i], med_snr[i], c='r')

        wlcents = np.nanmedian(wlgrid, axis=1)
        plt.plot(wlcents, med_per_order, c='blue', lw=3, zorder=5)

        plt.ylabel('SNR per res element')
        plt.xlabel('Wavelength [micron]')
        plt.show()

        self.med_snr = med_snr
        self.med_snr_per_order = med_per_order

        return med_snr

    def save(self, path=None):
        '''
        Save the data and assocated arrays to pickle files
        Input: path - directory to save in. If None, will save in same directory as the data
        '''
        if path == None:
            path = self.path

        data_raw = self.data_raw
        data = self.data 
        pca_data = self.pca_data
        pca_scale = self.pca_scale
        wlgrid = self.wlgrid 
        ph_arr = self.ph_arr
        rvel_arr = self.rvel_arr
        time_arr = self.time_arr
        am_arr = self.am_arr
        hum_arr = self.hum_arr
        NPC = str(self.NPC)
        var = self.variance
        nanmask = self.nanmask 
        var_nanmask = self.var_nanmask

        pickle.dump([wlgrid, data_raw],open(path+'/RAW_cube.pic','wb'), protocol=2)
        pickle.dump([wlgrid, data], open(path+'/cube.pic','wb'), protocol=2)
        pickle.dump([wlgrid, pca_data], open(path+'/PCA_matrix_'+NPC+'_PC.pic','wb'), protocol=2)
        pickle.dump([wlgrid, pca_scale], open(path+'/data_to_scale_with_'+NPC+'_PC.pic','wb'),protocol=2)
        pickle.dump(ph_arr, open(path+'/ph.pic','wb'),protocol=2)
        pickle.dump(rvel_arr, open(path+'/rvel.pic','wb'), protocol=2)
        pickle.dump(time_arr, open(path+'/time_BJD.pic','wb'), protocol=2)
        pickle.dump(am_arr, open(path+'/airmass.pic','wb'), protocol=2)
        pickle.dump(hum_arr, open(path+'/humidity.pic','wb'), protocol=2)
        pickle.dump([wlgrid, var], open(path+'/variance.pic','wb'), protocol=2)

        pickle.dump(nanmask, open(path+'/nanmask.pic','wb'), protocol=2)
        pickle.dump(var_nanmask, open(path+'/var_nanmask.pic','wb'), protocol=2)

        if 'mask_arr' in vars(self):
            pickle.dump(self.mask_arr, open(path+'/'+str(self.threshold)+'_mask.pic','wb'), protocol = 2)
        if 'med_snr' in vars(self):
            pickle.dump([wlgrid, self.med_snr], open(path+'/SNR.pic','wb'), protocol=2)
        if 'std_arr' in vars(self):
            pickle.dump([wlgrid, self.std_arr], open(path+'/std_arr.pic','wb'), protocol=2)
        if 'where_corr' in vars(self):
            pickle.dump([self.where_corr], open(path+'/where_corr.pic','wb'), protocol=2)


    def do_it_all(self, path=None, NPC=4,NPCstd =5,ref=None, get_RV=False, 
        source='tepcat', transit_only=False, low_index=0, high_index=0,
        whichorders=None, normed=False, order=20, method='svd',get_err=True,
        which='last'):
        """
        Do everything instead of having to manually call each method individually.
        """
        self.get_params(ref=ref, get_RV=get_RV,source=source)
        self.cubify()
        self.trim(transit_only=transit_only,low_index=low_index, high_index=high_index, which=which)
        self.align(whichorders=whichorders, normed=normed)
        self.plot(order=order)
        self.plot_snr()
        if method == 'svd':
            self.do_svd(NPC=NPC)
        elif method == 'sysrem':
            self.do_sysrem(NPC=NPC)

        if get_err == True:
            self.get_error(method='gibson',NPC=NPCstd)
        if path == None:
            path=self.path
        self.save(path)


### WARNING THE AUTOMLRPCA CLASS IS EXPERIMENTAL AND I DO NOT RECCOMEND 
### USING IT YET ###

class AutoMLRPCA:
    def __init__(self, cube, airmass, humidity, time, variance,standardize=False, NPC=None, coefdeter=0.68, operation='subtract', intercept=False):
        """
        Select components to remove and remove them. Propagates variance through this process.

        cube - pre-PCA data
        airmass - air mass array
        humidity - humidity array
        time - time in BJD
        variance - variance per resolution element as output by the PLP
        standardize - boolean, if True, automatic component selection doesn't work so well
        NPC - manually set number of principal components to remove in all orders. If set to None, will automatically select components to remove in each order.
        coefdeter - float, coefficient of determination for selecting components. ranges from 0 to 1, with lower values being more aggressive.
        operation - str, subtract or divide. The operation to remove the regression fit.
        """
        time -= time.min() #subtract the minimum from the time array. Makes the regressions easier.

        self.cube = cube
        self.airmass = airmass
        self.humidity = humidity
        self.time = time
        self.continuum = np.median(cube, axis=2) #median counts in each frame, proxy for the continuum level
        self.variance = variance
        self.NPC = NPC
        self.coefdeter = coefdeter
        self.operation = operation
        self.intercept = intercept
        
        self.mask = np.ones((cube.shape[0], cube.shape[2])).astype(bool) #if you want to set a telluric mask, manually set it as an attribute after initializing the object
        self.standardize = standardize
        if standardize == True:
            means = np.mean(cube, axis=1)
            self.means = means
            
            stds = np.std(cube,axis=1)
            self.stds = stds
            
            standard_cube = np.zeros_like(cube)
            for i in range(cube.shape[0]):
                standard_cube[i] = (cube[i] - means[i]) / stds[i]
            
            self.standard_cube = standard_cube
        
    def linear_regression(self, xdat, ydat, p=1):
        ni = len(xdat)
        X = np.zeros((ni, p+1))
        for i in range(p+1): X[:,i] = xdat**i

        Y = ydat.reshape(ni, 1)

        B = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y)) #normal equation

        predict = np.dot(X, B)

        SStot = np.sum((Y - np.mean(Y))**2)
        SSres = np.sum((Y - predict)**2)

        R2 = 1. - SSres/SStot
        R = np.sqrt(R2)

        return B, predict, R
    
    def multilinear_regression(self,X, ydat, W, intercept=False):
        """
        Multilinear regression via the normal equation
        X - design matrix of predictor variables
        ydat - data to fit to
        W - precision matrix or other weights for weighted regression
        """
        ni, p = X.shape

        if intercept == True:
            X = np.column_stack([np.ones(ni), X]) #add an intercept to the predictor variables

        Y = ydat.reshape(ni, 1)

        MB = (np.linalg.inv(np.dot(X.T,np.dot(W, X)))) #variance of regression coefficients
        B = np.dot(MB, np.dot(X.T,np.dot(W,Y))) #normal equation

        predict = np.dot(X, B) #regression fit

        SStot = np.sum((Y - np.mean(Y))**2)
        SSres = np.sum((Y - predict)**2)

        R2 = 1. - SSres/SStot #coefficient of determination
        R = np.sqrt(R2)

        sigmaf = np.dot(X, np.dot(MB, X.T)) #variance on the regression fit

        return B, predict, R2, MB, sigmaf
    
    def check_corr(self, predictors, LSV):
        """
        Check correlations between the predictor variables and left singular vectors.
        Returns a mask of which right singular vectors to use in the mutlilinear regression to the data.
        """
        K = LSV.shape[1] #number of principal components
        
        R2s = np.array([self.multilinear_regression(predictors, LSV[:,j], np.diag(np.ones(predictors.shape[0])), intercept=True)[2] for j in range(K)])
        
        where_corr = (R2s >= self.coefdeter)
        
        return where_corr
    
    def get_filters(self, U, variance):
        """
        Get the matrices for filtering the model as described in Gibson et al. 2022. May need to be debugged.
        """
        sigma = np.nanmean(np.sqrt(variance), axis=1)
        
        lamb = np.diag(sigma)
        
        lamu = np.dot(lamb, U)
        lamump = np.dot(np.linalg.inv(np.dot(lamu.T, lamu)), lamu.T)
        
        return lamb, lamump
        
    
    def svd(self, order):
        """
        Decompose the data matrix using singular value decomposition. Select components to use in a regression fit to the data, and remove that.
        """
        if self.standardize == True:
            A = self.standard_cube[order].copy()
            var = self.variance[order].copy() / self.stds[order]**2
        else:
            A = self.cube[order].copy()
            var = self.variance[order].copy()
            
            
        Omega = 1. / var #precision matrix
        Nphi, Npix = A.shape
        
        LSV, S, RSV = np.linalg.svd(A[:,self.mask[order]], full_matrices=False)
        
        #if didn't provide global number of PCs to remove, automatically select them.
        if self.NPC == None:
                
            continuum = self.continuum[order]

            predictors0 = np.column_stack([continuum, self.time, self.airmass, self.humidity])

            predictors = np.concatenate([predictors0, predictors0**2, predictors0**3], axis=1)

            where_corr = self.check_corr(predictors, LSV)
            
        #or, just take out the first N
        else:
            where_corr = np.zeros(A.shape[0])
            where_corr[:self.NPC] = 1
            where_corr = where_corr.astype(bool)
        
        X = RSV[where_corr].T #design matrix of the selected right singular vectors
        
        regression = np.zeros((Nphi, Npix)) #initialize regression fit
        
        NPC = where_corr.sum()
        if self.intercept == True:NPC+=1
        Bs = np.zeros((Nphi, NPC)) #regression coefficients
        MBs = np.zeros((Nphi, NPC)) #errors in regression coefficients

        sigmas = np.zeros_like(regression) #variance in regression
    
        for i in range(Nphi):
            ydat = A[i].copy()[self.mask[order]] #data to fit to

            weights = np.diag(Omega[i, self.mask[order]]) #weights for regression
            B, fi, r2, MB, sigmaf = self.multilinear_regression(X, ydat, weights, self.intercept) #multilinear regression on the data

            regression[i, self.mask[order]] = fi.reshape(self.mask[order].sum())
            
            Bs[i] = B.reshape(NPC)
            MBs[i] = np.diag(MB)
            sigmas[i, self.mask[order]] = np.diag(sigmaf)
        
        if self.standardize == True:
            regression = regression * self.stds[order] + self.means[order]
            A = A * self.stds[order] + self.means[order]
            sigmas *= self.stds[order]**2
            var *= self.stds[order]**2
                
        detrended = A.copy()
        if self.operation == 'subtract':
            detrended[:, self.mask[order]] -= regression[:, self.mask[order]]
        elif self.operation == 'divide':
            detrended[:, self.mask[order]] /= regression[:, self.mask[order]]
        
        
        ##################
        # propagate error through the detrending operations #
        post_pca_var = np.zeros_like(regression)

        for i in range(Nphi):
            a = A[i]
            sigmaa = var[i]

            b = regression[i]
            sigmab = sigmas[i]

            if self.operation == 'subtract':
                post_pca_var[i] = sigmaa + sigmab -2 *np.corrcoef(a, b)[0,1]*np.sqrt(sigmaa)*np.sqrt(sigmab)
            elif self.operation == 'divide':
                post_pca_var[i] = detrended[i]**2 * ( (np.sqrt(sigmaa) / a)**2  + (np.sqrt(sigmab) / b)**2 - 2 * np.corrcoef(a, b)[0,1]* np.sqrt(sigmaa)*np.sqrt(sigmab) / a / b)
                
        if self.operation == 'divide':
            detrended -= 1.
            
        #high pass filter
        for i in range(Nphi):
            detrended[i, self.mask[order]] -= gaussian_filter1d(detrended[i, self.mask[order]], 80)
    
        #mean subtract
        detrended -= np.mean(detrended[:, self.mask[order]], axis=1)[:, np.newaxis]
        #sigma clip
        sig = np.std(detrended)
        med = np.median(detrended)
        detrended[detrended > 3. *sig + med] = 0.
        detrended[detrended < -3. * sig + med] = 0.
                 
        #if this isn't centered at zero by now something is very wrong lol
                                                    
        ### get stuff for Gibson model filtering ##
        # LL, LP = self.get_filters(Bs, var)
                                                    
        return regression, detrended, where_corr, post_pca_var, Bs, X #LL, LP
    
    def __call__(self):
        Ndet, Nphi, Npix = self.cube.shape
        
        pca_scale = np.zeros_like(self.cube)
        pca_data = np.zeros_like(self.cube)
        where_corr = np.zeros((Ndet, Nphi))
        sigma_Ahat = np.zeros_like(self.cube)
        if self.intercept == True:
            U = np.zeros((Ndet, Nphi, Nphi+1))
            vectors = np.zeros((Ndet, Nphi+1, Npix))
        else:
            U = np.zeros((Ndet, Nphi, Nphi))
            vectors = np.zeros((Ndet, Nphi, Npix))
        LL = np.zeros((Ndet, Nphi, Nphi))
        LP = np.zeros((Ndet, Nphi+1, Nphi))
                                             
        for i in range(Ndet):
            # scale, dat, wc, sigma, Bs,ll, lp = self.svd(i)
            scale, dat, wc, sigma, Bs, X = self.svd(i)
            if wc.shape[0] < Nphi: 
                print(i)
                continue
            pca_scale[i,:,self.mask[i]] = scale[:,self.mask[i]].T
            pca_data[i,:,self.mask[i]] = dat[:,self.mask[i]].T
            where_corr[i] = wc

            sigma_Ahat[i,:, self.mask[i]] = sigma[:, self.mask[i]].T
            
            if self.intercept:
                U[i,:,0] = Bs[:,0]
                U[i,:,1:][wc] = Bs[:,1:].T

                vectors[i,0] = 1.
                vectors[i,1:][wc] = X.T
            else:
                U[i,:][wc] = Bs.T
                vectors[i][wc] = X.T 
            # LL[i] = ll
            # LP[i,0] = lp[0]
            # LP[i,1:][wc] = lp[1:]
            
        self.pca_scale = pca_scale
        self.pca_data = pca_data
        self.where_corr = where_corr
        self.post_pca_var = sigma_Ahat
        self.basis = U
        self.vectors = vectors
        # self.lamb = LL
        # self.lamu = LP
        
        return pca_scale, pca_data, where_corr, sigma_Ahat, U, vectors# LL, LP


################################## VERSION HISTORY ##############################
"""
VERSION 1.0.1

PREVIOUS VERSION: 1.0.0, released 10/03/2024

NEW FEATURES:

BUG FIXES:
    removed extraneous call of .tdb method from times already in BJD

***************************************************************************
VERSION 1.0.0

PREVIOUS VERSION: 0.11.0, released 06/06/2024

NEW FEATURES:
    do_sysrem method, which does sysrem
    for align method, can choose whether to align to the first or last frame
    can get a nan mask when doing wavelength alignment (also in SVD process)
    Added get_error_PLP method which gets errors from the PLP variance.

BUG FIXES:
*********************************************************************************
VERSION 0.11.0
06/06/2024

PREVIOUS VERSION: 0.10.0, released 01/09/2024

NEW FEATURES:
    AutoMLRPCA class, which does PCA/multilinear regression, and can 
    do "automatic" component selection, which is still experimental.

    do_pca is now do_svd

BUG FIXES:
******************************************************************************
VERSION 0.10.0
01/09/2024

PREVIOUS VERSION: 0.9.2, released 11/11/2023

NEW FEATURES: 
    New method make_std_arr estimates per-pixel uncertainties via Gibson+ 2020.
    This method has been added to do_it_all.
    cubify now saves the humidity and variance for each frame as well.

BUG FIXES:
    do_pca now mean-subtracts post-pca data. 
*****************************************************************************
VERSION 0.9.2
11/6/2023

PREVIOUS VERSION: 0.9.1, released 10/23/2023

NEW FEATURES:

BUG FIXES:
    Fixed issues with read_TEPCAT and WASP planets.
********************************************************************************
VERSION 0.9.1
10/23/2023

PREVIOUS VERSION: 0.9.0, released 10/10/2023

NEW FEATURES:

BUG FIXES:
    read_TEPCAT can handle WASP planets with A's in the name.
********************************************************************************
VERSION: 0.9.0
10/10/2023

PREVIOUS VERSION: [unnumbered], released 8/23/2023

NEW FEATURES:
    do_it_all method does everything automatically instead of having to call
    each method individually.

    plot_snr method plots the median SNR per order.

    cubify method creates airmass and exposure time arrays.

    save method saves the airmass and SNR arrays.


BUG FIXES:
    read_TEPCAT adds an underscore into planet names from the HD catalogue to 
    match the URL.


############################### DEV NOTES ##############################################
OCTOBER 10, 2023 - Added do_it_all method. Cubify makes arrays of the airmass 
and exposure times. Added plot_snr method to plot the snr. -P.

AUGUST 23, 2023 - Added fix for reading WASP planets with two digits from NEA. -P

AUGUST 21, 2023 - Added the option to not normalize each frame in the align() process,
added the plot_scree and get_mask functions. -P.

APRIL 30, 2023 - Added the ability to save an array of each frame's time in BJD. -P.

APRIL 10, 2023 - Added read_TEPCAT and read_NEA methods, now reading from TEPCat is
the default. get_params() also prints the Kp, Rp, Mp, and Rstar. RV=0 by default. -P.

APRIL 06, 2023 - Added transit duration and trimming for in transit only. -P.

JULY 21, 2022 - Fixed bug in save() that set wlgrid as the data, saving the data lmao. -P.

JUNE 27, 2022 - Added RA and DEC attributes due to reports that sometimes RA and DEC
reported in FITS headers are wrong. Removed ID arg from save(). Added more docstrings.
-P.


"""