from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from alexmods.specutils import Spectrum1D
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
import glob, os, sys, time

def parse_directory(datadir, rb, datatype="fox", suffix="ds"):
    fnames = glob.glob(os.path.join(datadir, "{}*{}_{}_specs.fits".format(rb,suffix,datatype)))
    Nframe = len(fnames)
    print(f"{Nframe} files to read and coadd")
    
    with fits.open(fnames[0]) as hdulist:
        Nband, Npix, Nord, Nobj = hdulist[0].data.shape
    print(f"{Nobj} objects, {Nord} orders, {Npix} pixels")
    
    # wave, flux, err, flatflux, flaterr
    alloutput = np.zeros((Nobj, Nframe, Nord, Npix, 5))
    headers = []
    for iframe,fname in enumerate(fnames):
        with fits.open(fname) as hdulist:
            headers.append(hdulist[0].header)
            data = hdulist[0].data
            Fdata = hdulist[1].data
            for iobj in range(Nobj):
                for iord in range(Nord):
                    for iband in range(3):
                        alloutput[iobj, iframe, iord, :, iband] = data[iband, :, iord, iobj]
                    alloutput[iobj, iframe, iord, :, 3] = Fdata[1, :, iord, iobj]
                    alloutput[iobj, iframe, iord, :, 4] = Fdata[2, :, iord, iobj]
    print(f"Output array size: {Nobj} obj x {Nframe} frame x {Nord} orders x {Npix} pixels x 5 bands")
    print("Bands = wavelength, flux, error, flat flux, flat err")
    
    all_objs = verify_headers(alloutput, headers)
    
    return alloutput, headers, all_objs

def verify_headers(alloutput, headers):
    """
    Verifies that all the info needed in the headers is there and consistent
    - order numbers (ECORD{iorder})
    - object names (OBJ{iobj:03})
    """
    print("Verifying data and headers...")
    Nobj, Nframe, Nord, Npix, Nband = alloutput.shape
    assert len(headers) == Nframe
    all_objs = []
    all_exptime = []
    for iframe, h in enumerate(headers):
        # Check ECORD all there and the same
        try:
            for iorder in range(Nord):
                x = h[f"ECORD{iorder}"]
        except Exception as e:
            print(f"ERROR: ECORD not all there, broken at frame {iframe} order {iorder}")
            print(e)
            raise
        # Check OBJ all there
        thisobjs = []
        try:
            for iobj in range(Nobj):
                thisobjs.append(h[f"OBJ{iobj:03}"])
        except Exception as e:
            print(f"ERROR: Objects not all there, broken at frame {iframe} object {iobj}")
            print(e)
            raise
        all_objs.append(thisobjs)
        x = h["MJD"]
        all_exptime.append(h["EXPTIME"])
    all_objs = np.array(all_objs).astype(str)
    for iobj in range(Nobj):
        objs = all_objs[:,iobj]
        print(f"  OBJ{iobj:03} == {objs[0]}...")
        assert np.all(objs==objs[0]), objs
    print("  Exposure times:", all_exptime)
    print(f"  Total exposure: {np.sum(all_exptime)}")
    print("Verified!")
    return all_objs[0,:]

def interpolate_onto_common_dispersion(alloutput):
    print("Interpolating onto common dispersion")
    Nobj, Nframe, Nord, Npix, Nband = alloutput.shape
    Npix_interp_max = int(Npix*1)
    
    wave_interp = np.zeros((Nord, Npix_interp_max)) + np.nan
    flux_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) + np.nan
    ivar_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max))
    Fflux_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max)) + np.nan
    Fivar_interp = np.zeros((Nobj, Nframe, Nord, Npix_interp_max))
    for iord in range(Nord):
        data = alloutput[:,:,iord,:,:]
        wave = data[:,:,:,0]
        wave[wave < 1] = np.nan
        finite_wave = np.where(np.isfinite(wave))[2]
        wave_ix_min, wave_ix_max = np.min(finite_wave), np.max(finite_wave) + 1
        
        Npix_used = wave_ix_max - wave_ix_min #int(np.median(np.sum(np.isfinite(wave), axis=2)))
        ## TODO testing this to see how it goes
        #Npix_interp = int(Npix_used * 2)
        #interp_ix_min, interp_ix_max = int(wave_ix_min*2), int(wave_ix_max*2)
        Npix_interp = int(Npix_used)
        interp_ix_min, interp_ix_max = int(wave_ix_min), int(wave_ix_max)
        
        wmin, wmax = np.nanmin(wave), np.nanmax(wave)
        this_wave_interp = np.linspace(wmin, wmax, Npix_interp)
        wave_interp[iord, interp_ix_min:interp_ix_max] = this_wave_interp
        print(iord, Npix_used, Npix_interp, np.diff(this_wave_interp)[0])

        for iobj in range(Nobj):
            for iframe in range(Nframe):
                flux_interp[iobj,iframe,iord,interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,1],
                              left=np.nan, right=np.nan)
                total_ivar1 = np.nansum(data[iobj,iframe,wave_ix_min:wave_ix_max,2]**-2.)
                ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,2]**-2.,
                              left=np.nan, right=np.nan)
                total_ivar2 = np.nansum(ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max])
                ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = ivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]*total_ivar1/total_ivar2
                
                Fflux_interp[iobj,iframe,iord,interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,3],
                              left=np.nan, right=np.nan)
                total_ivar1 = np.nansum(data[iobj,iframe,wave_ix_min:wave_ix_max,4]**-2.)
                Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = \
                    np.interp(this_wave_interp,
                              data[iobj,iframe,wave_ix_min:wave_ix_max,0],
                              data[iobj,iframe,wave_ix_min:wave_ix_max,4]**-2.,
                              left=np.nan, right=np.nan)
                total_ivar2 = np.nansum(Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max])
                Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max] = Fivar_interp[iobj,iframe,iord, interp_ix_min:interp_ix_max]*total_ivar1/total_ivar2
            ## This was cutting like 2/3 of the pixels, not really sure why but probably that my CRR algorithm is just wrong!
            #new_flux, new_ivar, new_mask = cosmic_ray_reject(flux_interp[iobj,:,iord,:],
            #                                                 ivar_interp[iobj,:,iord,:],
            #                                                 sigma=10., minflux=-100, verbose=True, use_mad=True)
            #print("Cosmic ray rejection for obj {} rejected {}/{}".format(iobj,new_mask.sum(),new_mask.size))
            new_flux, new_ivar = flux_interp[iobj,:,iord,:], ivar_interp[iobj,:,iord,:]
            thisflat = Fflux_interp[iobj,:,iord,:]
            thisflaterr = Fivar_interp[iobj,:,iord,:]**-0.5
            flux_interp[iobj,:,iord,:] = new_flux/thisflat
            ivar_interp[iobj,:,iord,:] = new_ivar*(thisflat**2)
    return wave_interp, flux_interp, ivar_interp

def simple_coadd(alloutput):
    print("Simple coadd: interpolate onto common dispersion, normalize all spectra by the median, doing a weighted average to remove cosmic rays, then multiplying back and adding")
    wave_interp, flux_interp, ivar_interp = interpolate_onto_common_dispersion(alloutput)
    errs_interp = ivar_interp**-0.5
    Nobj, Nframe, Nord, Npix = flux_interp.shape
    
    ## Cosmic Ray Reject
    print(f"Using {Nframe} frames to do cosmic ray rejection")
    sigma = 10
    # normalize using median over pixels
    norm_objframeord = np.nanmedian(flux_interp, axis=3)
    norm_flux = flux_interp / norm_objframeord[:,:,:,np.newaxis]
    norm_errs = errs_interp / norm_objframeord[:,:,:,np.newaxis]
    # find outliers taking median over frames and fill with median spectrum
    median_flux = np.nanmedian(norm_flux, axis=1)
    norm_flux_diff = (norm_flux - median_flux[:,np.newaxis,:]) / norm_errs
    # mask pixels that are too deviating or not finite
    # the mask is applied as things with norm_errs = 9999.
    mask = (np.abs(norm_flux_diff) > sigma) | (~np.isfinite(norm_flux_diff))
    print(f"Masking {mask.sum()}/{mask.size} pixels")
    norm_errs[mask] = 9999.
    for iobj in range(Nobj):
        for iframe in range(Nframe):
            for iord in range(Nord):
                this_mask = mask[iobj, iframe, iord]
                norm_flux[iobj, iframe, iord, this_mask] = median_flux[iobj, iord, this_mask]
    # do a weighted average as the coadd
    norm_ivar = norm_errs**-2.
    norm_ivar[~np.isfinite(norm_ivar)] = 9999.**-2
    total_norm_ivar = np.sum(norm_ivar, axis=1) # sum over frames
    total_weighted_flux = np.sum(norm_flux * norm_ivar, axis=1) # sum over frames
    average_norm_flux = total_weighted_flux / total_norm_ivar
    #sumsquare_weighted_flux = np.sum(norm_ivar * (norm_flux - average_norm_flux[:,np.newaxis,:,:])**2, axis=1)
    #average_norm_stdv = np.sqrt(sumsquare_weighted_flux / total_norm_ivar)
    average_norm_stdv = total_norm_ivar**-0.5
    
    # Final output: Nobj x Nord x Npix
    # rescale by the total counts
    scaling_factor = np.sum(norm_objframeord, axis=1)
    final_flux = average_norm_flux * scaling_factor[:,:,np.newaxis]
    final_errs = average_norm_stdv * scaling_factor[:,:,np.newaxis]
    return wave_interp, final_flux, final_errs

def write_outputs(waves, fluxs, errss, all_objs, headers, outdir=".",
                  leftclip=50, rightclip=50):
    Nobj, Nord, Npix = fluxs.shape
    waves = waves[:,leftclip:-rightclip]
    fluxs = fluxs[:,:,leftclip:-rightclip]
    errss = errss[:,:,leftclip:-rightclip]
    
    h = headers[0]
    ivars = errss**-2.
    ivars[~np.isfinite(ivars)] = (np.nanmax(fluxs)*1000)**-2.
    for iobj in range(Nobj):
        objdir = os.path.join(outdir, all_objs[iobj])
        os.makedirs(objdir, exist_ok=True)
        for iord in range(Nord):
            order_number = "{:03}".format(h[f"ECORD{iord}"])
            outfname = f"{objdir}/order_{order_number}.txt"
            meta = dict(h.copy())
            meta["IOBJ"]=iobj
            meta["IORDER"]=iord
            ii = np.isfinite(waves[iord])
            spec = Spectrum1D(waves[iord,ii], fluxs[iobj,iord,ii], ivars[iobj,iord,ii], metadata=meta)
            spec.write(outfname)
        print("Wrote",objdir)
    
