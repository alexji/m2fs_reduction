from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from scipy import optimize
import re

def mrdfits(fname, ext):
    """ Read fits file """
    with fits.open(fname) as hdulist:
        hdu = hdulist[ext]
        data = hdu.data.T
        header = hdu.header
    return data, header

def write_fits_two(outfname,d1,d2,h):
    """ Write fits files with two arrays """
    hdu1 = fits.PrimaryHDU(d1.T, h)
    hdu2 = fits.ImageHDU(d2.T)
    hdulist = fits.HDUList([hdu1, hdu2])
    hdulist.writeto(outfname, overwrite=True)

def read_fits_two(fname):
    """ Read fits files with two arrays """
    with fits.open(fname) as hdulist:
        assert len(hdulist)==2
        header = hdulist[0].header
        d1 = hdulist[0].data.T
        d2 = hdulist[1].data.T
    return d1, d2, header

def m2fs_load_files_two(fnames):
    """ Create arrays of data from multiple fnames """
    assert len(fnames) >= 1, fnames
    N = len(fnames)
    img, h = mrdfits(fnames[0],0)
    Nx, Ny = img.shape
    
    headers = []
    imgarr = np.empty((N, Nx, Ny))
    imgerrarr = np.empty((N, Nx, Ny))
    for k, fname in enumerate(fnames):
        imgarr[k], imgerrarr[k], h = read_fits_two(fname)
        headers.append(h)
    return imgarr, imgerrarr, headers

#def m2fs_extract_ghlb(datafname,flatfname):
    

def m2fs_extract1d(data, tracecoef, id, yaper=7, method="sum"):
    nx = data.shape[0]
    ypeak = np.polyval(tracecoef[id], np.arange(nx))
    spec1d = np.zeros(nx)
    outerror = np.zeros(nx)
    vround = np.vectorize(lambda x: int(round(x)))
    ix1s = vround(ypeak) - int(yaper/2.)
    ix2s = ix1s + yaper
    for j in range(nx):
        flux = data[j,ix1s[j]:ix2s[j]]
        if method == "sum":
            try:
                spec1d[j] = np.nansum(flux)
            except:
                spec1d[j] = np.nan
        else:
            raise NotImplementedError
    return spec1d

def jds_poly_reject(x,y,ndeg,nsig_lower,nsig_upper,niter=5):
    good = np.ones(len(x), dtype=bool)
    w = np.arange(len(x))
    for i in range(niter):
        coeff = np.polyfit(x[w], y[w], ndeg)
        res = y[w] - np.polyval(coeff, x[w])
        sig = np.std(res)
        good[w] = good[w] * (((res >= 0) & (res <= nsig_upper*sig)) | \
                             ((res < 0)  & (res >= -1*nsig_lower*sig)))
        w = np.where(good)[0]
    coeff = np.polyfit(x[w], y[w], ndeg)
    yfit = np.polyval(coeff, x[w])
    return yfit, coeff

def gaussfit(xdata, ydata, p0, **kwargs):
    """
    p0 = (amplitude, mean, sigma) (bias; linear; quadratic)
    """
    NTERMS = len(p0)
    if NTERMS == 3:
        def func(x, *theta):
            z = (x-theta[1])/theta[2]
            return theta[0] * np.exp(-z**2/2.)
    elif NTERMS == 4:
        def func(x, *theta):
            z = (x-theta[1])/theta[2]
            return theta[0] * np.exp(-z**2/2.) + theta[3]
    elif NTERMS == 5:
        def func(x, *theta):
            z = (x-theta[1])/theta[2]
            return theta[0] * np.exp(-z**2/2.) + theta[3] + theta[4]*x
    elif NTERMS == 6:
        def func(x, *theta):
            z = (x-theta[1])/theta[2]
            return theta[0] * np.exp(-z**2/2.) + theta[3] + theta[4]*x + theta[5]*x**2
    else:
        raise ValueError("p0 must be 3-6 terms long, {}".format(p0))
        
    popt, pcov = optimize.curve_fit(func, xdata, ydata, p0, **kwargs)
    return popt
    #fit=gaussfit(auxx, auxy, coef, NTERMS=4, ESTIMATES=[auxdata[peak1[i]], peak1[i], 2, thresh/2.])
    

def m2fs_biassubtract(ime, h):
    """
    ;+----------------------------------------------------------------------------
    ; PURPOSE:
    ;       Do bias subtraction on an M2FS image (from a single amplifier)
    ;+----------------------------------------------------------------------------
    ; INPUTS:
    ;       ime - image (in units of e-)
    ;         note: must be transposed to match IDL convention!!!
    ;       h - associated header
    ;+----------------------------------------------------------------------------
    ; COMMENTS:
    ;       For completeness, would probably be good to implement optional use
    ;       of the bias region on the top of the frame as well 
    ;+----------------------------------------------------------------------------
    ; HISTORY:
    ;       J. Simon, 02/14  
    ;       A. Ji 05/18 (converted to python)
    ;+----------------------------------------------------------------------------
    """
    # EXTRACT RELEVANT HEADER KEYWORDS
    biassec = h["BIASSEC"]
    datasec = h["DATASEC"]
    
    def strpos(x, c, reverse=False):
        if reverse:
            return x.rfind(c)
        else:
            return x.find(c)
    #EXTRACT APPROPRIATE CCD SECTIONS FROM HEADER
    #BIAS SECTION ON THE RIGHT SIDE OF THE IMAGE
    begin_biasright_x1 = strpos(biassec,'[') + 1
    end_biasright_x1 = strpos(biassec,':')
    begin_biasright_x2 = end_biasright_x1 + 1
    end_biasright_x2 = strpos(biassec,',')
    #NOTE THAT HEADER DEFINITION OF THE BIAS SECTION ACTUALLY ONLY CORRESPONDS
    #TO A CORNER OF THE IMAGE, NOT A STRIP; USE DATA SECTION AS A REPLACEMENT
    begin_biasright_y1 = strpos(datasec,',') + 1
    end_biasright_y1 = strpos(datasec,':',True)
    begin_biasright_y2 = end_biasright_y1 + 1
    end_biasright_y2 = strpos(datasec,']')
    
    #BIAS SECTION ON THE TOP OF THE IMAGE
    begin_biastop_x1 = strpos(datasec,'[') + 1
    end_biastop_x1 = strpos(datasec,':')
    begin_biastop_x2 = end_biastop_x1 + 1
    end_biastop_x2 = strpos(datasec,',')
    
    begin_biastop_y1 = strpos(biassec,',') + 1
    end_biastop_y1 = strpos(biassec,':',True)
    begin_biastop_y2 = end_biastop_y1 + 1
    end_biastop_y2 = strpos(biassec,']')
    
    #DATA SECTION
    begin_data_x1 = strpos(datasec,'[') + 1
    end_data_x1 = strpos(datasec,':')
    begin_data_x2 = end_data_x1 + 1
    end_data_x2 = strpos(datasec,',')
    
    begin_data_y1 = strpos(datasec,',') + 1
    end_data_y1 = strpos(datasec,':',True)
    begin_data_y2 = end_biasright_y1 + 1
    end_data_y2 = strpos(datasec,']')
    
    #CUT OUT BIAS SECTION ON RIGHT SIDE OF IMAGE
    i1 = int(biassec[begin_biasright_x1:end_biasright_x1])-1
    i2 = int(biassec[begin_biasright_x2:end_biasright_x2])
    i3 = int(datasec[begin_biasright_y1:end_biasright_y1])-1
    i4 = int(datasec[begin_biasright_y2:end_biasright_y2])
    #print(i1,i2,i3,i4,ime.shape)
    biasright = ime[i1:i2,i3:i4]

    #TRIM IMAGE TO JUST THE PART WITH PHOTONS IN IT
    i1 = int(datasec[begin_data_x1:end_data_x1])-1
    i2 = int(datasec[begin_data_x2:end_data_x2])
    i3 = int(datasec[begin_data_y1:end_data_y1])-1
    i4 = int(datasec[begin_data_y2:end_data_y2])
    #print(i1,i2,i3,i4,ime.shape)
    ime_trim = ime[i1:i2,i3:i4]

    #print(ime.shape, ime_trim.shape, biasright.shape)

    #REMOVE COLUMN BIAS
    # Note: IDL median doesn't set the /EVEN keyword by default.
    # I find this makes ~0.3 e- difference.
    ime_bstemp = (ime_trim - np.median(biasright,axis=0)).T
    
    #CUT OUT BIAS SECTION ON TOP OF IMAGE
    #COULD SUBTRACT THIS ONE TOO, BUT IT LOOKS TOTALLY FLAT TO ME
    #ime_biastop =  $
    #  ime[fix(strmid(datasec,begin_biastop_x1,end_biastop_x1-begin_biastop_x1))-1: $
    #      fix(strmid(datasec,begin_biastop_x2,end_biastop_x2-begin_biastop_x2))-1, $
    #      fix(strmid(biassec,begin_biastop_y1,end_biastop_y1-begin_biastop_y1))-1: $
    #      fix(strmid(biassec,begin_biastop_y2,end_biastop_y2-begin_biastop_y2))-1]

    #BIAS SUBTRACTED IMAGE
    ime_bs = ime_bstemp
    return ime_bs

def m2fs_4amp(infile, outfile=None):
    """
    ;+----------------------------------------------------------------------------
    ; PURPOSE:
    ;       Take FITS files from the four M2FS amplifiers and combine them into
    ;       a single frame
    ;+----------------------------------------------------------------------------
    ; INPUTS:
    ;       infile
    ;+----------------------------------------------------------------------------
    ; HISTORY:
    ;       J. Simon, 02/14  
    ;       G. Blanc 03/14: Fixed several header keywords
    ;                       Save error frame instead of variance
    ;                       Fixed bug at mutliplying instead of dividing EGAIN
    ;       A. Ji 05/18: converted to python
    ;+----------------------------------------------------------------------------
    """    

    if outfile is None:
        outfile=infile+'.fits'
    
    # CONSTRUCT FILE NAMES
    c1name = infile + 'c1.fits'
    c2name = infile + 'c2.fits'
    c3name = infile + 'c3.fits'
    c4name = infile + 'c4.fits'

    # READ IN DATA
    c1,h1 = mrdfits(c1name,0)
    c2,h2 = mrdfits(c2name,0)
    c3,h3 = mrdfits(c3name,0)
    c4,h4 = mrdfits(c4name,0)

    # GET GAIN AND READNOISE OF EACH AMPLIFIER FROM HEADER
    gain_c1 = h1["EGAIN"]
    gain_c2 = h2["EGAIN"]
    gain_c3 = h3["EGAIN"]
    gain_c4 = h4["EGAIN"]

    readnoise_c1 = h1["ENOISE"]
    readnoise_c2 = h2["ENOISE"]
    readnoise_c3 = h3["ENOISE"]
    readnoise_c4 = h4["ENOISE"]


    # CONVERT TO ELECTRONS
    c1e = c1*gain_c1
    c2e = c2*gain_c2
    c3e = c3*gain_c3
    c4e = c4*gain_c4

    c1e_bs = m2fs_biassubtract(c1e,h1)
    c2e_bs = m2fs_biassubtract(c2e,h2)
    c3e_bs = m2fs_biassubtract(c3e,h3)
    c4e_bs = m2fs_biassubtract(c4e,h4)

    # PLACE DATA IN MERGED OUTPUT ARRAY
    # Note: IDL and python axes are reversed!
    def reverse(x,axis):
        if axis == 1: # reverse columns
            return x[:,::-1]
        if axis == 2:
            return x[::-1,:]
        raise ValueError("axis={} must be 1 or 2".format(axis))
    outim = np.zeros((2056,2048))
    outim[1028:2056,0:1024] = reverse(c1e_bs[0:1028,0:1024],2)
    outim[1028:2056,1024:2048] = reverse(reverse(c2e_bs[0:1028,0:1024],2),1)
    outim[0:1028,1024:2048] = reverse(c3e_bs[0:1028,0:1024],1)
    outim[0:1028,0:1024] = c4e_bs[0:1028,0:1024]

    # MAKE MATCHING ERROR IMAGE
    # NOTE THAT NOISE IN THE BIAS REGION HAS BEEN IGNORED HERE!
    outerr = np.zeros((2056,2048))
    outerr[1028:2056,0:1024] = \
        np.sqrt(readnoise_c1**2 + np.abs(reverse(c1e_bs[0:1028,0:1024],2)))
    outerr[1028:2056,1024:2048] = \
        np.sqrt(readnoise_c2**2 + np.abs(reverse(reverse(c2e_bs[0:1028,0:1024],2),1)))
    outerr[0:1028,1024:2048] = \
        np.sqrt(readnoise_c3**2 + np.abs(reverse(c3e_bs[0:1028,0:1024],1)))
    outerr[0:1028,0:1024] = \
        np.sqrt(readnoise_c4**2 + np.abs(c4e_bs[0:1028,0:1024]))

    # UPDATE HEADER
    def sxaddpar(h,k,v):
        if k in h:
            _ = h.pop(k)
        h[k] = v
    def sxdelpar(h,k):
        _ = h.pop(k)
    sxaddpar(h1,'BUNIT   ','E-/PIXEL')
    sxdelpar(h1,'EGAIN   ')
    sxaddpar(h1,'ENOISE  ', np.mean([readnoise_c1,readnoise_c2,readnoise_c3,readnoise_c4]))
    sxdelpar(h1,'BIASSEC ')
    sxaddpar(h1,'DATASEC ', '[1:2048,1:2056]')
    sxaddpar(h1,'TRIMSEC ', '[1:2048,1:2056]')
    sxaddpar(h1,'FILENAME', infile)

    h1.add_history('m2fs_biassubtract: Subtracted bias on a per column basis')
    h1.add_history('m2fs_4amp: Merged 4 amplifiers into single frame')

    ## when writing, flip the array here (gets unflipped in write_fits_two)
    ## This keeps conventions in a way that makes sense later.
    #hdu1 = fits.PrimaryHDU(outim, h1)
    #hdu2 = fits.ImageHDU(outerr)
    #hdulist = fits.HDUList([hdu1, hdu2])
    #hdulist.writeto(outfile, overwrite=True)
    write_fits_two(outfile, outim.T, outerr.T, h1)

def make_multispec(outfname, bands, bandids, header=None):
    assert len(bands) == len(bandids)
    Nbands = len(bands)
    # create output image array
    # Note we have to reverse the order of axes in a fits file
    shape = bands[0].shape
    output = np.zeros((Nbands, shape[1], shape[0]))
    for k, (band, bandid) in enumerate(zip(bands, bandids)):
        output[k] = band.T
    if Nbands == 1:
        output = output[0]
        klist = [1,2]
        wcsdim = 2
    else:
        klist = [1,2,3]
        wcsdim = 3
    
    hdu = fits.PrimaryHDU(output)
    header = hdu.header
    for k in klist:
        header.append(("CDELT"+str(k), 1.))
        header.append(("CD{}_{}".format(k,k), 1.))
        header.append(("LTM{}_{}".format(k,k), 1))
        header.append(("CTYPE"+str(k), "MULTISPE"))
    header.append(("WCSDIM", wcsdim))
    for k, bandid in enumerate(bandids):
        header.append(("BANDID{}".format(k+1), bandid))
    # Do not fill in wavelength/WAT2 yet
    
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(outfname, overwrite=True)

def parse_idb(fname):
    with open(fname) as fp:
        lines = fp.readlines()
    istarts = []
    for i,line in enumerate(lines):
        if line.startswith("begin"): istarts.append(i)
    Nlines = len(lines)
    Naper = len(istarts)
    
    iranges = []
    for j in range(Naper):
        if j == Naper-1: iranges.append((istarts[j],Nlines))
        else: iranges.append((istarts[j],istarts[j+1]))
    output = {}
    for j in range(Naper):
        data = lines[iranges[j][0]:iranges[j][1]]
        out = {}
        Nskip = 0
        for i, line in enumerate(data):
            if Nskip > 0:
                Nskip -= 1
                continue
            s = line.split()
            try:
                key = s[0]
                value = s[1]
            except IndexError:
                continue
            if key == "begin": pass
            elif key in ["id", "task","units","function"]:
                out[key] = value
            elif key == "image":
                out[key] = " ".join(s[1:])
            elif key in ["aperture","order","naverage","niterate"]:
                out[key] = int(value)
            elif key in ["aplow","aphigh","low_reject","high_reject","grow"]:
                out[key] = float(value)
            elif key in ["features"]:
                Nfeatures = int(value)
                Nskip = Nfeatures
                linelist = list(map(lambda x: list(map(float, x.split()[:6])), data[(i+1):(i+1+Nfeatures)]))
                out["features"] = np.array(linelist)
            elif key in ["coefficients"]:
                Ncoeff = int(value)
                Nskip = Ncoeff
                coeffarr = list(map(float, data[i+1:i+1+Ncoeff]))
                out["coefficients"] = np.array(coeffarr)
        if "aperture" in out:
            aperture = out["aperture"]
        else:
            im = out["image"]
            aperture = int(re.findall(r".*\[\*,(\d+)\]", im)[0])
        output[aperture]=out

    return output

def calc_wave(x, coeffs):
    # only supports legendre poly right now
    functype, order, xmin, xmax = coeffs[0:4]
    coeffs = coeffs[4:]
    assert functype == 2, functype
    assert order == len(coeffs), (order, len(coeffs))
    if order > 5:
        print("Cutting order down to 5 from {}".format(order))
        order = 5
        coeffs = coeffs[:5]
    #if x is None:
    #    x = np.arange(int(np.ceil(xmin)), int(np.floor(xmax)))
    #else:
    #    assert np.all(np.logical_and(x >= xmin, x <= xmax))
    
    # IRAF is one-indexed, python is zero-indexed
    x = x.copy() + 1
    xn = (2*x - (xmax+xmin))/(xmax-xmin)
    wl = np.polynomial.legendre.legval(xn, coeffs)
    return wl

def calc_wave_corr(xarr, coeffs, framenum, ipeak, pixcortable=None):
    """
    Calc wave, but with offset from the default arc
    """
    if pixcortable is None: pixcortable = np.load("arc1d/fit_frame_offsets.npy")
    pixcor = pixcortable[framenum,ipeak]
    # The correction is in pixels. We apply a zero-point offset based on this.
    wave = calc_wave(xarr, coeffs)
    dwave = np.nanmedian(np.diff(wave))
    wavecor = pixcor * dwave
    return wave + wavecor

def load_frame2arc():
    return np.loadtxt("frame_to_arc.txt",delimiter=',').astype(int)
    

import glob, os, sys
import pandas as pd
from astropy.io import ascii, fits
from astropy.table import Table
def get_exptype(header):
    exptype = header["EXPTYPE"]
    object = header["OBJECT"].lower()
    if "tharne" in object:
        exptype = "Comp"
    elif "b1kw4k" in object:
        exptype = "Flat"
    return exptype

def make_db_file(rawdir, dbname=None):
    rawdir = os.path.abspath(rawdir)
    if dbname is None:
        dbname = os.path.join(os.path.dirname(rawdir),os.path.basename(rawdir)+"M2FS.db")
    else:
        dbname = os.path.abspath(dbname)
    fnames = glob.glob(os.path.join(rawdir,"*c1.fits"))
    
    colheads = ["FILE","INST","CONFIG","FILTER","SLIT","BIN","SPEED","NAMP","DATE","UT-START","UT-END","MJD","EXPTIME","EXPTYPE","OBJECT"]
    alldata = []
    for fname in fnames:
        with fits.open(fname) as hdul:
            header = hdul[0].header
        fname = os.path.abspath(fname)[:-7]
        instrument = "-".join([header["INSTRUME"],header["SHOE"],header["SLIDE"]])
        config = header["CONFIGFL"]
        filter = header["FILTER"]
        slitname = header["SLITNAME"].replace(" ","")
        #plate = header["PLATE"]
        binning = header["BINNING"]
        speed = header["SPEED"]
        nopamps = header["NOPAMPS"]
        
        date = header["UT-DATE"]
        start = header["UT-TIME"]
        end = header["UT-END"]
        mjd = header["MJD"]
        
        exptime = header["EXPTIME"]
        exptype = get_exptype(header)
        object = header["OBJECT"]
        
        data = [fname, instrument, config, filter, slitname, binning, speed, nopamps, date, start, end, mjd, exptime, exptype, object]
        alldata.append(data)
    tab = Table(rows=alldata, names=colheads)
    tab["EXPTIME"].format = ".1f"
    tab["MJD"].format = ".3f"
    tab.write(dbname,format="ascii.fixed_width_two_line",overwrite=True)



def m2fs_make_master_dark(filenames, outfname, exptime=3600.):
    """
    Make a master dark by taking the median of all dark frames
    """
    # Load data
    master_dark, master_darkerr, headers = m2fs_load_files_two(filenames)
    h = headers[0]
    # Rescale to common exptime
    for k in range(len(filenames)):
        dh = headers[k]
        if dh["EXPTIME"] != exptime:
            master_dark[k] = master_dark[k] * exptime/dh["EXPTIME"]
            master_darkerr[k] = master_darkerr[k] * np.sqrt(dh["EXPTIME"]/exptime)
    # Take median + calculate error
    master_dark = np.median(master_dark, axis=0)
    master_darkerr = np.sqrt(np.sum(master_darkerr**2, axis=0))
        
    _ = h.pop("EXPTIME")
    h["EXPTIME"] = exptime

    write_fits_two(outfname, master_dark, master_darkerr, h)
    print("Created dark frame with texp={} and wrote to {}".format(
            exptime, outfname))
def m2fs_subtract_one_dark(infile, outfile, dark, darkerr, darkheader):
    """ Dark subtraction """
    img, imgerr, header = read_fits_two(infile)
    # Subtract dark
    exptimeratio = header["EXPTIME"]/darkheader["EXPTIME"]
    darksub = img - dark * exptimeratio
    # Adjust the errors
    darksuberr = np.sqrt(imgerr**2 + darkerr**2)
    # Zero negative values: I don't want to do this
    write_fits_two(outfile, darksub, darksuberr, header)
    
def m2fs_make_master_flat(filenames, outfname):
    master_flat, master_flaterr, headers = m2fs_load_files_two(filenames)
    master_flat = np.median(master_flat, axis=0)
    master_flaterr = np.median(master_flaterr, axis=0)
    write_fits_two(outfname, master_flat, master_flaterr, headers[0])
    print("Created master flat and wrote to {}".format(outfname))
def m2fs_trace_orders():
    raise NotImplementedError
