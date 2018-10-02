from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from scipy import optimize, ndimage, spatial, linalg, special, interpolate
import re
import glob, os, sys, time, subprocess

import matplotlib.pyplot as plt

##############
# File I/O
##############
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

def write_fits_one(outfname,d1,h):
    """ Write fits files with one arrays """
    hdu1 = fits.PrimaryHDU(d1.T, h)
    hdulist = fits.HDUList([hdu1])
    hdulist.writeto(outfname, overwrite=True)

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

############################
# Misc math stuff
############################
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
    




##################
# Creating DB file
##################
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


def m2fs_get_calib_files(db, calibconfig):
    objnums = np.unique(calibconfig["sciencenum"])
    flatnums = np.unique(calibconfig["flatnum"])
    arcnums = np.unique(calibconfig["arcnum"])
    
    objtab = tab[tab["EXPTYPE"]=="Object"]
    flattab = tab[tab["EXPTYPE"]=="Flat"]
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    

##################
# Pipeline calls
##################
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

    write_fits_two(outfile, outim.T, outerr.T, h1)

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
def m2fs_parse_fiberconfig(fname):
    with open(fname) as fp:
        # First three lines: Nobj, Nord, trueord
        Nobj = int(fp.readline().strip())
        Nord = int(fp.readline().strip())
        ordlist = list(map(int, fp.readline().strip().split()))
        
        # Next Nord lines: wlmin, wlmax
        ordwaveranges = []
        for j in range(Nord):
            wl1, wl2 = list(map(float, fp.readline().strip().split()))
            ordwaveranges.append([wl1,wl2])
        
        # Next Nord lines: Xmin, Xmax
        ordpixranges = []
        for j in range(Nord):
            X1, X2 = list(map(int, fp.readline().strip().split()))
            ordpixranges.append([X1, X2])
        
        # All other order lines are tetris and fiber info
        lines = fp.readlines()
    lines = list(map(lambda x: x.strip().split(), lines))
    lines = Table(rows=lines, names=["tetris","fiber"])
    return Nobj, Nord, ordlist, ordwaveranges, ordpixranges, lines

def m2fs_get_trace_fnames(fname):
    assert fname.endswith(".fits")
    dir = os.path.dirname(fname)
    name = os.path.basename(fname)[:-5]
    fname1 = os.path.join(dir, name+"_tracecoeff.txt")
    fname2 = os.path.join(dir, name+"_tracestdcoeff.txt")
    return fname1, fname2
def m2fs_load_trace_function(flatname, fiberconfig):
    """
    Define a function y(iobj, iorder, x) using prefit functions
    iobj = 0 starts from y ~ 0 (the bottom of the frame in DS9)
    """
    fin, _2 = m2fs_get_trace_fnames(flatname)
    coeff1 = np.loadtxt(fin).T
    Nobjs, Norders = fiberconfig[0], fiberconfig[1]
    Ntrace = Nobjs*Norders
    assert coeff1.shape[1] == Ntrace
    functions = [np.poly1d(coeff1[:,j]) for j in range(Ntrace)]
    def trace_func(iobj, iorder, x):
        assert 0 <= iobj < Nobjs
        assert 0 <= iorder < Norders
        itrace = iobj*Norders + iorder
        return functions[itrace](x)
    return trace_func
def m2fs_load_tracestd_function(flatname, fiberconfig):
    """
    Define a function ystd(iobj, iorder, x) using prefit functions
    iobj = 0 starts from y ~ 0 (the bottom of the frame in DS9)
    """
    _1, fin = m2fs_get_trace_fnames(flatname)
    coeff2 = np.loadtxt(fin).T
    Nobjs, Norders = fiberconfig[0], fiberconfig[1]
    Ntrace = Nobjs*Norders
    assert coeff2.shape[1] == Ntrace
    functions = [np.poly1d(coeff2[:,j]) for j in range(Ntrace)]
    def tracestd_func(iobj, iorder, x):
        assert 0 <= iobj < Nobjs
        assert 0 <= iorder < Norders
        itrace = iobj*Norders + iorder
        return functions[itrace](x)
    return tracestd_func

def m2fs_trace_orders(fname, fiberconfig,
                      nthresh=2.0, ystart=0, dx=20, dy=5, nstep=10, degree=5, ythresh=500,
                      trace_degree=None, stdev_degree=None,
                      make_plot=True):
    """
    Order tracing by fitting. Adapted from Terese Hansen
    """
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    midx = round(nx/2.)
    thresh = nthresh*np.median(data)
    
    if trace_degree is None: trace_degree = degree
    if stdev_degree is None: stdev_degree = degree

    Nobjs, Norders = fiberconfig[0], fiberconfig[1]
    expected_fibers = Nobjs * Norders
    ordpixranges = fiberconfig[4]
    
    # Find peaks at center of CCD
    auxdata = np.zeros(ny)
    for i in range(ny):
        ix1 = int(np.floor(midx-dx/2.))
        ix2 = int(np.ceil(midx+dx/2.))+1
        auxdata[i] = np.median(data[ix1:ix2,i])
    dauxdata = np.gradient(auxdata)
    yarr = np.arange(ny)
    peak1 = np.zeros(ny, dtype=bool)
    for i in range(ny-1):
        if (dauxdata[i] >= 0) and (dauxdata[i+1] < 0) and (auxdata[i] >= thresh) and (i > ystart):
            peak1[i] = True
    peak1 = np.where(peak1)[0]
    npeak = len(peak1)
    peak = np.zeros(npeak)
    for i in range(npeak):
        ix1 = int(np.floor(peak1[i]-dy/2.))
        ix2 = int(np.ceil(peak1[i]+dy/2.))+1
        auxx = yarr[ix1:ix2]
        auxy = auxdata[ix1:ix2]
        coef = gaussfit(auxx, auxy, [auxdata[peak1[i]], peak1[i], 2, thresh/2.])
        peak[i] = coef[1]
    assert npeak==expected_fibers
    # TODO allow some interfacing of the parameters and plotting
    
    ## FIRST TRACE: do in windows
    # Trace peaks across dispersion direction
    ypeak = np.zeros((nx,npeak))
    ystdv = np.zeros((nx,npeak))
    nopeak = np.zeros((nx,npeak))
    ypeak[midx,:] = peak
    start = time.time()
    for i in range(npeak):
        iobj = i // Norders
        iord = i % Norders
        
        sys.stdout.write("\r")
        sys.stdout.write("TRACING FIBER {} of {}".format(i+1,npeak))
        # Trace right
        for j in range(midx+nstep, nx, nstep):
            ix1 = int(np.floor(j-dx/2.))
            ix2 = int(np.ceil(j+dx/2.))+1
            auxdata0 = np.median(data[ix1:ix2,:], axis=0)
            #auxdata0 = np.zeros(ny)
            #for k in range(ny):
            #    auxdata0[i] = np.median(data[ix1:ix2,k])
            auxthresh = 2*np.median(auxdata0)
            ix1 = max(0,int(np.floor(ypeak[j-nstep,i]-dy/2.)))
            ix2 = min(ny,int(np.ceil(ypeak[j-nstep,i]+dy/2.))+1)
            auxx = yarr[ix1:ix2]
            auxy = auxdata0[ix1:ix2]
            # stop tracing orders that run out of signal
            if (data[j,int(ypeak[j-nstep,i])] <= data[j-nstep,int(ypeak[j-2*nstep,i])]) and \
               (data[j,int(ypeak[j-nstep,i])] <= ythresh) or \
               j > ordpixranges[iord][1]:
                break
            if np.max(auxy) >= auxthresh:
                try:
                    coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j-nstep,i])], ypeak[j-nstep,i], 2, thresh/2.],
                                    xtol=1e-6,maxfev=10000)
                    ypeak[j,i] = coef[1]
                    ystdv[j,i] = min(coef[2],dy/2.)
                except RuntimeError:
                    ypeak[j,i] = ypeak[j-nstep,i] # so i don't get lost
                    ystdv[j,i] = ystdv[j-nstep,i]
                    nopeak[j,i] = 1
            else:
                ypeak[j,i] = ypeak[j-nstep,i] # so i don't get lost
                ystdv[j,i] = ystdv[j-nstep,i]
                nopeak[j,i] = 1
        # Trace left
        for j in range(midx-nstep, int(np.ceil(dx/2.))+1, -1*nstep):
            #auxdata0 = np.zeros(ny)
            ix1 = int(np.floor(j-dx/2.))
            ix2 = min(nx, int(np.ceil(j+dx/2.))+1)
            auxdata0 = np.median(data[ix1:ix2,:], axis=0)
            #for k in range(ny):
            #    auxdata0[i] = np.median(data[ix1:ix2,k])
            auxthresh = 2*np.median(auxdata0)
            ix1 = int(np.floor(ypeak[j+nstep,i]-dy/2.))
            ix2 = min(ny, int(np.ceil(ypeak[j+nstep,i]+dy/2.))+1)
            auxx = yarr[ix1:ix2]
            auxy = auxdata0[ix1:ix2]
            # stop tracing orders that run out of signal
            if (data[j,int(ypeak[j+nstep,i])] <= data[j+nstep,int(ypeak[j+2*nstep,i])]) and \
               (data[j,int(ypeak[j+nstep,i])] <= ythresh) or \
               j < ordpixranges[iord][0]:
                break
            if np.max(auxy) >= auxthresh:
                try:
                    coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j+nstep,i])], ypeak[j+nstep,i], 2, thresh/2.],
                                    xtol=1e-6,maxfev=10000)
                    ypeak[j,i] = coef[1]
                    ystdv[j,i] = min(coef[2], dy/2.)
                except RuntimeError:
                    ypeak[j,i] = ypeak[j+nstep,i] # so i don't get lost
                    ystdv[j,i] = ystdv[j+nstep,i]
                    nopeak[j,i] = 1
            else:
                ypeak[j,i] = ypeak[j+nstep,i] # so i don't get lost
                ystdv[j,i] = ystdv[j+nstep,i]
                nopeak[j,i] = 1
    ypeak[(nopeak == 1) | (ypeak == 0)] = np.nan
    ystdv[(nopeak == 1) | (ypeak == 0)] = np.nan
    print("\nTracing took {:.1f}s".format(time.time()-start))
    
    coeff = np.zeros((trace_degree+1,npeak))
    coeff2 = np.zeros((stdev_degree+1,npeak))
    for i in range(npeak):
        sel = np.isfinite(ypeak[:,i]) #np.where(ypeak[:,i] != -666)[0]
        xarr_fit = np.arange(nx)[sel]
        #auxcoeff = np.polyfit(xarr_fit, ypeak[sel,i], trace_degree)
        _, auxcoeff = jds_poly_reject(xarr_fit, ypeak[sel,i], trace_degree, 5, 5)
        coeff[:,i] = auxcoeff
        #auxcoeff2 = np.polyfit(xarr_fit, ystdv[sel,i], stdev_degree)
        _, auxcoeff2 = jds_poly_reject(xarr_fit, ystdv[sel,i], stdev_degree, 5, 5)
        coeff2[:,i] = auxcoeff2

    fname1, fname2 = m2fs_get_trace_fnames(fname)
    print(fname,fname1,fname2)
    np.savetxt(fname1, coeff.T)
    np.savetxt(fname2, coeff2.T)
    
    if make_plot:
        ## Trace Image
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(data.T, origin="lower")
        xarr = np.arange(2048)
        #for j in range(npeak):
        #    ax.plot(xarr, np.polyval(coeff[:,j],xarr), color='orange', lw=.5)
        #    ax.errorbar(xarr, ypeak[:,j], yerr=ystdv[:,j], fmt='r.', ms=1, elinewidth=.5)
        tracefn = m2fs_load_trace_function(fname, fiberconfig)
        trstdfn = m2fs_load_tracestd_function(fname, fiberconfig)
        for i in range(Nobjs):
            for j in range(Norders):
                yarr = tracefn(i,j,xarr)
                eyarr = trstdfn(i,j,xarr)
                ax.plot(xarr, yarr, color='orange', lw=.5)
                #ax.errorbar(xarr[::nstep], yarr[::nstep], yerr=eyarr[::nstep], fmt='r.', ms=1., elinewidth=.5)
                ipeak = i*Norders + j
                finite = np.isfinite(ypeak[:,ipeak])
                ax.errorbar(xarr[finite], yarr[finite], yerr=eyarr[finite], fmt='r.', ms=1., elinewidth=1)
                ax.errorbar(xarr, ypeak[:,ipeak], yerr=ystdv[:,ipeak], fmt='co', ms=1, elinewidth=.5, ecolor='c')
        fig.savefig("{}/{}_traceimg.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
        plt.close(fig)
        
        ## Trace Residuals
        fig, axes = plt.subplots(1,2,figsize=(16,6))
        for ax in axes:
            ax.axhline(0,color='k',ls=':')
        for iobj in range(Nobjs):
            for iord in range(Norders):
                ipeak = iobj*Norders + iord
                finite = np.isfinite(ypeak[:,ipeak])
                axes[0].plot(xarr[finite], ypeak[finite,ipeak] - tracefn(iobj, iord, xarr[finite]),'.-')
                axes[1].plot(xarr[finite], ystdv[finite,ipeak] - trstdfn(iobj, iord, xarr[finite]),'.-')
        axes[0].set_title("ypeak residual")
        axes[1].set_title("ystdv residual")
        fig.savefig("{}/{}_traceresid.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    bbox_inches="tight")
        plt.close(fig)
            
        fig, ax = plt.subplots(figsize=(8,8))
        stdevs = np.zeros((nx,npeak))
        #for j in range(npeak):
        #    stdevs[:,j] = np.polyval(coeff2[:,j],xarr)
        for i in range(Nobjs):
            for j in range(Norders):
                stdevs[:,i*Norders+j] = trstdfn(i,j,xarr)
        im = ax.imshow(stdevs.T, origin="lower", aspect='auto')
        fig.colorbar(im)
        fig.savefig("{}/{}_stdevs.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
        plt.close(fig)

def medsubtract(fin, fout, size=15):
    d, e, h = read_fits_two(fin)
    m = ndimage.median_filter(d, size=size)
    write_fits_one(fout, d-m, h)
    
def m2fs_wavecal_find_sources_one_arc(fname, workdir):
    assert fname.endswith(".fits")
    dir = os.path.dirname(fname)
    name = os.path.basename(fname)[:-5]
    thispath = os.path.dirname(__file__)
    sourcefind_path = os.path.join(thispath,"source_finding")
    medfiltname = os.path.join(dir,name+"m.fits")
    
    ## Median filter
    if not os.path.exists(medfiltname):
        print("Running median filter on {}".format(fname))
        medsubtract(fname, medfiltname)
    ## Other filters could be done here, if needed...
    
    ## Run sextractor from the shell
    cmd = "sex {0:} -c {1:}/batch_config.sex -parameters_name {1:}/default.param -filter_name {1:}/default.conv -catalog_name {2:}_sources.cat -checkimage_type OBJECTS -checkimage_name {2:}_chkobj.fits".format(medfiltname, sourcefind_path, os.path.join(workdir, name+"m"))
    subprocess.run(cmd, shell=True)
    
def m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources, max_match_dist=2.0):
    ## Load positions of previously identified sources
    identified_positions = np.vstack([identified_sources["X"],identified_sources["Y"]]).T
    
    ## Load current source positions
    name = os.path.basename(fname)[:-5]
    source_catalog_fname = os.path.join(workdir, name+"m_sources.cat")
    sources = Table.read(source_catalog_fname,hdu=2)
    # Allow neighbors and blends, but not anything else bad
    iigoodflag = sources["FLAGS"] < 4
    sources = sources[iigoodflag]
    Xs = sources["XWIN_IMAGE"] - 1.0
    Ys = sources["YWIN_IMAGE"] - 1.0
    source_positions = np.vstack([Xs,Ys]).T
    print("Identifying sources in {} ({}/{} have good flags)".format(
            source_catalog_fname, len(sources), len(iigoodflag)))
    
    ## Match identified sources to actual sources, approximate
    kdtree = spatial.KDTree(source_positions)
    distances, indexes = kdtree.query(identified_positions, distance_upper_bound=max_match_dist)
    finite = np.isfinite(distances)
    num_matched = np.sum(finite)
    iobjs = identified_sources[finite]["iobj"]
    iorders = identified_sources[finite]["iorder"]
    itetris = identified_sources[finite]["itetris"]
    Ycs = identified_sources[finite]["Ycen"]
    Ls = identified_sources[finite]["L"]
    Xi = Xs[indexes[finite]]
    Yi = Ys[indexes[finite]]
    
    ## Transform identified points to actual points
    ## Right now just a mean zero-point offset in X and Y, no scaling/rotation
    ## TODO can update to affine transformation or CPD, depending on if it works or not
    dX = biweight_location(Xi - identified_sources[finite]["X"])
    dY = biweight_location(Yi - identified_sources[finite]["Y"])
    print("Mean overall shift: dX={:.3f} dY={:.3f}".format(dX, dY))
    if np.isnan(dX): dX = 0.0; print("biweight_location failed, setting dX=0")
    if np.isnan(dY): dY = 0.0; print("biweight_location failed, setting dY=0")
    identified_positions[:,0] = identified_positions[:,0] + dX
    identified_positions[:,1] = identified_positions[:,1] + dY
    
    ## Do final match after transformation
    distances, indexes = kdtree.query(identified_positions, distance_upper_bound=max_match_dist)
    finite = np.isfinite(distances)
    num_matched = np.sum(finite)
    assert np.sum(finite) == len(np.unique(indexes[finite])), "Did not match identified sources uniquely!"
    print("Matched {}/{} identified features".format(num_matched, len(finite)))
    
    ## Save out the data
    iobjs = identified_sources[finite]["iobj"]
    iorders = identified_sources[finite]["iorder"]
    itetris = identified_sources[finite]["itetris"]
    Ycs = identified_sources[finite]["Ycen"] + dY
    Ls = identified_sources[finite]["L"]
    Xi = Xs[indexes[finite]]
    Yi = Ys[indexes[finite]]
    data = Table([iobjs,iorders,itetris,Ycs,Ls,Xi,Yi],
                 names=["iobj","iorder","itetris","Ycen","L","X","Y"])
    
    data.write(os.path.join(workdir, name+"_wavecal_id.txt"), format="ascii", overwrite=True)
    

def make_wavecal_feature_matrix(iobj,iord,X,Y,
                                Nobj,trueord,
                                Xmin,Xmax,
                                Ymin,Ymax,
                                deg):
    """
    Convert identified features into a feature matrix.
    
    Parameterization:
    lambda = fL(nord, X, Y) + L0(iobj)
    <Y> = fY(nord, X, Y) + <Y0>(iobj)
    fL and fY are 3D legendre polynomials, X0 and Y0 are offsets for each object.
    
    First 4 variables are for each identified feature in the arc:
    iobj = object number of the feature, from 0 to Nobj-1
    iord = index of feature order number (into trueord to get actual nord)
           from 0 to len(trueord)-1
    X = pixel X
    Y = pixel Y at reference location (for a given iobj)
    
    Next variables are used for normalization.
    Nobj: total number of objects
    trueord: map from iord to nord
    Xmin, Xmax, Ymin, Ymax: used to standardize X and Y from -1 to 1
    
    degree is a 3-length tuple/list of integers, passed to np.polynomial.legendre.legvander3d
    
    Note: IRAF ecidentify describes the basic functional form of order number.
    Let yord = offset +/- iord (to get the true order number)
    Then fit lambda = f(x, yord)/yord
    This means that wave -> wave*yord is a better variable to fit when determining x
    """
    try: N1 = len(iobj)
    except TypeError: N1 = 1
    try: N2 = len(iord)
    except TypeError: N2 = 1
    try: N3 = len(X)
    except TypeError: N3 = 1
    try: N4 = len(Y)
    except TypeError: N4 = 1
    N = np.max([N1,N2,N3,N4])
    
    if N1 == 1: iobj = np.full(N, iobj)
    if N2 == 1: iord = np.full(N, iord)
    if N3 == 1: X = np.full(N, X)
    if N4 == 1: Y = np.full(N, Y)
    
    #try: NXmins = len(Xmins)
    #except TypeError: NXmins = np.full(Xmins, len(trueord))
    #else: assert len(Xmins)==len(trueord)
    #try: NXmaxs = len(Xmaxs)
    #except TypeError: NXmaxs = np.full(Xmaxs, len(trueord))
    #else: assert len(Xmaxs)==len(trueord)
    
    def center(x,xmin,xmax):
        """ Make x go from -1 to 1 """
        x = np.array(x)
        assert np.all(np.logical_or(x >= xmin, np.isnan(x)))
        assert np.all(np.logical_or(x <= xmax, np.isnan(x)))
        xmean = (xmax+xmin)/2.
        xdiff = (xmax-xmin)/2.
        return (x - xmean)/xdiff
    
    # Normalize coordinates to -1, 1
    Xn = center(X, Xmin, Xmax)
    Yn = center(Y, Ymin, Ymax)
    
    # Normalize order number to -1, 1
    cord = np.array(trueord)[iord]
    cordmin, cordmax = np.min(trueord), np.max(trueord)
    cordnorm = center(cord,cordmin,cordmax)
    
    ## Convert wave -> wave * cord
    #waveord = cord * wave
    #waveordmin = wavemin * np.min(trueord)
    #waveordmax = wavemax * np.max(trueord)
    #waveordnorm = center(waveord, waveordmin, waveordmax)
    
    # Legendre(ycen, nord, wave*nord, degrees)
    legpolymat = np.polynomial.legendre.legvander3d(Xn, Yn, cordnorm, deg)
    legpolymat = legpolymat[:,1:] # Get rid of the constant offset here
    # Add a constant offset for each object with indicator variables
    indicatorpolymat = np.zeros((len(legpolymat), Nobj))
    for it in range(Nobj):
        indicatorpolymat[iobj==it,it] = 1.
    return np.concatenate([legpolymat, indicatorpolymat], axis=1)

def m2fs_wavecal_fit_solution_one_arc(fname, workdir, fiberconfig,
                                      flatfname=None, make_plot=True):
    ## parameters of the fit. Should do this eventually with a config file, kwargs, or something.
    maxiter = 5
    min_lines_per_order = 10
    sigma = 3.
    ycenmin = 0.
    ycenmax = 2055.
    deg = [3,4,5]
    Xmin,Xmax = 0., 2048-1.
    #Xmins = [700, 0, 0, 0]
    #Xmaxs = [2047, 2047, 2047, 1300]
    Ymin,Ymax = 0., 2056-1.
    
    ## Load identified wavelength features
    name = os.path.basename(fname)[:-5]
    data = ascii.read(os.path.join(workdir, name+"_wavecal_id.txt"))
    outfname = os.path.join(workdir, name+"_wavecal_fitdata.npy")
    print("Fitting wavelength solution for {} ({} features)".format(fname, len(data)))
    
    ## TODO GET ALL THESE FROM FIBERCONFIG
    Nobj, Norder, trueord = fiberconfig[0], fiberconfig[1], fiberconfig[2]
     
   ## Construct feature matrix to fit
    Xmat = make_wavecal_feature_matrix(data["iobj"],data["iorder"],data["X"],data["Y"],
                                       Nobj, trueord, Xmin, Xmax, Ymin, Ymax, deg)
    good = np.ones(len(Xmat), dtype=bool)
    #ymat = np.vstack([data["L"],data["Y"]]).T
    ymat = data["L"][:,np.newaxis]
    
    ## Solve for fit with all features
    pfit, residues, rank, svals = linalg.lstsq(Xmat, ymat)
    
    ## Iterative sigma clipping and refitting
    for iter in range(maxiter):
        print("Iteration {}: {} features to start".format(iter+1, np.sum(good)))
        ## Find outliers in each order
        ## I decided to do it this way for easier interpretability, rather than
        ## a global sigma-clip rejection
        yfitall = Xmat.dot(pfit)
        #dX, dY = (ymat - yfitall).T
        dX, = (ymat - yfitall).T
        to_clip = np.zeros(len(Xmat), dtype=bool)
        for iobj in range(Nobj):
            for iorder in range(Norder):
                iiobjorder = np.logical_and(data["iobj"]==iobj,
                                            data["iorder"]==iorder)
                iiobjorderused = np.logical_and(iiobjorder, good)
                
                # Skip if too few lines
                if np.sum(iiobjorderused) <= min_lines_per_order: continue
                # TODO: add back lines if less
                
                ## Find outliers and sigma clip them
                #tdX, tdY = dX[iiobjorderused], dY[iiobjorderused]
                #dXcen, dXstd = biweight_location(tdX), biweight_scale(tdX)
                #dYcen, dYstd = biweight_location(tdY), biweight_scale(tdY)
                #
                #to_clip_X = np.logical_and(iiobjorderused, np.abs((dX-dXcen)/dXstd) > sigma)
                #to_clip_Y = np.logical_and(iiobjorderused, np.abs((dY-dYcen)/dYstd) > sigma)
                #if np.sum(iiobjorderused) - np.sum(to_clip_X | to_clip_Y) < min_lines_per_order:
                #    print("    Could not clip {}/{} features from obj {} order {}".format(
                #            np.sum(to_clip_X | to_clip_Y), np.sum(iiobjorderused), iobj, iorder))
                #    continue
                #else:
                #    to_clip = np.logical_or(to_clip, np.logical_or(to_clip_X, to_clip_Y))
                
                tdX = dX[iiobjorderused]
                dXcen, dXstd = biweight_location(tdX), biweight_scale(tdX)
                
                to_clip_X = np.logical_and(iiobjorderused, np.abs((dX-dXcen)/dXstd) > sigma)
                if np.sum(iiobjorderused) - np.sum(to_clip_X) < min_lines_per_order:
                    print("    Could not clip {}/{} features from obj {} order {}".format(
                            np.sum(to_clip_X), np.sum(iiobjorderused), iobj, iorder))
                    continue
                else:
                    to_clip = np.logical_or(to_clip, to_clip_X)
                
        good = np.logical_and(good, np.logical_not(to_clip))
        print("  Cut down to {} features".format(np.sum(good)))
        
        ## Rerun fit on valid features
        ## this Xmat, this Ymat
        tXmat = Xmat[good]
        tymat = ymat[good]
        pfit, residues, rank, svals = linalg.lstsq(tXmat, tymat)
    
    ## Calculate the RMSs
    yfitall = Xmat.dot(pfit)
    Lrmsarr = np.zeros((Nobj,Norder))
    for iobj in range(Nobj):
        for iorder in range(Norder):
            iiobjorder = np.logical_and(data["iobj"]==iobj,
                                        data["iorder"]==iorder)
            iiobjorderused = np.logical_and(iiobjorder, good)
            Luse = data["L"][iiobjorderused]-yfitall[iiobjorderused,0]
            Xuse = data["X"][iiobjorderused]
            Lrmsarr[iobj,iorder] = np.std(Luse)
    ## Information to save:
    np.save(outfname, [pfit, good, Xmat, data["iobj"], data["iorder"], trueord, Nobj, maxiter, deg, sigma, Lrmsarr,
                       (Xmin,Xmax), (Ymin,Ymax)])
    
    if make_plot:
        yfitall = Xmat.dot(pfit)
        fig, axes = plt.subplots(Nobj,Norder, figsize=(Norder*4,Nobj*3))
        figfname = os.path.join(workdir, name+"_wavecal_fitdata.png")
        for iobj in range(Nobj):
            for iorder in range(Norder):
                ax = axes[iobj, iorder]
                iiobjorder = np.logical_and(data["iobj"]==iobj,
                                            data["iorder"]==iorder)
                iiobjorderused = np.logical_and(iiobjorder, good)
                Lall = data["L"][iiobjorder]-yfitall[iiobjorder,0]
                Xall = data["X"][iiobjorder]
                #Yall = data["Y"][iiobjorder]-yfitall[iiobjorder,1]
                Nall = len(Lall)
                Luse = data["L"][iiobjorderused]-yfitall[iiobjorderused,0]
                Xuse = data["X"][iiobjorderused]
                #Yuse = data["Y"][iiobjorderused]-yfitall[iiobjorderused,1]
                Nuse = len(Luse)
                ax.axhline(0,color='k',linestyle=':')
                ax.set_ylim(-1,1)
                l, = ax.plot(Xall, Lall, '.')
                ax.plot(Xuse, Luse, 'o', color=l.get_color())
                #l, = ax.plot(Xall, Yall, '.')
                #ax.plot(Xuse, Yuse, 'o', color=l.get_color())
                Lrms = np.std(Luse)
                #Yrms = np.std(Yuse)
                #ax.set_title("N={}, used={} Lstd={:.3f} Ystd={:.2f}".format(
                #        Nall,Nuse,Lrms,Yrms))
                ax.set_title("N={}, used={} Lstd={:.3f}".format(
                        Nall,Nuse,Lrms))
        fig.savefig(figfname, bbox_inches="tight")
        plt.close(fig)

        figfname = os.path.join(workdir, name+"_wavecal_Lrmshist.png")
        fig, axes = plt.subplots(2,2, figsize=(12,12))
        axes[0,0].hist(np.ravel(Lrmsarr), bins="auto")
        axes[0,0].set_xlabel("Lrms")
        axes[0,0].set_ylabel("N")

        for iobj in range(Nobj):
            axes[0,1].plot(trueord, Lrmsarr[iobj,:], "o-")
        axes[0,1].set_xlabel("Order")
        axes[0,1].set_ylabel("Lrms")

        for iord in range(Norder):
            axes[1,1].plot(np.arange(Nobj), Lrmsarr[:,iord], "o-", label=str(trueord[iord]))
        axes[1,1].set_xlabel("Object")
        axes[1,1].set_ylabel("Lrms")
        axes[1,1].legend(fontsize=8, framealpha=.5)
        
        ax = axes[1,0]
        if flatfname is not None:
            tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
            ax.set_title("Y from trace")
        else:
            ax.set_title("Y = Ycen at X=1023")
        for iobj in range(Nobj):
            for iord in range(Norder):
                Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
                if flatfname is None:
                    Yarr = np.unique(data["Ycen"][np.logical_and(data["iobj"]==iobj, data["iorder"]==iord)])[0]
                else:
                    Yarr = tracefn(iobj, iord, Xarr)
                Xmat = make_wavecal_feature_matrix(iobj, iord, Xarr, Yarr,
                                                   Nobj, trueord, Xmin, Xmax, Ymin, Ymax, deg)
                Lfit = Xmat.dot(pfit)
                ax.plot(Xarr, Lfit, lw=.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Lambda")
        fig.savefig(figfname, bbox_inches="tight")
        plt.close(fig)

def m2fs_get_pixel_functions(flatfname, arcfname, fiberconfig):
    """
    Get the functions Lambda(iobj, iord, X, Y) and ys(iobj, iord, X, Y)

    For each iobj and iord:
    ys(X, Y) = (Y - <y>(X))/(<std_y>(X))
      where <y> and <std_y> were determined while tracing the flat
    Lambda(X, Y) = f(iobj, iord, X, Y) = constant(iobj) + LegPoly(iord, X, Y)
      is determined from the arc
    """
    
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    trstdfn = m2fs_load_tracestd_function(flatfname, fiberconfig)
    def ys(iobj, iord, X, Y):
        yarr = tracefn(iobj,iord,X)
        sarr = trstdfn(iobj,iord,X)
        return (Y-yarr)/(sarr)
    
    workdir = os.path.dirname(arcfname)
    name = os.path.basename(arcfname)[:-5]
    fitdata = np.load(os.path.join(workdir, name+"_wavecal_fitdata.npy"))
    pfit, trueord, Nobj, deg, (Xmin, Xmax), (Ymin,Ymax) = [fitdata[i] for i in [0, 5, 6, 8, 11, 12]]
    def flambda(iobj, iord, X, Y):
        twodim = len(X.shape)==2
        
        if twodim:
            assert np.all(X.shape == Y.shape)
            shape = X.shape
            X = np.ravel(X)
            Y = np.ravel(Y)
        Xmat = make_wavecal_feature_matrix(iobj, iord, X, Y,
                                           Nobj, trueord, Xmin, Xmax, Ymin, Ymax, deg)
        
        out = Xmat.dot(pfit)
        if twodim:
            return out.reshape(shape)
        else:
            return np.ravel(out)
    return ys, flambda


def make_ghlb_y_matrix(iobj, iord, ys, R, eR,
                       yscut = 4.0):
    """
    Get data, errors, good mask, indices of original array
    Cut on |ys| < yscut to make the matrix managable in size
    """
    assert ys.shape == R.shape, (ys.shape, R.shape)
    assert ys.shape == eR.shape, (ys.shape, R.shape)
    indices = np.where(np.abs(ys) < yscut)
    Rout = np.ravel(R[indices])
    eRout = np.ravel(eR[indices])
    return Rout, eRout, indices
def make_ghlb_feature_matrix(iobj, iord, L, ys, Sprime,
                             fiberconfig, deg):
    """
    Create GHLB feature matrix
    Legendre in wavelength (use fiberconfig and iord to get L limits)
    Gauss-Hermite in ys
    MIKE uses deg=[2,10]
    """
    degL, degH = deg
    NL, NH = degL+1, degH+1
    Nparam = NL*NH
    L = np.ravel(L)
    ys = np.ravel(ys)
    assert len(L)==len(ys)
    assert len(L)==len(Sprime)
    N = len(L)
    
    ## Normalize wavelengths
    Lmin, Lmax = fiberconfig[3][iord]
    Lcen = (Lmax+Lmin)/2.
    Lwid = (Lmax-Lmin)/2.
    Ln = (L-Lcen)/Lwid
    ## Construct features: legendre x gauss-hermite
    La = [special.eval_legendre(a, Ln)*Sprime for a in range(NL)]
    expys = np.exp(-ys**2/2.)
    Hb = [special.eval_hermitenorm(b, ys)*expys for b in range(NH)]
    polygrid = np.zeros((N,Nparam))
    for a in range(NL):
        for b in range(NH):
            polygrid[:,b+a*NH] = La[a]*Hb[b]
    return polygrid

def fit_S_with_profile(P, L, R, eR, Npix, dx=0.1):
    """ R(L,y) = S(L) * P(L,y) """
    RP = R/P
    W = (P/eR)**2.
    W = W/W.sum()
    Lmin, Lmax = L.min()+dx, L.max()-dx
    knots = np.linspace(Lmin, Lmax, Npix)[1:-1] # LSQUnivariateSpline adds two end knots
    ## Fit B Spline
    iisort = np.argsort(L)
    Sfunc = interpolate.LSQUnivariateSpline(L[iisort], RP[iisort], knots, W[iisort])
    return Sfunc
def fit_Sprime(ys, L, R, eR, Npix, ysmax=1.0):
    ii = np.abs(ys) < ysmax
    ys, R, eR, L = ys[ii], R[ii], eR[ii], L[ii]
    P = np.exp(-ys**2/2.)
    return fit_S_with_profile(P, L, R, eR, Npix)
    
def m2fs_subtract_scattered_light(fname, flatfname, arcfname, fiberconfig, 
                                  badcols=[], yscut=2.0, deg=[2,2], sigma=5.0, maxiter=10,
                                  verbose=True, make_plot=True):
    """
    The basic idea is to mask out the defined extraction regions in the 2D image,
    then fit a 2D legendre polynomial to the rest of the pixels.
    """
    start = time.time()
    outdir = os.path.dirname(fname)
    assert fname.endswith(".fits")
    name = os.path.basename(fname)[:-5]
    outname = name+"s"
    outfname = os.path.join(outdir, outname+".fits")
    
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    R, eR, header = read_fits_two(fname)
    shape = R.shape
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    used = np.zeros_like(R, dtype=bool)
    
    # Find all pixels used in extraction
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    for iobj in range(Nobj):
        for iord in range(Norder):
            ys = ysfunc(iobj, iord, X, Y)
            used[np.abs(ys) < yscut] = True
    print("m2fs_subtract_scattered_light: took {:.1f}s to find extracted pixels".format(time.time()-start))
    
    scatlight = R.copy()
    scatlight[used] = np.nan
    
    ## Fit scattered light with iterative rejection
    def normalize(x):
        """ Linearly scale from -1 to 1 """
        x = np.array(x)
        nx = len(x)
        xmin, xmax = x.min(), x.max()
        xhalf = (x.max()-x.min())/2.
        return (x-xhalf)/xhalf
    XN, YN = np.meshgrid(normalize(np.arange(shape[0])), normalize(np.arange(shape[1])), indexing="ij")
    finite = np.isfinite(scatlight)
    _XN = XN[finite].ravel()
    _YN = YN[finite].ravel()
    _scatlight = scatlight[finite].ravel()
    _scatlighterr = eR[finite].ravel()
    _scatlightfit = np.full_like(_scatlight, np.nanmedian(_scatlight)) # initialize fit to constant
    Noutliertotal = 0
    for iter in range(maxiter):
        # Clip outlier pixels
        normresid = (_scatlight - _scatlightfit)/_scatlighterr
        #mu = np.nanmedian(resid)
        #sigma = np.nanstd(resid)
        #iinotoutlier = np.logical_and(resid < mu + sigmathresh*sigma, resid > mu - sigmathresh*sigma)
        iinotoutlier = np.abs(normresid < sigma)
        Noutlier = np.sum(~iinotoutlier)
        if verbose: print("  m2fs_subtract_scattered_light: Iter {} removed {} pixels".format(iter, Noutlier))
        if Noutlier == 0: break
        Noutliertotal += Noutlier
        _XN = _XN[iinotoutlier]
        _YN = _YN[iinotoutlier]
        _scatlight = _scatlight[iinotoutlier]
        _scatlighterr = _scatlighterr[iinotoutlier]
        # Fit scattered light model
        xypoly = np.polynomial.legendre.legvander2d(_XN, _YN, deg)
        coeff = np.linalg.lstsq(xypoly, _scatlight, rcond=-1)[0]
        # Evaluate the scattered light model
        _scatlightfit = xypoly.dot(coeff)
    scatlightpoly = np.polynomial.legendre.legvander2d(XN.ravel(), YN.ravel(), deg)
    scatlightfit = (scatlightpoly.dot(coeff)).reshape(shape)
    
    data = R - scatlightfit
    edata = eR + scatlightfit
    header.add_history("m2fs_subtract_scattered_light: subtracted scattered light")
    header.add_history("m2fs_subtract_scattered_light: degree={}".format(deg))
    header.add_history("m2fs_subtract_scattered_light: removed {} outlier pixels in {} iters".format(Noutliertotal, iter+1))
    write_fits_two(outfname, data, edata, header)
    print("Wrote to {}".format(outfname))
    print("m2fs_scatlight took {:.1f}s".format(time.time()-start))
    
    if make_plot:
        fig2 = plt.figure(figsize=(8,8))
        im = plt.imshow(scatlightfit.T, origin='lower', aspect='auto', interpolation='none')
        #plt.axvline(badcolmin,lw=1,color='k',linestyle=':')
        #plt.axvline(badcolmax,lw=1,color='k',linestyle=':')
        plt.title("Scattered light fit (deg={}, max={})".format(deg,np.nanmax(scatlightfit)))
        plt.colorbar()
        outpre = outdir+"/"+outname
        fig2.savefig(outpre+"_scatlight_fit.png",bbox_inches='tight')
        fig3 = plt.figure(figsize=(8,8))
        resid = scatlight-scatlightfit
        minresid = np.nanpercentile(resid,.1)
        im = plt.imshow(resid.T, origin='lower', aspect='auto', cmap='coolwarm',
                        vmin=minresid, vmax=abs(minresid), interpolation='none')
        #plt.axvline(badcolmin,lw=1,color='k',linestyle=':')
        #plt.axvline(badcolmax,lw=1,color='k',linestyle=':')
        plt.title("Scattered Light Residual (median = {:.2f})".format(np.nanmedian(resid)))
        plt.colorbar()
        fig3.savefig(outpre+"_scatlight_resid.png",bbox_inches='tight')
        fig1 = plt.figure(figsize=(8,8))
        residfrac = (scatlight-scatlightfit)/scatlight
        minresid = np.nanpercentile(residfrac,.1)
        im = plt.imshow(residfrac.T, origin='lower', aspect='auto', cmap='coolwarm',
                        vmin=minresid, vmax=abs(minresid), interpolation='none')
        #plt.axvline(badcolmin,lw=1,color='k',linestyle=':')
        #plt.axvline(badcolmax,lw=1,color='k',linestyle=':')
        plt.title("Scattered Light Relative Residual (med={:.2f})".format(np.nanmedian(residfrac)))
        plt.colorbar()
        fig1.savefig(outpre+"_scatlight_residfrac.png",bbox_inches='tight')
        plt.close(fig1); plt.close(fig2); plt.close(fig3)
    

def fit_ghlb(iobj, iord, fiberconfig,
             X, Y, R, eR,
             ysfunc, Lfunc, deg,
             pixel_spacing=1, yscut = 2.0, maxiter1=5, maxiter2=5, sigma = 5.0):
    """
    Iteratively fit GHLB with outlier rejection (using biweight_scale)
    Input: iobj, iord, fiberconfig
           X (x-pixels), Y (y-pixels), R (data image), eR (error image)
           ysfunc, Lfunc (from m2fs_get_pixel_functions)
           deg (Legendre degree, Hermite degree)
           yscut (sets pixels to use for fitting |ys| < yscut)
           maxiter, sigma (sigma clipping)
    Returns TODO
    
    Algorithm:
    * Initialize S'(L), fit assuming a Gaussian spatial profile
      * Initial guess uses |ys| < 1.0
    * Iterate (up to maxiter1):
      * Fit GHLB, iteratively rejecting cosmic rays with sigma (up to maxiter2)
      * Extract S(L)
    """
    assert len(deg)==2, deg
    
    start = time.time()
    ## Evaluate ys
    ys = ysfunc(iobj, iord, X, Y)
    ## Get arrays of relevant pixels
    Yarr, eYarr, indices = make_ghlb_y_matrix(iobj, iord, ys, R, eR, yscut=yscut)
    this_X = X[indices]
    Xmin, Xmax = fiberconfig[4][iord]
    Npix = (Xmax-Xmin+1)//pixel_spacing
    iiXcut = (Xmin <= this_X) & (this_X <= Xmax)
    indices = tuple([ix[iiXcut] for ix in indices])
    Yarr, eYarr = Yarr[iiXcut], eYarr[iiXcut]
    ys = ys[indices]
    ## Evaluate L
    L = Lfunc(iobj, iord, np.ravel(X[indices]), np.ravel(Y[indices]))

    ## Compute S'(L)
    Sprimefunc = fit_Sprime(ys, L, Yarr, eYarr, Npix)
    Sprime = Sprimefunc(L)
    
    ## Iterate: compute GHLB, reextract S
    S = Sprime
    mask = np.zeros(len(Yarr), dtype=bool)
    lastNmask1 = 0
    for iter in range(maxiter1):
        ## Create GHLB feature matrix
        Xmatprime = make_ghlb_feature_matrix(iobj, iord, L, ys, S,
                                        fiberconfig, deg)
        ## TODO Horne86 says use the model profile + read noise to recompute variance.
        ## The original estimate over-weights downward noise fluctuations!
        
        ## Fit GHLB coefficients with weighted linear least squares
        warr = 1./eYarr
        wXarr = (Xmatprime.T*warr).T
        wYarr = warr*Yarr
        pfit, residues, rank, svals = linalg.lstsq(wXarr, wYarr)
        yfit = Xmatprime.dot(pfit)
        lastNmask2 = np.sum(mask)
        ## Iteratively remove outliers and refit
        for iter2 in range(maxiter2):
            #dY = Yarr - yfit
            #stdev = biweight_scale(dY[~mask])
            normresid = (Yarr-yfit)/eYarr
            mask |= (np.abs(normresid) > sigma) #(np.abs(dY) > sigma * stdev)
            #print("  Iter {}: {}/{} pixels masked".format(iter+1, np.sum(mask), len(mask)))
            pfit, residues, rank, svals = linalg.lstsq(wXarr[~mask], wYarr[~mask])
            yfit = Xmatprime.dot(pfit)
            Nmask = np.sum(mask)
            if lastNmask2 == Nmask: break
            lastNmask2 = Nmask
        
        ## Re-extract S
        Xmat = make_ghlb_feature_matrix(iobj, iord, L, ys, np.ones_like(L),
                                        fiberconfig, deg)
        Pfit = Xmat.dot(pfit)
        Sfunc = fit_S_with_profile(Pfit, L, Yarr, eYarr, Npix)
        S = Sfunc(L)
        print("  Iter {}: {}/{} pixels masked".format(iter+1, np.sum(mask), len(mask)))
        if lastNmask1 == Nmask: break
        lastNmask1 = Nmask
        
    print("Total time took {:.1f}s".format(time.time()-start))
    return pfit, yfit, Yarr, eYarr, Xmat, L, ys, mask, indices, Sprimefunc, Sfunc, Pfit

def m2fs_ghlb_extract(fname, flatfname, arcfname, fiberconfig, yscut, deg, sigma,
                      make_plot=True, make_obj_plots=True):
    """
    """
    outdir = os.path.dirname(fname)
    assert fname.endswith(".fits")
    name = os.path.basename(fname)[:-5]
    outfname1 = os.path.join(outdir,name+"_GHLB.npy")
    outfname2 = os.path.join(outdir,name+"_specs.fits")
    
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    R, eR, header = read_fits_two(fname)
    shape = R.shape
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    
    alloutput = []
    used = np.zeros_like(R)
    modeled_flat = np.zeros_like(R)
    
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    Ntrace = Nobj*Norder
    
    start = time.time()
    for iobj in range(Nobj):
        if make_obj_plots: fig, axes = plt.subplots(Norder, 4, figsize=(4*4,4*Norder))
        for iord in range(Norder):
            print("iobj={} iord={}".format(iobj,iord))
            itrace = iord + iobj*Norder
            output = fit_ghlb(iobj, iord, fiberconfig,
                              X, Y, R, eR,
                              ysfunc, Lfunc, deg,
                              pixel_spacing = 1,
                              yscut = yscut, maxiter1=5, maxiter2=5, sigma = sigma)
            pfit, Rfit, Yarr, eYarr, Xmat, L, ys, mask, indices, Sprimefunc, Sfunc, Pfit = output
            #alloutput.append([pfit,Rfit,Yarr,eYarr,Xmat,L,ys,mask,indices,Sprimefunc,Sfunc,Pfit])
            alloutput.append([pfit,mask,indices,Sfunc,Pfit])
            used[indices] = used[indices] + 1
            modeled_flat[indices] += Rfit
            
            if make_obj_plots:
                ax = axes[iord,0]
                ax.plot(L, Yarr, 'k,')
                Lplot = np.arange(L.min(), L.max()+0.1, 0.1)
                ax.plot(Lplot, Sfunc(Lplot), '-', lw=1)
                ax.set_xlabel("L"); ax.set_ylabel("S(L)")
                ax.set_title("iobj={} iord={}".format(iobj,iord))
                
                ax = axes[iord,1]
                ax.scatter(L,ys,c=Pfit, alpha=.1)
                ax.set_xlabel("L"); ax.set_ylabel("ys")
                ax.set_title("GHLB Profile")
                
                ax = axes[iord,2]
                resid = (Yarr - Rfit)/eYarr
                yslims = [0, 1, 2, np.inf]
                markersizes = [6, 3, 2]
                colors = ['r','orange','k']
                absys = np.abs(ys)
                for i in range(len(yslims)-1):
                    ys1, ys2 = yslims[i], yslims[i+1]
                    ii = np.logical_and(absys >= ys1, absys < ys2)
                    ax.plot(L[ii], resid[ii], 'o', color=colors[i], ms=markersizes[i], alpha=.5, mew=0,
                            label="{:.1f} < ys < {:.1f}".format(ys1,ys2))
                #ax.scatter(L, resid, c=ys, vmin=-yscut, vmax=yscut, alpha=.2, cmap='coolwarm')
                ax.axhline(0, color='r', ls=':')
                ax.legend(fancybox=True)
                ax.set_xlabel("L"); ax.set_ylabel("(R - Rfit)/eR")
                ax.set_ylim(-5,5)
                Nbad = np.sum(np.abs(resid) > 5)
                ax.set_title("Leg={} GH={} chi2r={:.2f}".format(deg[0], deg[1], np.sum(resid**2)/len(resid)))
                
                ax = axes[iord,3]
                ax.hist(resid, bins=np.arange(-5, 5.1,.1), normed=True, histtype='step')
                xplot = np.linspace(-5,5,1000)
                ax.plot(xplot, np.exp(-xplot**2/2.)/np.sqrt(2*np.pi), 'k')
                ax.set_xlabel("(R - Rfit)/eR")
                ax.set_title("{}/{} points outside limits".format(Nbad, len(resid)))
        if make_obj_plots:
            start2 = time.time()
            fig.savefig("{}/{}_GHLB_Obj{:03}.png".format(outdir,name,iobj), bbox_inches="tight")
            print("Saved figure: {:.1f}s".format(time.time()-start2))
            plt.close(fig)
    print("Finished huge fit: {:.1f}s".format(time.time()-start))
    np.save(outfname1, [alloutput, used, modeled_flat])
    write_fits_one("{}/{}_GHLB_used.fits".format(outdir, name), used, header)
    write_fits_one("{}/{}_GHLB_model.fits".format(outdir, name), modeled_flat, header)
    write_fits_one("{}/{}_GHLB_resid.fits".format(outdir, name), R-modeled_flat, header)
    ## TODO make outfname2, which is the extracted spectra evaluated at specific points
    if make_plot:
        fig = plt.figure(figsize=(8,8))
        plt.imshow(used.T, origin="lower")
        plt.title("Pixels used in model")
        plt.colorbar(label="number times used")
        fig.savefig(os.path.join(outdir,name+"_GHLB_used.png"))
        plt.close(fig)
        
        fig = plt.figure(figsize=(8,8))
        plt.imshow(modeled_flat.T, origin="lower")
        plt.title("Model fit")
        plt.colorbar()
        fig.savefig(os.path.join(outdir,name+"_GHLB_fit.png"))
        plt.close(fig)
        
        fig = plt.figure(figsize=(8,8))
        plt.imshow((R - modeled_flat).T, origin="lower", vmin=-200, vmax=+200)
        plt.colorbar()
        plt.title("Data - model")
        fig.savefig(os.path.join(outdir,name+"_GHLB_resid.png"))
        plt.close(fig)
    print("Total time: {:.1f}".format(time.time()-start))
    
