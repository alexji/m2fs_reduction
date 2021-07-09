from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
#from scipy import fft # someday we will update scipy...
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from scipy import optimize, ndimage, spatial, linalg, special, interpolate
from skimage import morphology
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
    if pixcortable is None: pixcortable = np.load("arc1d/fit_frame_offsets.npy", allow_pickle=True)
    pixcor = pixcortable[framenum,ipeak]
    # The correction is in pixels. We apply a zero-point offset based on this.
    wave = calc_wave(xarr, coeffs)
    dwave = np.nanmedian(np.diff(wave))
    wavecor = pixcor * dwave
    return wave + wavecor

def load_frame2arc():
    return np.loadtxt("frame_to_arc.txt",delimiter=',').astype(int)
    
def window_stdev(X, window_size):
    """
    Not tested yet
    https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    """
    c1 = ndimage.filters.uniform_filter(X, window_size, mode='reflect')
    c2 = ndimage.filters.uniform_filter(X*X, window_size, mode='reflect')
    out2 = c2 - c1*c1
    out2[out2 < 0] = 0
    return np.sqrt(out2)

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
    darksuberr = np.sqrt(imgerr**2) # no-var-sub + darkerr**2)
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
        
        # Next line: identification data
        line_identifications = os.path.join(os.path.dirname(__file__), fp.readline().strip())
        # All other order lines are tetris and fiber info
        lines = fp.readlines()
    lines = list(map(lambda x: x.strip().split(), lines))
    lines = Table(rows=lines, names=["tetris","fiber"])
    return Nobj, Nord, ordlist, ordwaveranges, ordpixranges, line_identifications, lines

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

def m2fs_new_trace_orders_multidetect(fname, fiberconfig, detection_scale_factors = [3,4,5,6,7],
                                      **kwargs):
    """ Iterate through different detection scale factors for tracing flats """
    detection_scale_factor = kwargs.pop("detection_scale_factor", None)
    if detection_scale_factor is not None:
        print("Warning: ignoring keyword detection_scale_factor, trying {}".format(detection_scale_factors))
    
    all_good = True
    for detection_scale_factor in detection_scale_factors:
        try:
            m2fs_new_trace_orders(fname, fiberconfig,
                                  detection_scale_factor=detection_scale_factor,
                                  **kwargs)
        except Exception as e:
            print("new trace orders Failed {}".format(detection_scale_factor))
            print(e)
        else:
            break
    else:
        print("ERROR: {} could not be traced! Run this manually (change midx)".format(fname))
        all_good=False
        raise RuntimeError("Could not trace flat")
    return all_good

def m2fs_new_trace_orders(fname, fiberconfig,
                          midx=None,
                          detection_scale_factor=7.0,
                          noise_removal_scale=-1,
                          fiber_count_check=True,
                          degree=5,
                          trace_degree=None,
                          make_plot=True):
    """ New order tracing algorithm for flats """
    start = time.time()
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    if midx is None: midx = round(nx/2.)
    Nobjs, Norders = fiberconfig[0], fiberconfig[1]
    expected_peaks = Nobjs * Norders
    
    if trace_degree is None: trace_degree=degree
    
    # Convenience function for writing intermediate output fits files
    def write_array(suffix, data):
        plotname = "{}/{}_{}.fits".format(os.path.dirname(fname), os.path.basename(fname)[:-5], suffix)
        write_fits_one(plotname, data, header)

    # Find all pixels above the detection threshold
    background = ndimage.median_filter(data, size=11)
    #if make_plot: write_array("background", background)
    data_diff = data - background
    detection_threshold = detection_scale_factor * edata
    detections = ((data_diff - background) > detection_threshold).astype(int)
    # Hot pixel/cosmic noise removal (opening = erosion + dilation)
    if noise_removal_scale > 0:
        selem = morphology.square(noise_removal_scale)
        cleaned_detections = morphology.opening(detections, selem)
        Ndiff = detections.sum() - cleaned_detections.sum()
        print("Removed {} pixels below scale {}".format(Ndiff, noise_removal_scale))
        detections = cleaned_detections
    if make_plot: write_array("detections", detections)
    # Segment detected pixels into traces, using midx as the initial seeds of the watershed algorithm
    trace_peaks = np.where(np.diff(detections[midx]) == -1)[0]
    markers = np.zeros_like(detections, dtype=bool)
    markers[midx, trace_peaks] = True
    markers, num_peaks = ndimage.label(markers)
    if fiber_count_check: assert num_peaks == expected_peaks, "Found {}/{} peaks".format(num_peaks,expected_peaks)
    segmentation = morphology.watershed(detections, markers, mask=detections.astype(bool))
    if make_plot: write_array("segmentation", segmentation)
    
    # Fit traces
    trace_coeffs = np.zeros((num_peaks, trace_degree+1))
    traces_all_x  = []
    traces_all_y  = []
    traces_all_ys = []
    traces_all_iobj = []
    traces_all_iord = []
    traces_all_Rnorm = []
    traces_all_Rnorm_coeffs = np.zeros((num_peaks,6))
    for ipeak in range(num_peaks):
        iobj, iord = ipeak//Norders, ipeak % Norders
        indices_x, indices_y = np.where(segmentation == ipeak + 1)
        # Fit mean trace
        yfit, trace_coeff = jds_poly_reject(indices_x, indices_y, trace_degree, 5, 5)
        trace_coeffs[ipeak,:] = trace_coeff
        ys = indices_y - yfit
        traces_all_x.append(indices_x)
        traces_all_y.append(indices_y)
        traces_all_ys.append(ys)
        traces_all_iobj.append(np.zeros_like(indices_x.astype(int)) + iobj)
        traces_all_iord.append(np.zeros_like(indices_x.astype(int)) + iord)
        # Quick spectrum estimate for width purposes
        R = data[indices_x,indices_y]
        Rfit, coeff = jds_poly_reject(indices_x, R, 5, 5, 5)
        Rnorm = R/np.polyval(coeff, indices_x)
        traces_all_Rnorm.append(Rnorm)
        traces_all_Rnorm_coeffs[ipeak,:] = coeff
        # Note that X is not rectified, so we can't reliably do an X-dependent fit
    trace_output =  [trace_coeffs, traces_all_x, traces_all_y, traces_all_ys, traces_all_iobj, traces_all_iord, traces_all_Rnorm, traces_all_Rnorm_coeffs]
    
    Nparam = 3
    labels = ["A","sigma","exponent"]
    ix_sigma, ix_exponent = 1,2;
    psfexp2_coeffs = np.zeros((Nobjs, Nparam))
    if make_plot:
        ## Plot the actual data to fit and the best-fit
        Ncol = 4
        Nrow = Nobjs // Ncol
        if Ncol*Nrow < Nobjs: Nrow += 1
        fig, axes = plt.subplots(Nrow,Ncol, figsize=(8*Ncol, 6*Nrow))
    for iobj in range(Nobjs):
        p0 = [1.2, 2.3, 2.0]
        ysfit = np.concatenate([traces_all_ys[iobj*Norders + iord] for iord in range(Norders)])
        Rnormfit = np.concatenate([traces_all_Rnorm[iobj*Norders + iord] for iord in range(Norders)])
        popt, pcov = iterfit_psfexp2(ysfit, Rnormfit, p0)
        psfexp2_coeffs[iobj] = popt
        if make_plot:
            ysplot = np.linspace(-3, 3)
            ax = axes.flat[iobj]
            ax.plot(ysfit, Rnormfit, 'k,')
            ax.plot(ysplot, psfexp2(ysplot, *popt), 'r-')
            ax.set_xlim(-3,3)
            ax.set_ylim(0,2)
            ax.set_title(str(iobj))
            ax.set_xlabel("ys = y-ycen")
            ax.set_ylabel("Rnorm = R/<R(x)> (5th degree polyfit)")
    trace_output.append(psfexp2_coeffs)
    if make_plot:
        fig.savefig("{}/{}_{}.png".format(os.path.dirname(fname), os.path.basename(fname)[:-5], "psfexp2fit"),
                    bbox_inches="tight")
        plt.close(fig)
        ## Plot the best-fit psfexp2 parameters
        fig, axes = plt.subplots(1,Nparam,figsize=(6*Nparam,6))
        
        for i in range(Nparam):
            axes[i].plot(psfexp2_coeffs[:,i],'o-')
            axes[i].set_xlabel(labels[i])
        fig.tight_layout(); fig.savefig("{}/{}_{}.png".format(os.path.dirname(fname), os.path.basename(fname)[:-5],
                                                            "psfexp2params"),
                                        bbox_inches="tight")
        ## Forward model and save the residuals
        model = np.zeros_like(data)
        for iobj in range(Nobjs):
            for iord in range(Norders):
                itrace = iobj*Norders + iord
                _all_x = traces_all_x[itrace]
                _all_y = traces_all_y[itrace]
                mask = np.ones_like(_all_x, dtype=bool)
                mask[np.abs(_all_x - biweight_location(_all_x)) > 5*biweight_scale(_all_x)] = False
                mask[np.abs(_all_y - biweight_location(_all_y)) > 5*biweight_scale(_all_y)] = False
                xmin, xmax = np.min(_all_x[mask]), np.max(_all_x[mask])
                ymin, ymax = np.min(_all_y[mask]), np.max(_all_y[mask])
                #print("obj {} ord {}: x={}-{}, y={}-{}".format(iobj, iord, xmin, xmax, ymin, ymax))
                XN, YN = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin,ymax+1), indexing="ij")
                RN = np.polyval(traces_all_Rnorm_coeffs[itrace], XN)
                ysN = YN - np.polyval(trace_coeffs[itrace], XN)
                profileN = psfexp2(ysN, *psfexp2_coeffs[iobj])
                model[XN,YN] = model[XN,YN] + RN*profileN
        write_array("fasttraceresidual", data - model)
        write_array("fasttracemodel", model)
    outfname = "{}/{}_{}.npy".format(os.path.dirname(fname), os.path.basename(fname)[:-5], "fasttrace")
    np.save(outfname, [trace_coeffs, psfexp2_coeffs, traces_all_Rnorm_coeffs])
    fname1, fname2 = m2fs_get_trace_fnames(fname)
    np.savetxt(fname1, trace_coeffs)
    # FWHM of the psfexp2 is 2 sigma (ln5)**(1/exponent)
    # FWHM of gaussian is 2.355 sigma_gauss
    # The second column will just be 0: for now, we will not let sigma_gauss vary across the chip.
    coeff2 = np.zeros((Nobjs*Norders,2))
    for iobj in range(Nobjs):
        for iord in range(Norders):
            itrace = iobj*Norders + iord
            sigma = psfexp2_coeffs[iobj,ix_sigma]
            exponent = psfexp2_coeffs[iobj,ix_exponent]
            sigma_gauss = (2/2.355) * sigma * (np.log(5))**(1/exponent)
            coeff2[itrace,0] = sigma_gauss
    np.savetxt(fname2, coeff2)
    print("m2fs_new_trace_orders took {:.1f}s".format(time.time()-start))
def iterfit_psfexp2(x,y,p0,maxiter=5,sigclip=5):
    iigood = np.ones_like(x, dtype=bool)
    Ngood = iigood.sum()
    for i in range(maxiter):
        _x, _y = x[iigood], y[iigood]
        popt, pcov = optimize.curve_fit(psfexp2, _x, _y, p0)
        yfit = psfexp2(x, *popt)
        ystd = biweight_scale(y - yfit)
        iigood[np.abs(yfit - y) > sigclip*ystd] = False
        if iigood.sum() == Ngood: break
        Ngood = iigood.sum()
    popt, pcov = optimize.curve_fit(psfexp2, x[iigood], y[iigood], p0)
    return popt, pcov
def psfexp2(x, A, sigma, exponent):
    return A * np.exp(-(np.abs(x)/sigma)**exponent)

def m2fs_trace_orders(fname, fiberconfig,
                      midx=None,
                      nthresh=2.0, ystart=0, dx=20, dy=5, nstep=10, degree=5, ythresh=500,
                      trace_degree=None, stdev_degree=None,
                      fiber_count_check=True,
                      bad_indexes=[],
                      make_plot=True):
    """
    Order tracing by fitting. Adapted from Terese Hansen
    """
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    if midx is None: midx = round(nx/2.)
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
        coef = gaussfit(auxx, auxy, [auxdata[peak1[i]], peak1[i], 2, thresh/2.], maxfev=99999)
        peak[i] = coef[1]
    if npeak != expected_fibers:
        if fiber_count_check:
            print("Npeak {} != Expected peaks {}, making plot to show peaks".format(npeak,expected_fibers))
            plt.plot(auxdata,color="k")
            for i in range(npeak):
                plt.axvline(peak[i], linestyle=':', linewidth=1, color='r')
            plt.axhline(thresh)
            plt.ylim(0, np.nanmax(auxdata)+10)
            plt.show()
            raise ValueError
        else:
            print("WARNING: npeak = {}, expected_fibers = {} [pressing on anyway with bad_indexes]".format(npeak, expected_fibers))
            assert len(bad_indexes) + npeak == expected_fibers, bad_indexes
            _peak = peak.copy()
            peak = np.zeros(expected_fibers)
            igood = 0
            for i in range(expected_fibers):
                if i in bad_indexes: peak[i] = peak[0]
                else:
                    peak[i] = _peak[igood]
                    igood += 1
            npeak = expected_fibers
            
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
        if np.sum(sel) < trace_degree or np.sum(sel) < stdev_degree:
            print("WARNING: peak {} only has {} points".format(i,np.sum(sel)))
        try:
            xarr_fit = np.arange(nx)[sel]
            #auxcoeff = np.polyfit(xarr_fit, ypeak[sel,i], trace_degree)
            _, auxcoeff = jds_poly_reject(xarr_fit, ypeak[sel,i], trace_degree, 5, 5)
            coeff[:,i] = auxcoeff
            #auxcoeff2 = np.polyfit(xarr_fit, ystdv[sel,i], stdev_degree)
            _, auxcoeff2 = jds_poly_reject(xarr_fit, ystdv[sel,i], stdev_degree, 5, 5)
            coeff2[:,i] = auxcoeff2
        except:
            print("Failed on {}".format(i))
            raise

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
    
def _xcor(img1, img2, maxpercentile=99, smoothx=3, smoothy=1):
    """
    Calculate the shift between img1 and img2 (has to be the same size)
    Shift < 0 means img2 is before img1
    Returns shift_0, shift_1
    
    Code adapted from Katy Rodriguez-Wimberly
    """
    assert img1.shape == img2.shape
    
    ## Avoid messing up the actual data
    img1 = img1.copy()
    img2 = img2.copy()
    
    ## Clip the biggest values, arc emission lines can dominate the xcor
    for img in [img1, img2]:
        maxval = np.percentile(img, maxpercentile)
        img[img > maxval] = maxval
    
    ## Use hanning window to downweight the edges
    hanx = np.hanning(img1.shape[0])
    hany = np.hanning(img1.shape[1])
    han = hanx[:,np.newaxis] * hany[np.newaxis,:]
    
    ## Run xcor
    F1 = np.fft.fft2(img1 * han)
    F1 = ndimage.fourier_gaussian(F1, sigma=(smoothx, smoothy))
    F1 = np.conjugate(F1)
    F2 = np.fft.fft2(img2 * han)
    F2 = ndimage.fourier_gaussian(F2, sigma=(smoothx, smoothy))
    CC = np.real(np.fft.ifft2(F1*F2))
    CC = np.fft.fftshift(CC)
    
    ## Find shifts
    zero_index_0 = int(img1.shape[0] / 2)
    zero_index_1 = int(img1.shape[1] / 2)
    max_shift = np.where(CC == np.max(CC))
    shift_0 = max_shift[0] - zero_index_0
    shift_1 = max_shift[1] - zero_index_1
    return shift_0, shift_1

def m2fs_wavecal_xcor_by_object(arcfname, origarcfname, flatfname, fiberconfig):
    """
    Use cross-correlation to find shifts for individual fibers
    Outputs Nfiber x 2, where column 0 is the x-shift and column 1 is the y-shift
    
    The sign is such that you add the result of this to the original arc coordinates to shift them to the new arc.
    """
    ## Load the images
    newarc, err1, header1 = read_fits_two(arcfname)
    oldarc, err2, header2 = read_fits_two(origarcfname)
    assert newarc.shape == oldarc.shape, (newarc.shape, oldarc.shape)
    
    ## Set up
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    allYmin, allYmax = 0, newarc.shape[0]
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    allshifts = np.zeros((Nobj,2)) # X, Y
    
    ## For each object fiber, run a cross correlation and save the shift
    for iobj in range(Nobj):
        ## Get Ymin, Ymax
        Ymin, Ymax = allYmax, allYmin
        for iord in range(Norder):
            itrace = iobj*Norder + iord
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            Ymax = max(Ymax, np.max(Yarr)+1)
            Ymin = min(Ymin, np.min(Yarr))
        ## Increase the buffer by 5%
        dY = 0.05*(Ymax - Ymin)
        Ymin = int(max(allYmin, Ymin - dY))
        Ymax = int(min(allYmax, Ymax + dY))
        print(f"iobj {iobj} Ymin={Ymin} Ymax={Ymax}")
        
        ## Cross correlate image sections
        shiftX, shiftY = _xcor(oldarc[:,Ymin:Ymax], newarc[:,Ymin:Ymax])
        # if shift < 0, this means newarc < oldarc ==> oldarc + shift = newarc
        # This is the sign that we want: add the shift to the old coordinates to get the new coordinates.
        allshifts[iobj,0] = shiftX
        allshifts[iobj,1] = shiftY
    
    return allshifts
    
def m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources,
                                          max_match_dist=2.0,
                                          origarcfname=None, flatfname=None, fiberconfig=None):
    """ To run xcor, include origarcfname, flatfname, fiberconfig as keywords """
    ## Load positions of previously identified sources
    identified_positions = np.vstack([identified_sources["X"],identified_sources["Y"]]).T
    ## Shift previously identified sources using cross-correlation if origarcfname is given
    if (origarcfname is not None) and (flatfname is not None) and (fiberconfig is not None):
        print("Computing cross-correlations",fname,origarcfname)
        allshifts = m2fs_wavecal_xcor_by_object(fname, origarcfname, flatfname, fiberconfig)
        Nobjs, Norders = fiberconfig[0], fiberconfig[1]
        for iobj in range(Nobjs):
            print(f"Obj {iobj}: shiftX={allshifts[iobj,0]:+6.1f} shiftY={allshifts[iobj,1]:+6.1f}")
            ii = identified_sources["iobj"] == iobj
            identified_positions[ii,0] += allshifts[iobj,0]
            identified_positions[ii,1] += allshifts[iobj,1]
        print("Finished xcor shifts")
    
    ## Load current source positions
    name = os.path.basename(fname)[:-5]
    source_catalog_fname = os.path.join(workdir, name+"m_sources.cat")
    ## There is a weird bug here where it fails to read the first time, something about mutable OrderedDict
    ## Trying twice seems to make it work. Maybe something about the registry.
    from astropy.table import Table 
    for i in range(2):
        try:
            sources = Table.read(source_catalog_fname,hdu=2)
        except:
            print("Failed to read {}, trying again".format(source_catalog_fname))
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
    if origarcfname is None:
        dX = biweight_location(Xi - identified_sources[finite]["X"])
        dY = biweight_location(Yi - identified_sources[finite]["Y"])
        print("Mean overall shift: dX={:.3f} dY={:.3f}".format(dX, dY))
        if np.isnan(dX): dX = 0.0; print("biweight_location failed, setting dX=0")
        if np.isnan(dY): dY = 0.0; print("biweight_location failed, setting dY=0")
        identified_positions[:,0] = identified_positions[:,0] + dX
        identified_positions[:,1] = identified_positions[:,1] + dY
    else:
        dX = dY = 0.0
    
    ## Do final match after transformation
    distances, indexes = kdtree.query(identified_positions, distance_upper_bound=max_match_dist)
    finite = np.isfinite(distances)
    num_matched = np.sum(finite)
    if np.sum(finite) != len(np.unique(indexes[finite])):
        print("WARNING: Did not match identified sources uniquely to within {}!".format(max_match_dist))
        print("Got {} distances, {} unique".format(np.sum(finite), len(np.unique(indexes[finite]))))
        print("Just going to continue anyway")
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
    fitdata = np.load(os.path.join(workdir, name+"_wavecal_fitdata.npy"), allow_pickle=True)
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
    Sprime = np.ravel(Sprime)
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

def fit_S_with_profile(P, L, R, eR, Npix, dx=0.1, knots=None, maxiters=5, verbose=True):
    """ R(L,y) = S(L) * P(L,y) """
    RP = R/P
    W = (P/eR)**2.
    W = W/W.sum()
    if knots is None:
        Lmin, Lmax = L.min()+dx, L.max()-dx
        knots = np.linspace(Lmin, Lmax, Npix)[1:-1] # LSQUnivariateSpline adds two end knots
    else:
        knots = knots[1:-1]
    ## Fit B Spline
    iisort = np.argsort(L)
    bad_knots = np.zeros_like(knots, dtype=bool)
    bad_knots[knots > L.max()] = True
    bad_knots[knots < L.min()] = True
    for it in range(maxiters):
        try:
            Sfunc = interpolate.LSQUnivariateSpline(L[iisort], RP[iisort], knots[~bad_knots], W[iisort])
        except Exception as e:
            if verbose:
                print("ERROR: fit_S_with_profile: Failed to fit spline on iter {}/{}!".format(it+1,maxiters))
                print(e)
            if "strictly increasing" in str(e):
                ## Perturb the points by a negligible value
                print("Probably due to identical wavelength values")
                tiny_number = 1e-10
                print("Perturbing duplicate wavelengths by a negligible amount ({:.1e}) to avoid identical x".format(tiny_number))
                # https://stackoverflow.com/questions/30003068/get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
                vals, idx_start, count = np.unique(L[iisort], return_counts=True, return_index=True)
                indices = list(filter(lambda x: x.size > 1, np.split(iisort, idx_start[1:])))
                indices = np.concatenate(indices)
                print("Found {} duplicate indices".format(indices.size))
                
                L[indices] = L[indices] + np.random.normal(scale=tiny_number, size=len(indices))
                iisort = np.argsort(L)
            else:
                print("The most common failure case is losing pixels at the edges of orders")
                print("Trying knot removal")
                for i in range(len(knots)-1):
                    if np.sum(np.logical_and(L >= knots[i], L < knots[i+1])) <= 0:
                        if verbose: print("Bad knot: i={}/{} {:.3f}-{:.3f}".format(i,Npix,knots[i],knots[i+1]))
                        bad_knots[i] = True
                        bad_knots[i+1] = True
                print("{} bad knots".format(bad_knots.sum()))
                if it+1==maxiters:
                    print("n={} k={} used knots={}".format(len(L),3,len(knots)-bad_knots.sum()))
                    print("Lmin,Lmax={:.3f},{:.3f}".format(L.min(), L.max()))
                    print("knotmin,knotmax={:.3f},{:.3f}".format(knots.min(), knots.max()))
                    raise e
        else:
            break
    #else:
    #    raise RuntimeError("Could not fit spline :(")
    return Sfunc
def fit_Sprime(ys, L, R, eR, Npix, ysmax=1.0):
    ii = np.abs(ys) < ysmax
    ys, R, eR, L = ys[ii], R[ii], eR[ii], L[ii]
    P = np.exp(-ys**2/2.)
    return fit_S_with_profile(P, L, R, eR, Npix)
    
def model_scattered_light(data, errs, mask,
                          verbose=True,
                          deg=[5,5], sigma=3.0, maxiter=10):
    """
    Fit a 2D legendre polynomial to data (only using data in the mask).
    Iteratively sigma-clip outlier points.
    """
    scatlight = data.copy()
    scatlighterr = errs.copy()
    shape = data.shape

    ## Fit scattered light with iterative rejection
    def normalize(x):
        """ Linearly scale from -1 to 1 """
        x = np.array(x)
        nx = len(x)
        xmin, xmax = x.min(), x.max()
        xhalf = (x.max()-x.min())/2.
        return (x-xhalf)/xhalf
    XN, YN = np.meshgrid(normalize(np.arange(shape[0])), normalize(np.arange(shape[1])), indexing="ij")
    finite = np.isfinite(scatlight) & mask
    _XN = XN[finite].ravel()
    _YN = YN[finite].ravel()
    _scatlight = scatlight[finite].ravel()
    _scatlighterr = scatlighterr[finite].ravel()
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
        if Noutlier == 0 and iter > 0: break
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
    
    resid = (scatlight-scatlightfit)[finite].ravel()
    scatlightmed = np.median(resid)
    scatlighterr = biweight_scale(resid)
    print("scatlightmed",scatlightmed)
    print("scatlighterr",scatlighterr)

    return scatlightfit, (scatlightmed, scatlighterr, Noutliertotal, iter, scatlight)

def m2fs_subtract_scattered_light(fname, flatfname, arcfname, fiberconfig, Npixcut,
                                  badcolranges=[], deg=[5,5], sigma=3.0, maxiter=10,
                                  manual_tracefn=None,
                                  verbose=True, make_plot=True):
    """
    The basic idea is to mask out the defined extraction regions in the 2D image,
    then fit a 2D legendre polynomial to the rest of the pixels.
    
    You can specify a manual trace function by setting flatfname = None,
    and setting manual_tracefn to a function.
    manual_tracefn(iobj, iord, X) -> Yarr
    
    note: arcfname is not used.
    """
    start = time.time()
    print("Fitting scattered light with Npixcut={} degree={} sigma={:.1f} maxiter={}".format(Npixcut,deg,sigma,maxiter))
    
    outdir = os.path.dirname(fname)
    assert fname.endswith(".fits")
    name = os.path.basename(fname)[:-5]
    outname = name+"s"
    outfname = os.path.join(outdir, outname+".fits")
    
    R, eR, header = read_fits_two(fname)
    shape = R.shape
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    used = np.zeros_like(R, dtype=bool)
    
    Npixcut = int(Npixcut)
    dy = np.arange(-Npixcut, Npixcut+1)
    Npix = R.shape[0]
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    if flatfname is not None:
        tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    else:
        assert manual_tracefn is not None, "Must specify manual_tracefn"
        tracefn = manual_tracefn
    
    # Find all pixels used in extraction
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    for iobj in range(Nobj):
        for iord in range(Norder):
            #ys = ysfunc(iobj, iord, X, Y)
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            used[X_to_get, Y_to_get] = True
    #print("m2fs_subtract_scattered_light: took {:.1f}s to find extracted pixels".format(time.time()-start))
    
    scatlightfit, (scatlightmed, scatlighterr, Noutliertotal, iter, scatlight) = model_scattered_light(R, eR, ~used)
    
    data = R - scatlightfit
    edata = np.sqrt(eR**2 + scatlighterr**2) # + scatlightfit no-var-sub
    #print("edata",edata)
    header.add_history("m2fs_subtract_scattered_light: subtracted scattered light")
    header.add_history("m2fs_subtract_scattered_light: degree={}".format(deg))
    header.add_history("m2fs_subtract_scattered_light: resid median={} error={}".format(scatlightmed, scatlighterr))
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
    np.save(outfname1, [alloutput, used, modeled_flat, [yscut, deg, sigma]])
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
    
def m2fs_add_objnames_to_header(Nobj,header):
    rb = header["SHOE"].lower()
    if rb=="r": slitnums = [1,2,3,4,5,6,7,8]
    if rb=="b": slitnums = [8,7,6,5,4,3,2,1]
    fibernums = 1 + np.arange(16)
    
    namedict = {}
    iobj = 0
    for slitnum in slitnums:
        for fibernum in fibernums:
            if header["FIBER{}{:02}".format(slitnum,fibernum)] != "unplugged":
                namedict["OBJ{:03}".format(iobj)] = header["FIBER{}{:02}".format(slitnum,fibernum)]
                iobj += 1
    assert iobj == Nobj, (iobj, Nobj)
    for iobj in range(Nobj):
        key = "OBJ{:03}".format(iobj)
        header[key] = namedict[key]
    return

def fox_extract(S, eS, F, maxiter=99, kappa=5.0, readnoise=2.7, gain=1.0,
                apply_redchi2=True):
    assert np.all(S.shape == eS.shape)
    assert np.all(S.shape == F.shape)
    Nx = S.shape[0] # number of pixels to extract
    # Pixel Mask and weight
    w = eS**-2
    M = w > .000001 # we'll never reach SNR > 1000, right?
    assert M.sum() > Nx, M.sum()
    
    # Precompute some things
    F2 = F*F
    FS = F*S
    
    # Inital extraction estimate
    rx = np.sum(M * w * FS, axis=1)/np.sum(M * w * F2, axis=1)
    S_hat = F*rx[:,np.newaxis]
    error_squared = gain*S_hat + readnoise**2
    w = M/error_squared
    erx2 = 1/np.sum(w*F2, axis=1)
    
    # Iterate
    for i in range(maxiter):
        # Find cosmic rays and update mask
        new_M = (np.abs(S - S_hat) < kappa*(error_squared - F2*erx2[:,np.newaxis]))
        if new_M.sum() == 0: break
        if np.sum(M & new_M) > Nx: M = M & new_M
        
        # Re-extract
        rx = np.sum(w * FS, axis=1)/np.sum(w * F2, axis=1)
        # Estimate new pixel uncertainties using model
        S_hat = F*rx[:,np.newaxis]
        error_squared = gain*S_hat + readnoise**2
        w = M.astype(float)/error_squared
        erx2 = 1/np.sum(w*F2, axis=1)
    # Rescale errors
    dof = M.sum() - Nx
    reduced_chi = np.sqrt(np.nansum(w * (S - F*rx[:,np.newaxis])**2)/dof)
    erx = np.sqrt(erx2)
    if apply_redchi2:
        print("Rescaling errors by reduced chi = {:.1f} M={} Nx={}".format(reduced_chi, M.sum(), Nx))
        erx = reduced_chi * erx
    return rx, erx, M, reduced_chi

def m2fs_fox_extract(objfname, flatfname, arcfname, fiberconfig, Nextract,
                     maxiter=9, kappa=5.0, readnoise=2.7, gain=1.0,
                     Npix=2048, make_plot=True, throughput_fname=None):
    """
    Following Zechmeister et al. 2014. Extracts s/f, the object spectrum divided by the intrinsic flat spectrum.
    Then multiply by the simple sum-extraction of the flat (in the same pixels).
    """
    start = time.time()
    outdir = os.path.dirname(objfname)
    assert objfname.endswith(".fits")
    name = os.path.basename(objfname)[:-5]
    outfname = os.path.join(outdir,name+"_fox_specs.fits")
    outfname_resid = os.path.join(outdir,name+"_fox_resid.fits")
    
    R, eR, header = read_fits_two(objfname)
    F, eF, header = read_fits_two(flatfname)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    fiber_thru = m2fs_load_fiber_throughput(throughput_fname, fiberconfig)
    
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    
    # Wave, Flux, Err
    outspec = np.zeros((Nobj,Norder,Npix,3))
    Foutspec = np.zeros((Nobj,Norder,Npix,3))
    used = np.zeros(R.shape, dtype=int)
    model = np.zeros_like(R)
    TMPRESID = []
    for iobj in range(Nobj):
        for iord in range(Norder):
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            Larr = Lfunc(iobj, iord, Xarr, Yarr)
            outspec[iobj, iord, Xarr, 0] = Larr
            
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            assert np.all(X_to_get.shape == Y_to_get.shape)
            
            tdata =  R[X_to_get, Y_to_get]
            terrs = eR[X_to_get, Y_to_get]
            tflat =  F[X_to_get, Y_to_get]
            
            rx, erx, M, redchi = fox_extract(tdata, terrs, tflat, maxiter=maxiter, kappa=kappa, readnoise=readnoise, gain=gain)
            
            # Sum-extract flat spectrum
            Fdata_to_sum =  F[X_to_get, Y_to_get]
            Fvars_to_sum = eF[X_to_get, Y_to_get]**2.
            Foutspec[iobj, iord, Xarr, 0] = Larr
            Foutspec[iobj, iord, Xarr, 1] = np.sum(Fdata_to_sum, axis=1)/fiber_thru[iobj]
            # Rescale flat and object
            outspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]*rx
            outspec[iobj, iord, Xarr, 2] = Foutspec[iobj, iord, Xarr, 1]*erx
            
            Fscale = np.nanmedian(Foutspec[iobj, iord, Xarr, 1])
            Foutspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]/Fscale
            Foutspec[iobj, iord, Xarr, 2] = np.sqrt(np.sum(Fvars_to_sum, axis=1))/fiber_thru[iobj]/Fscale
            
            #print(redchi, M.sum())
            
            used[X_to_get, Y_to_get] += M.astype(int)
            
            model[X_to_get, Y_to_get] += Fdata_to_sum*rx[:,np.newaxis]
            TMPRESID.append(((tdata-model[X_to_get, Y_to_get])[M],terrs[M]))
    
    header["NEXTRACT"] = Nextract
    header.add_history("m2fs_fox_extract: FOX extraction with window 2*{}+1".format(Nextract))
    header.add_history("m2fs_fox_extract: flat: {}".format(flatfname))
    header.add_history("m2fs_fox_extract: arc: {}".format(arcfname))
    trueord = fiberconfig[2]
    for iord in range(Norder):
        header["ECORD{}".format(iord)] = trueord[iord]
    m2fs_add_objnames_to_header(Nobj,header)
    
    np.save("{}/{}_tmpresid.npy".format(outdir, name), TMPRESID)
    write_fits_two(outfname, outspec, Foutspec, header)
    
    print("FOX extract of {} took {:.1f}".format(name, time.time()-start))
    if make_plot:
        fig, ax = plt.subplots(figsize=(8,6),subplot_kw={"aspect":1})
        im = ax.imshow(used.T, origin="lower",interpolation='none')
        fig.colorbar(im)
        fig.savefig("{}/{}_fox_usedpix.png".format(outdir,name))

    write_fits_one(outfname_resid, R - model, header)
        
def m2fs_sum_extract(objfname, flatfname, arcfname, fiberconfig, Nextract,
                     Npix=2048, make_plot=True, throughput_fname=None):
    """
    Nextract: total 2xNextract+1 pixels will be added together
    """
    
    start = time.time()
    outdir = os.path.dirname(objfname)
    assert objfname.endswith(".fits")
    name = os.path.basename(objfname)[:-5]
    outfname = os.path.join(outdir,name+"_sum_specs.fits")
    
    R, eR, header = read_fits_two(objfname)
    F, eF, header = read_fits_two(flatfname)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    fiber_thru = m2fs_load_fiber_throughput(throughput_fname, fiberconfig)
    
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    
    # Wave, Flux, Err
    outspec = np.zeros((Nobj,Norder,Npix,3))
    Foutspec = np.zeros((Nobj,Norder,Npix,3))
    used = np.zeros(R.shape, dtype=int)
    for iobj in range(Nobj):
        for iord in range(Norder):
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            Larr = Lfunc(iobj, iord, Xarr, Yarr)
            outspec[iobj, iord, Xarr, 0] = Larr
            
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            assert np.all(X_to_get.shape == Y_to_get.shape)
            data_to_sum =  R[X_to_get, Y_to_get]
            vars_to_sum = eR[X_to_get, Y_to_get]**2.
            outspec[iobj, iord, Xarr, 1] = np.sum(data_to_sum, axis=1)/fiber_thru[iobj]
            outspec[iobj, iord, Xarr, 2] = np.sqrt(np.sum(vars_to_sum, axis=1))/fiber_thru[iobj]
            
            # Extract flat using identical pixels
            Fdata_to_sum =  F[X_to_get, Y_to_get]
            Fvars_to_sum = eF[X_to_get, Y_to_get]**2.
            Foutspec[iobj, iord, Xarr, 0] = Larr
            Foutspec[iobj, iord, Xarr, 1] = np.sum(Fdata_to_sum, axis=1)/fiber_thru[iobj]
            # Rescale flat
            Fscale = np.nanmedian(Foutspec[iobj, iord, Xarr, 1])
            Foutspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]/Fscale
            Foutspec[iobj, iord, Xarr, 2] = np.sqrt(np.sum(Fvars_to_sum, axis=1))/fiber_thru[iobj]/Fscale
            
            used[X_to_get, Y_to_get] += 1
    
    header["NEXTRACT"] = Nextract
    header.add_history("m2fs_sum_extract: sum extraction with window 2*{}+1".format(Nextract))
    header.add_history("m2fs_sum_extract: flat: {}".format(flatfname))
    header.add_history("m2fs_sum_extract: arc: {}".format(arcfname))
    trueord = fiberconfig[2]
    for iord in range(Norder):
        header["ECORD{}".format(iord)] = trueord[iord]
    #lines = fiberconfig[-1]
    #for iobj in range(Nobj):
    #    fibnum = "".join(lines[iobj])
    #    header["OBJ{:03}".format(iobj)] = header["FIBER{}".format(fibnum)]
    m2fs_add_objnames_to_header(Nobj,header)
    
    write_fits_two(outfname, outspec, Foutspec, header)
    
    print("Sum extract of {} took {:.1f}".format(name, time.time()-start))
    if make_plot:
        fig, ax = plt.subplots(figsize=(8,6),subplot_kw={"aspect":1})
        im = ax.imshow(used.T, origin="lower",interpolation='none')
        fig.colorbar(im)
        fig.savefig("{}/{}_sum_usedpix.png".format(outdir,name))
        
def m2fs_horne_flat_extract(objfname, flatfname, arcfname, fiberconfig, Nextract,
                            maxiter=5, sigma=5,
                            Npix=2048, make_plot=True, throughput_fname=None):
    """
    Nextract: total 2xNextract+1 pixels will be added together
    """
    
    start = time.time()
    outdir = os.path.dirname(objfname)
    assert objfname.endswith(".fits")
    name = os.path.basename(objfname)[:-5]
    outfname = os.path.join(outdir,name+"_horneflat_specs.fits")
    outfname_resid = os.path.join(outdir,name+"_horneflat_resid.fits")
    
    R, eR, header = read_fits_two(objfname)
    F, eF, headerF = read_fits_two(flatfname)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    fiber_thru = m2fs_load_fiber_throughput(throughput_fname, fiberconfig)
    
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    
    # Wave, Flux, Err
    outspec = np.zeros((Nobj,Norder,Npix,3))
    Foutspec = np.zeros((Nobj,Norder,Npix,3))
    used = np.zeros(R.shape, dtype=int)
    model = np.zeros_like(R)
    for iobj in range(Nobj):
        for iord in range(Norder):
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            Larr = Lfunc(iobj, iord, Xarr, Yarr)
            # Approximate X as constant wavelength over the full Y profile
            outspec[iobj, iord, Xarr, 0] = Larr
            Foutspec[iobj, iord, Xarr, 0] = Larr
            
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            assert np.all(X_to_get.shape == Y_to_get.shape)
            # Get data
            data_to_sum =  R[X_to_get, Y_to_get]
            errs_to_sum = eR[X_to_get, Y_to_get]
            ivar_to_sum = errs_to_sum**-2.
            ivar_to_sum[~np.isfinite(ivar_to_sum)] = 0.
            Fdata_to_sum =  F[X_to_get, Y_to_get]
            Ferrs_to_sum = eF[X_to_get, Y_to_get]
            Fivar_to_sum = Ferrs_to_sum**-2.
            Fivar_to_sum[~np.isfinite(Fivar_to_sum)] = 0.
            # Get profile
            flat_to_sum =  F[X_to_get, Y_to_get] 
            flat_to_sum[flat_to_sum < 0] = 0.
            flat_to_sum[~np.isfinite(flat_to_sum)] = 0.
            flat_to_sum = flat_to_sum/np.nansum(flat_to_sum, axis=1)[:,np.newaxis]
            
            specest = np.sum(data_to_sum, axis=1)
            mask = np.ones_like(data_to_sum)
            # object profile and mask
            lastNmask = 0
            for iter in range(maxiter):
                ## TODO the problem is that the initial estimate is bad
                mask = np.abs(specest[:,np.newaxis] * flat_to_sum - data_to_sum) < sigma * errs_to_sum
                specest = np.sum(mask * flat_to_sum * data_to_sum * ivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
                Fspecest = np.sum(mask * flat_to_sum * Fdata_to_sum * Fivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)

                ## TODO recalculate pixel variances?
                Nmask = np.sum(mask)
                if lastNmask == Nmask: break
                lastNmask = Nmask
            varest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
            outspec[iobj, iord, Xarr, 1] = specest/fiber_thru[iobj]
            outspec[iobj, iord, Xarr, 2] = np.sqrt(varest)/fiber_thru[iobj]
            model[X_to_get, Y_to_get] += flat_to_sum * specest[:,np.newaxis]
            Fvarest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)
            Foutspec[iobj, iord, Xarr, 1] = Fspecest/fiber_thru[iobj]
            # Rescale flat
            Fscale = np.nanmedian(Foutspec[iobj, iord, Xarr, 1])
            Foutspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]/Fscale
            Foutspec[iobj, iord, Xarr, 2] = np.sqrt(Fvarest)/fiber_thru[iobj]/Fscale
    header["NEXTRACT"] = Nextract
    header.add_history("m2fs_horne_extract: horne extraction with window 2*{}+1".format(Nextract))
    header.add_history("m2fs_horne_extract: flat: {}".format(flatfname))
    header.add_history("m2fs_horne_extract: arc: {}".format(arcfname))
    trueord = fiberconfig[2]
    for iord in range(Norder):
        header["ECORD{}".format(iord)] = trueord[iord]
    #lines = fiberconfig[-1]
    #for iobj in range(Nobj):
    #    fibnum = "".join(lines[iobj])
    #    header["OBJ{:03}".format(iobj)] = header["FIBER{}".format(fibnum)]
    m2fs_add_objnames_to_header(Nobj,header)
    
    write_fits_two(outfname, outspec, Foutspec, header)
    write_fits_one(outfname_resid, R - model, header)

    print("Horne extract of {} took {:.1f}".format(name, time.time()-start))

def m2fs_horne_ghlb_extract(objfname, flatfname, arcfname, fiberconfig, Nextract,
                            maxiter=5, sigma=5,
                            Npix=2048, make_plot=True, throughput_fname=None):
    """
    Does a Horne extraction using the GHLB spatial profile fit from the flat.
    Squashes all X-pixels to a single wavelength.
    Still need to specify a window Nextract within which to use the extraction,
    but you can be more generous because the GHLB fit is better.
    """
    
    start = time.time()
    outdir = os.path.dirname(objfname)
    assert objfname.endswith(".fits")
    name = os.path.basename(objfname)[:-5]
    outfname = os.path.join(outdir,name+"_horneghlb_specs.fits")
    outfname_resid = os.path.join(outdir,name+"_horneghlb_resid.fits")
    
    R, eR, header = read_fits_two(objfname)
    F, eF, header = read_fits_two(flatfname)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    fiber_thru = m2fs_load_fiber_throughput(throughput_fname, fiberconfig)
    
    ghlb_data_path = os.path.join(os.path.dirname(flatfname), os.path.basename(flatfname)[:-5]+"_GHLB.npy")
    ghlb_data = np.load(ghlb_data_path, allow_pickle=True)
    
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    
    # Wave, Flux, Err
    outspec = np.zeros((Nobj,Norder,Npix,3))
    Foutspec = np.zeros((Nobj,Norder,Npix,3))
    used = np.zeros(R.shape, dtype=int)
    model = np.zeros_like(R)
    for iobj in range(Nobj):
        for iord in range(Norder):
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            # The Horne extraction approximation is that wavelength is constant for fixed X
            Larr = Lfunc(iobj, iord, Xarr, Yarr)
            outspec[iobj, iord, Xarr, 0] = Larr
            Foutspec[iobj, iord, Xarr, 0] = Larr
            
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            assert np.all(X_to_get.shape == Y_to_get.shape)
            # Get data
            data_to_sum =  R[X_to_get, Y_to_get]
            errs_to_sum = eR[X_to_get, Y_to_get]
            ivar_to_sum = errs_to_sum**-2.
            ivar_to_sum[~np.isfinite(ivar_to_sum)] = 0.
            Fdata_to_sum =  F[X_to_get, Y_to_get]
            Ferrs_to_sum = eF[X_to_get, Y_to_get]
            Fivar_to_sum = Ferrs_to_sum**-2.
            Fivar_to_sum[~np.isfinite(Fivar_to_sum)] = 0.
            # Get profile and hold fixed
            itrace = iord + iobj*Norder
            pfit, mask, indices, Sfunc, Pfit = ghlb_data[0][itrace]
            L = Lfunc(iobj,iord,X_to_get,Y_to_get)
            ys = ysfunc(iobj,iord,X_to_get,Y_to_get)
            S = np.ones_like(L)
            deg = ghlb_data[3][1]
            Xmat = make_ghlb_feature_matrix(iobj, iord, L, ys, S,
                                            fiberconfig, deg)
            flat_to_sum = Xmat.dot(pfit).reshape(L.shape)
            flat_to_sum = flat_to_sum/np.sum(flat_to_sum, axis=1)[:,np.newaxis]
            
            specest = np.sum(data_to_sum, axis=1)
            Fspecest = np.sum(Fdata_to_sum, axis=1)
            mask = np.ones_like(data_to_sum)
            # object profile and mask
            lastNmask = 0
            for iter in range(maxiter):
                mask = np.abs(specest[:,np.newaxis] * flat_to_sum - data_to_sum) < sigma * errs_to_sum
                specest = np.sum(mask * flat_to_sum * data_to_sum * ivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
                Fspecest = np.sum(mask * flat_to_sum * Fdata_to_sum * Fivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)
                ## TODO recalculate pixel variances?
                Nmask = np.sum(mask)
                if lastNmask == Nmask: break
                lastNmask = Nmask
            varest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
            outspec[iobj, iord, Xarr, 1] = specest/fiber_thru[iobj]
            outspec[iobj, iord, Xarr, 2] = np.sqrt(varest)/fiber_thru[iobj]
            model[X_to_get, Y_to_get] += flat_to_sum * specest[:,np.newaxis]
            Fvarest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)
            Foutspec[iobj, iord, Xarr, 1] = Fspecest/fiber_thru[iobj]
            # Rescale flat
            Fscale = np.nanmedian(Foutspec[iobj, iord, Xarr, 1])
            Foutspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]/Fscale
            Foutspec[iobj, iord, Xarr, 2] = np.sqrt(Fvarest)/fiber_thru[iobj]/Fscale
    
    header["NEXTRACT"] = Nextract
    header.add_history("m2fs_horne_ghlb_extract: horne extraction with window 2*{}+1 and GHLB profile".format(Nextract))
    header.add_history("m2fs_horne_ghlb_extract: flat: {}".format(flatfname))
    header.add_history("m2fs_horne_ghlb_extract: arc: {}".format(arcfname))
    trueord = fiberconfig[2]
    for iord in range(Norder):
        header["ECORD{}".format(iord)] = trueord[iord]
    #lines = fiberconfig[-1]
    #for iobj in range(Nobj):
    #    fibnum = "".join(lines[iobj])
    #    header["OBJ{:03}".format(iobj)] = header["FIBER{}".format(fibnum)]
    m2fs_add_objnames_to_header(Nobj,header)
    
    write_fits_two(outfname, outspec, Foutspec, header)
    write_fits_one(outfname_resid, R - model, header)
    
    print("Horne GHLB extract of {} took {:.1f}".format(name, time.time()-start))

def m2fs_spline_ghlb_extract(objfname, flatfname, arcfname, fiberconfig, Nextract,
                             maxiter=5, sigma=5,
                             Npix=2048, make_plot=True, throughput_fname=None):
    """
    Does a spline fit extraction using the GHLB spatial profile fit from the flat.
    Still need to specify a window Nextract within which to use the extraction.
    """
    
    start = time.time()
    outdir = os.path.dirname(objfname)
    assert objfname.endswith(".fits")
    name = os.path.basename(objfname)[:-5]
    outfname = os.path.join(outdir,name+"_splineghlb_specs.fits")
    outfname_resid = os.path.join(outdir,name+"_splineghlb_resid.fits")
    
    R, eR, header = read_fits_two(objfname)
    F, eF, header = read_fits_two(flatfname)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatfname,arcfname,fiberconfig)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    fiber_thru = m2fs_load_fiber_throughput(throughput_fname, fiberconfig)
    
    ghlb_data_path = os.path.join(os.path.dirname(flatfname), os.path.basename(flatfname)[:-5]+"_GHLB.npy")
    ghlb_data = np.load(ghlb_data_path, allow_pickle=True)
    
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    
    # Wave, Flux, Err
    outspec = np.zeros((Nobj,Norder,Npix,3))
    Foutspec = np.zeros((Nobj,Norder,Npix,3))
    used = np.zeros(R.shape, dtype=int)
    model = np.zeros_like(R)
    for iobj in range(Nobj):
        for iord in range(Norder):
            print("Running iobj={} iord={}".format(iobj,iord))
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            # This is the L at which we will evaluate our final spline fit
            Larr = Lfunc(iobj, iord, Xarr, Yarr)
            outspec[iobj, iord, Xarr, 0] = Larr
            Foutspec[iobj, iord, Xarr, 0] = Larr
            
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            assert np.all(X_to_get.shape == Y_to_get.shape)
            # Get data
            data_to_sum =  R[X_to_get, Y_to_get]
            errs_to_sum = eR[X_to_get, Y_to_get]
            ivar_to_sum = errs_to_sum**-2.
            ivar_to_sum[~np.isfinite(ivar_to_sum)] = 0.
            Fdata_to_sum =  F[X_to_get, Y_to_get]
            Ferrs_to_sum = eF[X_to_get, Y_to_get]
            Fivar_to_sum = Ferrs_to_sum**-2.
            Fivar_to_sum[~np.isfinite(Fivar_to_sum)] = 0.
            # Get profile and hold fixed
            itrace = iord + iobj*Norder
            pfit, mask, indices, Sfunc, Pfit = ghlb_data[0][itrace]
            L = Lfunc(iobj,iord,X_to_get,Y_to_get)
            ys = ysfunc(iobj,iord,X_to_get,Y_to_get)
            S = np.ones_like(L)
            deg = ghlb_data[3][1]
            Xmat = make_ghlb_feature_matrix(iobj, iord, L, ys, S,
                                            fiberconfig, deg)
            flat_to_sum = Xmat.dot(pfit).reshape(L.shape)
            flat_to_sum = flat_to_sum/np.sum(flat_to_sum, axis=1)[:,np.newaxis]
            
            specestfunc = fit_S_with_profile(flat_to_sum.ravel(), L.ravel(), data_to_sum.ravel(), errs_to_sum.ravel(), 0,
                                             knots=Larr)
            specest = specestfunc(Larr)
            """
            Fspecestfunc = fit_S_with_profile(flat_to_sum.ravel(), L.ravel(), Fdata_to_sum.ravel(), Ferrs_to_sum.ravel(), 0,
                                             knots=Larr)
            Fspecest = Fspecestfunc(Larr)
            """
            #specest = np.sum(data_to_sum, axis=1)
            mask = np.ones_like(data_to_sum)
            # object profile and mask
            lastNmask = 0
            for iter in range(maxiter):
                mask = np.abs(specest[:,np.newaxis] * flat_to_sum - data_to_sum) < sigma * errs_to_sum
                #specest = np.sum(mask * flat_to_sum * data_to_sum * ivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
                specestfunc = fit_S_with_profile(flat_to_sum[mask].ravel(), L[mask].ravel(), data_to_sum[mask].ravel(), 
                                                 errs_to_sum[mask].ravel(), 0, knots=Larr)
                specest = specestfunc(Larr)
                """
                Fspecestfunc = fit_S_with_profile(flat_to_sum[mask].ravel(), L[mask].ravel(), Fdata_to_sum[mask].ravel(),
                                                  Ferrs_to_sum[mask].ravel(), 0, knots=Larr)
                Fspecest = Fspecestfunc(Larr)
                """
                Fspecest = np.sum(mask * flat_to_sum * Fdata_to_sum * Fivar_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)
                ## TODO recalculate pixel variances?
                Nmask = np.sum(mask)
                if lastNmask == Nmask: break
                print("iter {} masked {}/{} points".format(iter,Nmask,mask.size))
                lastNmask = Nmask
            varest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * ivar_to_sum, axis=1)
            outspec[iobj, iord, Xarr, 1] = specest/fiber_thru[iobj]
            outspec[iobj, iord, Xarr, 2] = np.sqrt(varest)/fiber_thru[iobj]
            model[X_to_get, Y_to_get] += flat_to_sum * specest[:,np.newaxis]
            Fvarest = np.sum(mask * flat_to_sum, axis=1)/np.sum(mask * flat_to_sum**2. * Fivar_to_sum, axis=1)
            Foutspec[iobj, iord, Xarr, 1] = Fspecest/fiber_thru[iobj]
            # Rescale flat
            Fscale = np.nanmedian(Foutspec[iobj, iord, Xarr, 1])
            Foutspec[iobj, iord, Xarr, 1] = Foutspec[iobj, iord, Xarr, 1]/Fscale
            Foutspec[iobj, iord, Xarr, 2] = np.sqrt(Fvarest)/fiber_thru[iobj]/Fscale

    header["NEXTRACT"] = Nextract
    header.add_history("m2fs_spline_ghlb_extract: spline extraction with window 2*{}+1 and GHLB profile".format(Nextract))
    header.add_history("m2fs_spline_ghlb_extract: flat: {}".format(flatfname))
    header.add_history("m2fs_spline_ghlb_extract: arc: {}".format(arcfname))
    trueord = fiberconfig[2]
    for iord in range(Norder):
        header["ECORD{}".format(iord)] = trueord[iord]
    #lines = fiberconfig[-1]
    #for iobj in range(Nobj):
    #    fibnum = "".join(lines[iobj])
    #    header["OBJ{:03}".format(iobj)] = header["FIBER{}".format(fibnum)]
    m2fs_add_objnames_to_header(Nobj,header)
    
    write_fits_two(outfname, outspec, Foutspec, header)
    write_fits_one(outfname_resid, R - model, header)
    
    print("Spline GHLB extract of {} took {:.1f}".format(name, time.time()-start))

def quick_1d_extract(objfname, flatfname, fiberconfig, outfname=None, Nextract=4, 
                     make_plot=False, **kwargs):
    """
    Trace the flat and aperture sum extract flux vs pixel, saving as multispec.
    Use e.g. for getting an initial arc identification, or sanity/improvement checks.
    Also used for throughput extractions.
    """
    if outfname is None:
        outfname = os.path.join(os.path.dirname(objfname),os.path.basename(objfname)[:-5]+".1d.ms.fits")
    ### Trace flat
    R, eR, hR = read_fits_two(objfname)
    F, eF, hF = read_fits_two(flatfname)
    assert R.shape == F.shape
    if not os.path.exists(m2fs_get_trace_fnames(flatfname)[0]): 
        m2fs_trace_orders(flatfname, fiberconfig, make_plot=make_plot, **kwargs)
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)    
    
    Npix = R.shape[0]
    dy = np.arange(-Nextract, Nextract+1)
    offsets = np.tile(dy, Npix).reshape((Npix,len(dy)))
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    
    ### Extract objects with sum extraction
    onedarcs = np.zeros((Nobj*Norder, Npix))
    for iobj in range(Nobj):
        for iord in range(Norder):
            itrace = iobj*Norder + iord
            Xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1) 
            Yarr = tracefn(iobj, iord, Xarr)
            X_to_get = np.vstack([Xarr for _ in dy]).T
            Y_to_get = (offsets[Xarr,:] + Yarr[:,np.newaxis]).astype(int)
            data_to_sum =  R[X_to_get, Y_to_get]
            onedarcs[itrace,Xarr] = np.sum(data_to_sum, axis=1)
    make_multispec(outfname, [onedarcs.T], ["sum extract spectrum"])

def m2fs_process_throughput_frames(thrunames, thruscatnames, outfname,
                                   flatfname,
                                   fiberconfig,
                                   detection_scale_factors=[1,2,3,4,5],
                                   Nextract=4,
                                   redo_scattered_light=False, 
                                   scattrace_ythresh=200, scattrace_nthresh=1.9, scat_Npixcut=13, scat_sigma=3.0, scat_deg=[5,5],
                                   make_plot=True):
    """
    Given a set of frames to use for throughput (e.g. twilight frames):
    * subtract scattered light
    * extract spectra
    * calculate throughput

    flatfname is used for the trace
    
    Output an array of Nobj, Norder which is the throughput in that order medianed across all the input throughput frames.
    """
    assert len(thrunames) == len(thruscatnames), (thrunames, thruscatnames)
    Nthru = len(thrunames)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    
    outdir = os.path.dirname(outfname)
    outname = os.path.basename(outfname)[:-4]
    
    allthrumat = np.zeros((Nthru, Nobj, Norder))
    start = time.time()
    for iframe, (thruname, thruscatname) in enumerate(zip(thrunames, thruscatnames)):
        workdir = os.path.dirname(thruscatname)
        ## Trace and scattered light subtraction
        if not os.path.exists(thruscatname) or redo_scattered_light:
            start2 = time.time()
            print("m2fs_process_throughput_frames: trace and scattered light {} -> {}".format(thruname, thruscatname))
            #m2fs_trace_orders(thruname, fiberconfig, make_plot=make_plot, ythresh=scattrace_ythresh, nthresh=scattrace_nthresh)
            #m2fs_subtract_scattered_light(thruname, thruname, None, fiberconfig,
            #                              make_plot=make_plot, Npixcut=scat_Npixcut, sigma=scat_sigma, deg=scat_deg)
            m2fs_subtract_scattered_light(thruname, flatfname, None, fiberconfig,
                                          make_plot=make_plot, Npixcut=scat_Npixcut, sigma=scat_sigma, deg=scat_deg)
            print("m2fs_process_throughput_frames: Took {:.1f}s".format(time.time()-start2))
        elif os.path.exists(thruscatname):
            print("m2fs_process_throughput_frames: {} already exists".format(thruscatname))
        
        ## Extract
        start2 = time.time()
        thru1dfname = os.path.join(workdir, os.path.basename(thruname)[:-5]+".1d.ms.fits")
        quick_1d_extract(thruscatname, flatfname, fiberconfig, Nextract=Nextract,
                         outfname=thru1dfname)
        print("m2fs_process_throughput_frames extraction: Took {:.1f}s -> {}".format(time.time()-start2, thru1dfname))
        
        ## Compute throughput
        thrumat = m2fs_find_throughput_oneframe(thru1dfname, fiberconfig)
        allthrumat[iframe] = thrumat
    overall_norm = np.median(allthrumat, axis=1)
    norm_allthrumat = allthrumat / overall_norm[:,np.newaxis,:]
    final_thrumat = np.median(norm_allthrumat, axis=0)
    np.save(outfname, [final_thrumat, norm_allthrumat])
    print("m2fs_process_throughput_frames: processing {} frames took {:.1f}s".format(
            Nthru, time.time()-start))
    for iobj in range(Nobj):
        for iord in range(Norder):
            #final_thrumat[iobj,iord] = np.median(norm_allthrumat[:,iobj,iord])
            print("obj={:2} ord={} thru = {:.2f} +/- {:.3f}".format(
                    iobj,iord,final_thrumat[iobj,iord],
                    np.std(norm_allthrumat[:,iobj,iord])))
    if make_plot:
        trueord = fiberconfig[2]
        objnumarr = np.arange(Nobj)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.axhline(1.0, color='k', linestyle=':')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for iord in range(Norder):
            for i in range(Nthru):
                ax.plot(objnumarr, norm_allthrumat[i,:,iord], color=colors[iord % len(colors)], lw=.5)
            ax.plot(objnumarr, final_thrumat[:, iord], color=colors[iord % len(colors)], alpha=.3, lw=5,
                    label=str(trueord[iord]))
        fiber_thru = m2fs_load_fiber_throughput(outfname, fiberconfig)
        ax.plot(objnumarr, fiber_thru, 'ko-', color='k', alpha=.3, lw=9)
        ax.set_xlabel("iobj")
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig("{}/{}_thru_all.png".format(outdir, outname))
        plt.close(fig)

def m2fs_find_throughput_oneframe(thru1dfname, fiberconfig,
                                  verbose=True):
    """
    Given a single 1D multispec file, calculate the throughput.
    """
    start2 = time.time()
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    with fits.open(thru1dfname) as hdul:
        data = hdul[0].data
    thrumat = np.zeros((Nobj,Norder)) + np.nan
    for iobj in range(Nobj):
        for iord in range(Norder):
            itrace = iord + iobj*Norder
            xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1)
            thrumat[iobj,iord] = np.median(data[itrace,xarr])
    if verbose:
        print("--m2fs_find_throughput_oneframe: {} Took {:.1f}".format(thru1dfname, time.time()-start2))
    return thrumat

def m2fs_load_fiber_throughput(thrufname, fiberconfig):
    """ Calculate the median of the inner orders to determine fiber throughput """
    if thrufname is None: return np.ones(fiberconfig[0])
    # Nobj * Norder
    thrumat, _ = np.load(thrufname, allow_pickle=True)
    throughput_orders = pick_throughput_orders(fiberconfig)
    thrumat = thrumat[:,throughput_orders]
    fiber_thru = np.median(thrumat, axis=1)
    assert fiber_thru.shape == (fiberconfig[0],)
    return fiber_thru
def pick_throughput_orders(fiberconfig):
    """ For now just reject the 0th and Nord-1th order numbers """
    Nord = fiberconfig[1]
    if Nord == 1: return np.array([0])
    if Nord == 2: raise NotImplementedError("Need to decide what to do for 2 orders")
    return np.arange(Nord)[1:-1]

def m2fs_pixel_flat(flatfname, fiberconfig, Npixcut):
    """
    Use the flat to find pixel variations.
    DON'T USE THIS. It doesn't work.
    """
    R, eR, header = read_fits_two(flatfname)
    #shape = R.shape
    #X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    
    tracefn = m2fs_load_trace_function(flatfname, fiberconfig)

    ## This is the output array
    pixelflat = np.ones_like(R)
    
    Npixuse = Npixcut-1

    Nobj, Nord = fiberconfig[0], fiberconfig[1]
    for iobj in range(Nobj):
        for iord in range(Nord):
            ## Set up empty array to process and put back
            Xmin, Xmax = fiberconfig[4][iord]
            Xarr = np.arange(Xmin,Xmax+1)
            Npix = len(Xarr)
            medarr = np.zeros((Npix,2*Npixcut+1))
            ## Follow the trace and get relevant pixels in every location
            y0 = tracefn(iobj, iord, Xarr)
            ix_y = np.round(y0).astype(int)
            ix_ymin = ix_y-Npixcut
            ix_ymax = ix_y+Npixcut
            for j in range(2*Npixcut+1):
                dj = j - Npixcut
                medarr[:,j] = R[Xarr,ix_y+dj]
            ## Rectify the pixels in just the y direction
            #yloc = y0[:,np.newaxis] + np.arange(-Npixcut,Npixcut+1)[np.newaxis,:]
            yoff = y0-ix_y
            
            def mapping1(x):
                return x[0], yoff[x[0]]+x[1]
            def mapping2(x):
                return x[0], -yoff[x[0]]+x[1]
            rect_medarr = ndimage.geometric_transform(medarr, mapping1, mode='nearest')
            # The typical value as a function of X
            medX = np.median(rect_medarr,axis=1)
            #medY = np.median(rect_medarr,axis=0)
            #medR = medX[:,np.newaxis] * medY[np.newaxis,:]
            #medR = medR * np.median(rect_medarr)/np.median(medR)
            medR = ndimage.median_filter(rect_medarr, (9,1), mode='nearest')
            np.save("medarr.npy",medarr)
            np.save("rect_medarr.npy",rect_medarr)
            np.save("rect_smooth.npy",medR)
            rect_ratio = rect_medarr/medR
            np.save("rect_ratio.npy",rect_ratio)
            # Shift back
            medR = ndimage.geometric_transform(medR, mapping2, mode='nearest')
            np.save("smooth.npy",medR)
            medR = medarr/medR
            #medR = medR/np.median(medR)
            for j in range(2*Npixuse+1):
                dj = j - Npixuse
                pixelflat[Xarr,ix_y+dj] = medR[:,j]
    return pixelflat
