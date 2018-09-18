from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from scipy import optimize, ndimage, spatial, linalg
import re
import glob, os, sys, time, subprocess

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
        Nobj = int(fp.readline().strip())
        Nord = int(fp.readline().strip())
        ordlist = list(map(int, fp.readline().strip().split()))
        lines = fp.readlines()
    lines = list(map(lambda x: x.strip().split(), lines))
    lines = Table(rows=lines, names=["tetris","fiber"])
    return Nobj, Nord, ordlist, lines

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
                      nthresh=2.0, ystart=0, dx=20, dy=5, nstep=10, degree=4, ythresh=500,
                      make_plot=True):
    """
    Order tracing by fitting. Adapted from Terese Hansen
    """
    data, edata, header = read_fits_two(fname)
    nx, ny = data.shape
    midx = round(nx/2.)
    thresh = nthresh*np.median(data)
    
    Nobjs, Norders = fiberconfig[0], fiberconfig[1]
    expected_fibers = Nobjs * Norders
    
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
               (data[j,int(ypeak[j-nstep,i])] <= ythresh):
                break
            if np.max(auxy) >= auxthresh:
                coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j-nstep,i])], ypeak[j-nstep,i], 2, thresh/2.],
                                xtol=1e-6,maxfev=10000)
                ypeak[j,i] = coef[1]
                ystdv[j,i] = min(coef[2],dy/2.)
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
               (data[j,int(ypeak[j+nstep,i])] <= ythresh):
                break
            if np.max(auxy) >= auxthresh:
                coef = gaussfit(auxx, auxy, [auxdata0[int(ypeak[j+nstep,i])], ypeak[j+nstep,i], 2, thresh/2.],
                                xtol=1e-6,maxfev=10000)
                ypeak[j,i] = coef[1]
                ystdv[j,i] = min(coef[2], dy/2.)
            else:
                ypeak[j,i] = ypeak[j+nstep,i] # so i don't get lost
                ystdv[j,i] = ystdv[j+nstep,i]
                nopeak[j,i] = 1
    ypeak[(nopeak == 1) | (ypeak == 0)] = np.nan
    ystdv[(nopeak == 1) | (ypeak == 0)] = np.nan
    print("\nTracing took {:.1f}s".format(time.time()-start))
    
    coeff = np.zeros((degree+1,npeak))
    coeff2 = np.zeros((degree+1,npeak))
    for i in range(npeak):
        sel = np.isfinite(ypeak[:,i]) #np.where(ypeak[:,i] != -666)[0]
        xarr_fit = np.arange(nx)[sel]
        auxcoeff = np.polyfit(xarr_fit, ypeak[sel,i], degree)
        coeff[:,i] = auxcoeff
        auxcoeff2 = np.polyfit(xarr_fit, ystdv[sel,i], degree)
        coeff2[:,i] = auxcoeff2

    fname1, fname2 = m2fs_get_trace_fnames(fname)
    print(fname,fname1,fname2)
    np.savetxt(fname1, coeff.T)
    np.savetxt(fname2, coeff2.T)
    
    if make_plot:
        import matplotlib.pyplot as plt
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
                ax.errorbar(xarr[::nstep], yarr[::nstep], yerr=eyarr[::nstep], fmt='r.', ms=1., elinewidth=.5)
        fig.savefig("{}/{}_trace.png".format(
                os.path.dirname(fname), os.path.basename(fname)[:-5]),
                    dpi=300,bbox_inches="tight")
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
    

def make_wavecal_feature_matrix(ycen,iobj,iord,wave,
                                Nobj,trueord,wavemin,wavemax,
                                ycenmin,ycenmax,deg):
    """
    Convert identified features into a feature matrix.
    
    Parameterization:
    X = fX(ycen, nord, nord*wave) + X0(iobj)
    Y = fY(ycen, nord, nord*wave) + Y0(iobj)
    fX and fY are 3D legendre polynomials, X0 and Y0 are offsets for each object.
    
    First 4 variables are for each identified feature in the arc:
    ycen = y-coordinate of object trace at center of image (X=1024)
           needs to be the same order for every object
    iobj = object number of the feature, from 0 to Nobj-1
           (note ycen and iobj are paired 1-1, but we need both values here)
    iord = index of feature order number (into trueord to get actual nord)
           from 0 to len(trueord)-1
    wave = wavelength of feature
    
    Next variables are used for normalization.
    Nobj: total number of objects
    trueord: map from iord to nord
    wavemin, wavemax: used to standardize wave from -1 to 1
    ycenmin, ycenmax: used to standardize ycen from -1 to 1
    
    degree is a 3-length tuple/list of integers, passed to np.polynomial.legendre.legvander3d

    Note: IRAF ecidentify describes the basic functional form of order number.
    Let yord = offset +/- iord (to get the true order number)
    Then fit lambda = f(x, yord)/yord
    This means that wave -> wave*yord is a better variable to fit when determining x
    """
    try: N1 = len(ycen)
    except TypeError: N1 = 1
    try: N2 = len(iord)
    except TypeError: N2 = 1
    try: N3 = len(wave)
    except TypeError: N3 = 1
    N = np.max([N1,N2,N3])
    
    if N1 == 1: ycen = np.full(N, ycen)
    if N2 == 1: iord = np.full(N, iord)
    if N3 == 1: wave = np.full(N, wave)
    
    def center(x,xmin=None,xmax=None):
        """ Make x go from -1 to 1 """
        x = np.array(x)
        if xmin is None: xmin = np.nanmin(x)
        else: assert np.all(np.logical_or(x >= xmin, np.isnan(x)))
        if xmax is None: xmax = np.nanmax(x)
        else: assert np.all(np.logical_or(x <= xmax, np.isnan(x)))
        xmean = (xmax+xmin)/2.
        xdiff = (xmax-xmin)/2.
        return (x - xmean)/xdiff
    
    # Normalize coordinates to -1, 1
    ycennorm = center(ycen,ycenmin,ycenmax)
    
    # Normalize order number to -1, 1
    cord = np.array(trueord)[iord]
    cordmin, cordmax = np.min(trueord), np.max(trueord)
    cordnorm = center(cord,cordmin,cordmax)
    
    # Convert wave -> wave * cord
    waveord = cord * wave
    waveordmin = wavemin * np.min(trueord)
    waveordmax = wavemax * np.max(trueord)
    waveordnorm = center(waveord, waveordmin, waveordmax)
    
    # Legendre(ycen, nord, wave*nord, degrees)
    legpolymat = np.polynomial.legendre.legvander3d(ycennorm, cordnorm, waveordnorm, deg)
    # Add a constant offset for each object
    # Use indicator variables, dropping the last object to remove degeneracy
    indicatorpolymat = np.zeros((len(legpolymat), Nobj-1))
    for it in range(Nobj-1):
        indicatorpolymat[iobj==it,it] = 1.
    return np.concatenate([legpolymat, indicatorpolymat], axis=1)

def m2fs_wavecal_fit_solution_one_arc(fname, workdir, fiberconfig,
                                      make_plot=True):
    ## parameters of the fit. Should do this eventually with a config file, kwargs, or something.
    maxiter = 5
    min_lines_per_order = 10
    sigma = 3.
    ycenmin = 0.
    ycenmax = 2055.
    deg = [3,3,5]
    
    ## Load identified wavelength features
    name = os.path.basename(fname)[:-5]
    data = ascii.read(os.path.join(workdir, name+"_wavecal_id.txt"))
    outfname = os.path.join(workdir, name+"_wavecal_fitdata.npy")
    print("Fitting wavelength solution for {} ({} features)".format(fname, len(data)))
    
    ## TODO GET ALL THESE FROM FIBERCONFIG
    wavemin, wavemax = 5100., 5450.
    Nobj, Norder, trueord = fiberconfig[0], fiberconfig[1], fiberconfig[2]
    
    ## Construct feature matrix to fit
    Xmat = make_wavecal_feature_matrix(data["Ycen"],data["iobj"],data["iorder"],data["L"], 
                                       Nobj, trueord, wavemin, wavemax,
                                       ycenmin, ycenmax, deg)
    good = np.ones(len(Xmat), dtype=bool)
    ymat = np.vstack([data["X"],data["Y"]]).T
    ## Solve for fit with all features
    pfit, residues, rank, svals = linalg.lstsq(Xmat, ymat)
    
    ## Iterative sigma clipping and refitting
    for iter in range(maxiter):
        print("Iteration {}: {} features to start".format(iter+1, np.sum(good)))
        ## Find outliers in each order
        ## I decided to do it this way for easier interpretability, rather than
        ## a global sigma-clip rejection
        yfitall = Xmat.dot(pfit)
        dX, dY = (ymat - yfitall).T
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
                tdX, tdY = dX[iiobjorderused], dY[iiobjorderused]
                dXcen, dXstd = biweight_location(tdX), biweight_scale(tdX)
                dYcen, dYstd = biweight_location(tdY), biweight_scale(tdY)
                
                to_clip_X = np.logical_and(iiobjorderused, np.abs((dX-dXcen)/dXstd) > sigma)
                to_clip_Y = np.logical_and(iiobjorderused, np.abs((dY-dYcen)/dYstd) > sigma)
                if np.sum(iiobjorderused) - np.sum(to_clip_X | to_clip_Y) < min_lines_per_order:
                    print("    Could not clip {}/{} features from obj {} order {}".format(
                            np.sum(to_clip_X | to_clip_Y), np.sum(iiobjorderused), iobj, iorder))
                    continue
                else:
                    to_clip = np.logical_or(to_clip, np.logical_or(to_clip_X, to_clip_Y))
                
        good = np.logical_and(good, np.logical_not(to_clip))
        print("  Cut down to {} features".format(np.sum(good)))
        
        ## Rerun fit on valid features
        ## this Xmat, this Ymat
        tXmat = Xmat[good]
        tymat = ymat[good]
        pfit, residues, rank, svals = linalg.lstsq(tXmat, tymat)
    
    ## Information to save:
    np.save(outfname, [pfit, good, Xmat, data["iobj"], data["iorder"], trueord, Nobj, maxiter, deg, sigma])
    
    if make_plot:
        import matplotlib.pyplot as plt
        yfitall = Xmat.dot(pfit)
        fig, axes = plt.subplots(Nobj,Norder, figsize=(Norder*4,Nobj*3))
        figfname = os.path.join(workdir, name+"_wavecal_fitdata.png")
        for iobj in range(Nobj):
            for iorder in range(Norder):
                ax = axes[iobj, iorder]
                iiobjorder = np.logical_and(data["iobj"]==iobj,
                                            data["iorder"]==iorder)
                iiobjorderused = np.logical_and(iiobjorder, good)
                Lall = data["L"][iiobjorder]
                Xall = data["X"][iiobjorder]-yfitall[iiobjorder,0]
                Yall = data["Y"][iiobjorder]-yfitall[iiobjorder,1]
                Nall = len(Lall)
                Luse = data["L"][iiobjorderused]
                Xuse = data["X"][iiobjorderused]-yfitall[iiobjorderused,0]
                Yuse = data["Y"][iiobjorderused]-yfitall[iiobjorderused,1]
                Nuse = len(Luse)
                ax.axhline(0,color='k',linestyle=':')
                ax.set_ylim(-1,1)
                l, = ax.plot(Lall, Xall, '.')
                ax.plot(Luse, Xuse, 'o', color=l.get_color())
                l, = ax.plot(Lall, Yall, '.')
                ax.plot(Luse, Yuse, 'o', color=l.get_color())
                Xrms = np.std(Xuse)
                Yrms = np.std(Yuse)
                ax.set_title("N={}, used={} Xstd={:.2f} Ystd={:.2f}".format(
                        Nall,Nuse,Xrms,Yrms))
        fig.savefig(figfname, bbox_inches="tight")
        plt.close(fig)
