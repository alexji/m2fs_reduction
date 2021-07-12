from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time
from astropy.io import ascii

from m2fs_utils import read_fits_two, write_fits_two, m2fs_parse_fiberconfig
from m2fs_utils import m2fs_4amp
from m2fs_utils import m2fs_make_master_dark, m2fs_subtract_one_dark
from m2fs_utils import m2fs_make_master_flat, m2fs_trace_orders
from m2fs_utils import m2fs_new_trace_orders, m2fs_new_trace_orders_multidetect
from m2fs_utils import m2fs_wavecal_find_sources_one_arc
from m2fs_utils import m2fs_wavecal_identify_sources_one_arc
from m2fs_utils import m2fs_wavecal_fit_solution_one_arc
from m2fs_utils import m2fs_get_pixel_functions, m2fs_load_trace_function
from m2fs_utils import m2fs_subtract_scattered_light
from m2fs_utils import m2fs_ghlb_extract, m2fs_sum_extract, m2fs_horne_flat_extract
from m2fs_utils import m2fs_fox_extract
from m2fs_utils import m2fs_horne_ghlb_extract, m2fs_spline_ghlb_extract
from m2fs_utils import m2fs_process_throughput_frames #quick_1d_extract

#################################################
# Tools to see if you have already done a step
#################################################
def file_finished(workdir, task):
    return os.path.join(workdir,"task_"+task)
def mark_finished(workdir, task):
    open(file_finished(workdir, task),'a').close()
def check_finished(workdir, task):
    finished = os.path.exists(file_finished(workdir, task))
    if finished: print("Already finished {} in {}".format(task, workdir))
    return finished
def get_file(dbrowfile, workdir, suffix=""):
    return os.path.join(workdir, os.path.basename(dbrowfile)+suffix+".fits")
def load_db(dbname, calibconfig=None):
    tab = ascii.read(dbname)
    nums = np.array([int(x[-4:]) for x in tab["FILE"]])
    tab.add_column(tab.Column(nums, "NUM"))
    assert len(np.unique(tab["INST"])) == 1, np.unique(tab["INST"])
    assert len(np.unique(tab["CONFIG"])) == 1, np.unique(tab["CONFIG"])
    assert len(np.unique(tab["FILTER"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["SLIT"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["BIN"])) == 1, np.unique(tab["BIN"])
    assert len(np.unique(tab["SPEED"])) == 1, np.unique(tab["SPEED"])
    assert len(np.unique(tab["NAMP"])) == 1, np.unique(tab["NAMP"])
    assert np.all([x in ["Dark","Comp","Flat","Object","Thru"] for x in tab["EXPTYPE"]])
    if calibconfig is None:
        return tab
    else:
        allnums = np.unique(calibconfig.to_pandas().values)
        indices = np.array([num in allnums for num in tab["NUM"]])
        return tab[indices]
def get_obj_nums(calibconfig):
    return np.array(calibconfig["sciencenum"])
def get_flat_num(objnum, calibconfig):
    ix = np.where(objnum==calibconfig["sciencenum"])[0][0]
    return calibconfig[ix]["flatnum"]
def get_arc_num(objnum, calibconfig):
    ix = np.where(objnum==calibconfig["sciencenum"])[0][0]
    return calibconfig[ix]["arcnum"]
def get_obj_file(objnum, dbname, workdir, calibconfig, suffix="d"):
    tab = load_db(dbname, calibconfig)
    ix = np.where(tab["NUM"]==objnum)[0][0]
    #assert tab[ix]["EXPTYPE"]=="Object"
    return get_file(tab[ix]["FILE"], workdir, suffix)
def get_flat_file(objnum, dbname, workdir, calibconfig, suffix="d"):
    flatnum = get_flat_num(objnum, calibconfig)
    tab = load_db(dbname, calibconfig)
    ix = np.where(tab["NUM"]==flatnum)[0][0]
    assert tab[ix]["EXPTYPE"]=="Flat"
    return get_file(tab[ix]["FILE"], workdir, suffix)
def get_arc_file(objnum, dbname, workdir, calibconfig, suffix="d"):
    arcnum = get_arc_num(objnum, calibconfig)
    tab = load_db(dbname, calibconfig)
    ix = np.where(tab["NUM"]==arcnum)[0][0]
    assert tab[ix]["EXPTYPE"]=="Comp"
    return get_file(tab[ix]["FILE"], workdir, suffix)

#################################################
# Pipeline steps
#################################################
def m2fs_biastrim(dbname, workdir):
    if check_finished(workdir, "biastrim"): return
    
    tab = load_db(dbname)
    for row in tab:
        outfile = os.path.join(workdir, os.path.basename(row["FILE"])+".fits")
        m2fs_4amp(row["FILE"], outfile)
    mark_finished(workdir, "biastrim")
def m2fs_darksub(dbname, workdir):
    if check_finished(workdir, "darksub"): return
    
    ## Make master dark frame
    masterdarkname = os.path.join(workdir, "master_dark.fits")
    tab = load_db(dbname)
    darktab = tab[tab["EXPTYPE"]=="Dark"]
    print("Found {} dark frames".format(len(darktab)))
    fnames = [get_file(x, workdir) for x in darktab["FILE"]]
    m2fs_make_master_dark(fnames, masterdarkname)
    
    dark, darkerr, darkheader = read_fits_two(masterdarkname)
    
    ## Subtract master dark frame from all the data
    for row in tab:
        if row["EXPTYPE"] != "Dark":
            fname = get_file(row["FILE"], workdir)
            outfname = get_file(row["FILE"], workdir, "d")
            print("Processing {} -> {}".format(fname, outfname))
            m2fs_subtract_one_dark(fname, outfname, dark, darkerr, darkheader)
    
    mark_finished(workdir, "darksub")
def m2fs_traceflat_fast(dbname, workdir, fiberconfig, calibconfig, suffix="d"):
    if check_finished(workdir, "traceflatfast-{}".format(suffix)): return
    print("Running traceflatfast with suffix {}".format(suffix))
    
    ## Make a master flat
    masterflatname = os.path.join(workdir, "master_flat_{}.fits".format(suffix))
    tab = load_db(dbname)
    if suffix=="d":
        flattab = tab[tab["EXPTYPE"]=="Flat"]
        fnames = [get_file(x, workdir, suffix) for x in flattab["FILE"]]
    else:
        objnums = get_obj_nums(calibconfig)
        fnames = [get_flat_file(objnum, dbname, workdir, calibconfig, suffix) for objnum in objnums]
    print("Found {} flatframes".format(len(fnames)))
    print("Running master flat (not really used right now)")
    m2fs_make_master_flat(fnames, masterflatname)
    # Trace master flat
    m2fs_new_trace_orders_multidetect(masterflatname, fiberconfig, make_plot=True)
    
    ## Trace individual flats
    objnums = get_obj_nums(calibconfig)
    fnames = [get_flat_file(objnum, dbname, workdir, calibconfig, suffix) for objnum in objnums]
    for fname in np.unique(fnames):
        m2fs_new_trace_orders_multidetect(fname, fiberconfig, make_plot=True)
    mark_finished(workdir, "traceflatfast-{}".format(suffix))
def m2fs_traceflat_slow(dbname, workdir, fiberconfig, calibconfig, suffix="ds", midx=None):
    if check_finished(workdir, "traceflatslow-{}".format(suffix)): return
    print("Running traceflatslow with suffix {}".format(suffix))
    
    ## Make a master flat
    masterflatname = os.path.join(workdir, "master_flat_{}.fits".format(suffix))
    tab = load_db(dbname)
    if suffix=="d":
        flattab = tab[tab["EXPTYPE"]=="Flat"]
        fnames = [get_file(x, workdir, suffix) for x in flattab["FILE"]]
    else:
        objnums = get_obj_nums(calibconfig)
        fnames = [get_flat_file(objnum, dbname, workdir, calibconfig, suffix) for objnum in objnums]
    print("Found {} flatframes".format(len(fnames)))
    print("Running master flat (not really used right now)")
    m2fs_make_master_flat(fnames, masterflatname)
    # Trace master flat
    m2fs_trace_orders(masterflatname, fiberconfig, trace_degree=7, stdev_degree=3, make_plot=True, midx=midx)
    
    ## Trace individual flats
    objnums = get_obj_nums(calibconfig)
    fnames = [get_flat_file(objnum, dbname, workdir, calibconfig, suffix) for objnum in objnums]
    for fname in np.unique(fnames):
        m2fs_trace_orders(fname, fiberconfig, make_plot=True, midx=midx)
    mark_finished(workdir, "traceflatslow-{}".format(suffix))

def m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname, masterflatname):
    if check_finished(workdir, "throughput"): return
    tab = load_db(dbname)
    tab = tab[tab["EXPTYPE"]=="Thru"]
    Nthru = len(tab)
    thrufnames = [get_file(row["FILE"], workdir, "d") for row in tab]
    thrufnames_scat = [get_file(row["FILE"], workdir, "ds") for row in tab]
    
    # Calculate throughput corrections
    m2fs_process_throughput_frames(thrufnames, thrufnames_scat, throughput_fname,
                                   masterflatname, fiberconfig,
                                   detection_scale_factors=[1,2,3])
    mark_finished(workdir, "throughput")

def m2fs_wavecal_find_sources(dbname, workdir, calibconfig):
    if check_finished(workdir, "wavecal-findsources"): return
    ## Get list of arcs to process
    objnums = get_obj_nums(calibconfig)
    fnames = [get_arc_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    
    start = time.time()
    for fname in fnames:
        ## Find sources
        m2fs_wavecal_find_sources_one_arc(fname, workdir)
    print("Finding all sources took {:.1f}s".format(time.time()-start))
    mark_finished(workdir, "wavecal-findsources")
    
def m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig, calibconfig, max_match_dist=2.0,
                                  origarcfname=None):
    if check_finished(workdir, "wavecal-identifysources"): return
    
    ## TODO use fiberconfig to get this somehow!!!
    identified_sources = ascii.read(fiberconfig[5])
    
    ## Get list of arcs to process
    #tab = load_db(dbname)
    #filter = np.unique(tab["FILTER"])[0]
    #config = np.unique(tab["CONFIG"])[0]
    #arctab = tab[tab["EXPTYPE"]=="Comp"]
    #print("Found {} arc frames".format(len(arctab)))
    #fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    ## Get list of arcs to process
    objnums = get_obj_nums(calibconfig)
    fnames = [get_arc_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    
    start = time.time()
    for fname, flatfname in zip(fnames, flatfnames):
        if origarcfname is None:
            m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources,
                                                  max_match_dist=max_match_dist)
        else:
            m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources,
                                                  max_match_dist=max_match_dist,
                                                  origarcfname=origarcfname,
                                                  flatfname=flatfname,
                                                  fiberconfig=fiberconfig)
    print("Identifying all sources took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "wavecal-identifysources")

def m2fs_wavecal_fit_solution(dbname, workdir, fiberconfig, calibconfig):
    if check_finished(workdir, "wavecal-fitsolution"): return
    
    ## Get list of arcs to process
    #tab = load_db(dbname)
    #filter = np.unique(tab["FILTER"])[0]
    #config = np.unique(tab["CONFIG"])[0]
    #arctab = tab[tab["EXPTYPE"]=="Comp"]
    #print("Found {} arc frames".format(len(arctab)))
    #fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    ## Get list of arcs to process
    objnums = get_obj_nums(calibconfig)
    fnames = [get_arc_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    
    start = time.time()
    already_done = []
    for fname,flatfname in zip(fnames, flatfnames):
        if fname in already_done: continue
        m2fs_wavecal_fit_solution_one_arc(fname, workdir, fiberconfig, make_plot=True, flatfname=flatfname)
        already_done.append(fname)
    print("Fitting wavelength solutions took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "wavecal-fitsolution")

def m2fs_scattered_light(dbname, workdir, fiberconfig, calibconfig, Npixcut=13, sigma=3.0, deg=[5,5]):
    if check_finished(workdir, "scatlight"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    
    start = time.time()
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_subtract_scattered_light(objfname, flatfname, arcfname, fiberconfig, Npixcut,
                                      sigma=sigma, deg=deg)
        m2fs_subtract_scattered_light(flatfname, flatfname, arcfname, fiberconfig, Npixcut,
                                      sigma=sigma, deg=deg)
    print("Scattered light subtraction took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "scatlight")

def m2fs_fit_profile_ghlb(dbname, workdir, fiberconfig, calibconfig):
    if check_finished(workdir, "flat-ghlb"): return
    
    ## Get list of arcs to process
    ## TODO think about this a bit more!!! Which files go with what?
    objnums = get_obj_nums(calibconfig)
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    
    start = time.time()
    for flatfname, arcfname in zip(flatfnames, arcfnames):
        m2fs_ghlb_extract(flatfname, flatfname, arcfname, fiberconfig,
                          yscut=2.0, deg=[3,12], sigma=4.0,
                          make_plot=True, make_obj_plots=True)
    print("Fitting GHLB to flats took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "flat-ghlb")

def m2fs_fit_flat_profiles(dbname, workdir, fiberconfig, calibconfig):
    if check_finished(workdir, "extract-fitflat"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    done = []
    for flatfname, arcfname in zip(flatfnames, arcfnames):
        if flatfname in done: continue
        m2fs_ghlb_extract(flatfname, flatfname, arcfname, fiberconfig, yscut=2.5, deg=[0,10], sigma=5.0,
                          make_plot=True, make_obj_plots=True)
        done.append(flatfname)
    print("Fitting GHLB to flats took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-fitflat")

def m2fs_extract_sum_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract,
                              throughput_fname=None):
    """
    Simple sum extraction within an aperture of 2*Nextract+1 pixels around the trace
    """
    if check_finished(workdir, "extract-sum"): return
    
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_sum_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract, make_plot=True,
                         throughput_fname=throughput_fname)
    
    mark_finished(workdir, "extract-sum")

def m2fs_extract_fox_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract,
                              throughput_fname=None):
    """
    Flat-relative optimal extraction (FOX) within an aperture of 2*Nextract+1 pixels around the trace
    """
    if check_finished(workdir, "extract-fox"): return
    
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_fox_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract, make_plot=True,
                         throughput_fname=throughput_fname)
    
    mark_finished(workdir, "extract-fox")

def m2fs_extract_horne_flat(dbname, workdir, fiberconfig, calibconfig, Nextract,
                            throughput_fname=None):
    """
    Horne extraction using flat as object profile for 2*Nextract+1 pixels around the trace
    """
    if check_finished(workdir, "extract-horneflat"): return
    
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_horne_flat_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract, make_plot=True,
                                throughput_fname=throughput_fname)
    
    mark_finished(workdir, "extract-horneflat")

def m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract,
                            throughput_fname=None):
    if check_finished(workdir, "extract-horneghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_horne_ghlb_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract,
                                throughput_fname=throughput_fname)
    print("Horne GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-horneghlb")

def m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract,
                             throughput_fname=None, sigma=5):
    if check_finished(workdir, "extract-splineghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_spline_ghlb_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract,
                                 throughput_fname=throughput_fname, sigma=sigma)
    print("Spline GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-splineghlb")

def m2fs_lsf_trace(dbname, workdir, fiberconfig, calibconfig):
    ## Run trace+profile on flats
    if check_finished(workdir, "lsf-trace"): return
    objnums = get_obj_nums(calibconfig)
    fnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    all_good = True
    for fname in np.unique(fnames):
        for detection_scale_factor in [3,4,5,6,7]:
            try:
                m2fs_new_trace_orders(fname, fiberconfig, make_plot=True,
                                      detection_scale_factor=detection_scale_factor)
            except Exception as e:
                print("new trace orders Failed {}".format(detection_scale_factor))
                print(e)
            else:
                break
        else:
            print("ERROR: {} could not be traced! Run this manually (change midx)".format(fname))
            all_good=False
    if all_good:
        mark_finished(workdir, "lsf-trace")

#################################################
# script to run
#################################################
if __name__=="__main__":
    start = time.time()
    """
    if True:
        dbname = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/redM2FS.db"
        workdir = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/reduced/red"
        calibconfigname = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/hyi1run.txt"
        fiberconfigname = "data/5targ_r.txt"
        throughput_fname = os.path.join(workdir,"5targ_r_throughput.npy")
    else:
        dbname = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/blueM2FS.db"
        workdir = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/reduced/blue"
        calibconfigname = "/Volumes/My Passport for Mac/observing_data/M2FS_DATA/Hyi1/hyi1run.txt"
        fiberconfigname = "data/5targ_b.txt"
        throughput_fname = os.path.join(workdir,"5targ_b_throughput.npy")
    """
    dbname = "/Users/alexji/M2FS_DATA/Hyi1/redM2FS.db"
    workdir = "/Users/alexji/M2FS_DATA/Hyi1/reduced"
    calibconfigname = "/Users/alexji/M2FS_DATA/Hyi1/hyi1run.txt"
    fiberconfigname = "data/5targ_r.txt"
    throughput_fname = os.path.join(workdir,"5targ_r_throughput.npy")
    midx=750 #from MKRW
    origarcfname="/Users/alexji/M2FS_DATA/Hyi1/reduced/r2573d.fits"
    #origarcfname=None # this turns off xcor
    max_match_dist=2.0
    
    assert os.path.exists(dbname)
    assert os.path.exists(workdir)
    assert os.path.exists(calibconfigname)
    assert os.path.exists(fiberconfigname)
    
    tab = load_db(dbname)
    ## I am assuming everything is part of the same setting
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    flattab= tab[tab["EXPTYPE"]=="Flat"]
    objtab = tab[tab["EXPTYPE"]=="Object"]
    arcnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    flatnames= [get_file(x, workdir, "d") for x in flattab["FILE"]]
    objnames = [get_file(x, workdir, "d") for x in objtab["FILE"]]
    
    ### Prep data
    m2fs_biastrim(dbname, workdir)
    m2fs_darksub(dbname, workdir)

    calibconfig = ascii.read(calibconfigname)
    fiberconfig = m2fs_parse_fiberconfig(fiberconfigname)
    objnums = get_obj_nums(calibconfig)
    
    ### Throughput correction with twilight flats
    #m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname)
    #tab = load_db(dbname)
    #tab = tab[tab["EXPTYPE"]=="Thru"]
    #Nthru = len(tab)
    #thrufnames = [get_file(row["FILE"], workdir, "d") for row in tab]
    #thrufnames_scat = [get_file(row["FILE"], workdir, "ds") for row in tab]
    ## Calculate throughput corrections
    #m2fs_process_throughput_frames(thrufnames, thrufnames_scat, throughput_fname, fiberconfig)
    
    ###### Tracing, scattered light
    ## Quick flat trace and object profile estimate
    m2fs_traceflat_slow(dbname, workdir, fiberconfig, calibconfig, suffix="d", midx=midx)
    ## Subtract scattered light
    m2fs_scattered_light(dbname, workdir, fiberconfig, calibconfig)
    ## Slower/more accurate flat trace and object profile AFTER scattered light subtraction
    m2fs_traceflat_slow(dbname, workdir, fiberconfig, calibconfig, suffix="ds", midx=midx)
    ## Throughput correction with twilight flats and master flat trace (includes scattered light)
    #m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname,
    #                get_file("master_flat",workdir,"_ds"))
    
    ###### Wavelength Calibration
    ## Find sources in 2D arc spectrum (currently a separate step running sextractor)
    m2fs_wavecal_find_sources(dbname, workdir, calibconfig)

    # NOTE: IF AN ARC HAS NOT BEEN IDENTIFIED, IT NEEDS TO BE DONE MANUALLY NOW
    ## Identify features in 2D spectrum. If specify origarcfname, does cross-correlation in as lice.
    m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig, calibconfig,
                                  origarcfname=origarcfname, max_match_dist=max_match_dist)
    ## Use features to fit Xccd,Yccd(obj, order, lambda)
    m2fs_wavecal_fit_solution(dbname, workdir, fiberconfig, calibconfig)
    
    ##### Extract flats and objects
    ### Simple sum extraction
    m2fs_extract_sum_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract=4)#, throughput_fname=throughput_fname)
    ### Flat-relative optimal extraction
    m2fs_extract_fox_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract=3)#, throughput_fname=throughput_fname)
    ### Horne extraction with flat as profile (this is bad actually)
    #m2fs_extract_horne_flat(dbname, workdir, fiberconfig, calibconfig, Nextract=4, throughput_fname=throughput_fname)
    
    ### GHLB fit flats as profiles
    #m2fs_fit_flat_profiles(dbname, workdir, fiberconfig, calibconfig)
    ### Horne extraction with GHLB fit as profile
    #m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5, throughput_fname=throughput_fname)
    ### Spline extraction with GHLB fit as profile
    #m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5, throughput_fname=throughput_fname, sigma=10)

    print("Total time for {} objects: {:.1f}s".format(len(objnums), time.time()-start))

