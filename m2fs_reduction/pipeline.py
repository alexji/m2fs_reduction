from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time
from astropy.io import ascii

from m2fs_utils import read_fits_two, write_fits_two, m2fs_parse_fiberconfig
from m2fs_utils import m2fs_4amp
from m2fs_utils import m2fs_make_master_dark, m2fs_subtract_one_dark
from m2fs_utils import m2fs_make_master_flat, m2fs_trace_orders
from m2fs_utils import m2fs_wavecal_find_sources_one_arc
from m2fs_utils import m2fs_wavecal_identify_sources_one_arc
from m2fs_utils import m2fs_wavecal_fit_solution_one_arc
from m2fs_utils import m2fs_get_pixel_functions, m2fs_load_trace_function
from m2fs_utils import m2fs_subtract_scattered_light
from m2fs_utils import m2fs_ghlb_extract, m2fs_sum_extract, m2fs_horne_flat_extract
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
        allnums = np.unique(calibconfig.to_pandas().as_matrix())
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
def m2fs_traceflat(dbname, workdir, fiberconfig, calibconfig):
    if check_finished(workdir, "traceflat"): return
    
    ## Make a master flat
    masterflatname = os.path.join(workdir, "master_flat.fits")
    tab = load_db(dbname)
    flattab = tab[tab["EXPTYPE"]=="Flat"]
    print("Found {} flatframes".format(len(flattab)))
    fnames = [get_file(x, workdir, "d") for x in flattab["FILE"]]
    print("Running master flat (not really used right now)")
    m2fs_make_master_flat(fnames, masterflatname)
    m2fs_trace_orders(masterflatname, fiberconfig, trace_degree=7, stdev_degree=3, make_plot=True)
    
    ## Run trace on individual flats
    objnums = get_obj_nums(calibconfig)
    fnames = [get_flat_file(objnum, dbname, workdir, calibconfig) for objnum in objnums]
    for fname in np.unique(fnames):
        m2fs_trace_orders(fname, fiberconfig, make_plot=True)
        for detection_scale_factor in [7.0, 6.0, 5.0, 4.0, 3.0]:
            try:
                m2fs_new_trace_orders(fname, fiberconfig, make_plot=True,
                                      detection_scale_factor=detection_scale_factor)
            except:
                pass
            else:
                break
    mark_finished(workdir, "traceflat")
def m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname):
    if check_finished(workdir, "throughput"): return
    tab = load_db(dbname)
    tab = tab[tab["EXPTYPE"]=="Thru"]
    Nthru = len(tab)
    thrufnames = [get_file(row["FILE"], workdir, "d") for row in tab]
    thrufnames_scat = [get_file(row["FILE"], workdir, "ds") for row in tab]
    # Calculate throughput corrections
    m2fs_process_throughput_frames(thrufnames, thrufnames_scat, throughput_fname, fiberconfig)
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
    
def m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig, calibconfig, max_match_dist=2.0):
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
    
    start = time.time()
    for fname in fnames:
        m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources, max_match_dist=max_match_dist)
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
    
    ##tab = load_db(dbname)
    ##flattab = tab[tab["EXPTYPE"]=="Flat"]
    ##print("Found {} flats".format(len(flattab)))
    ##arctab = tab[tab["EXPTYPE"]=="Comp"]
    #### HACK TODO need to do something about picking this, but for now....
    ##arctab = arctab[arctab["EXPTIME"] < 30]
    ##print("Found {} arcs".format(len(flattab)))
    ##
    ###masterflatname = os.path.join(workdir, "master_flat.fits")
    ##flatfnames = [get_file(x, workdir, "d") for x in flattab["FILE"]]
    ##arcfnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
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

def m2fs_fit_flat_profiles(dbname, workdir, fiberconfig, calibconfig):
    if check_finished(workdir, "extract-fitflat"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    flatfnames2 = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    done = []
    for flatfname, flatfname2, arcfname in zip(flatfnames, flatfnames2, arcfnames):
        if flatfname in done: continue
        try:
            m2fs_ghlb_extract(flatfname2, flatfname, arcfname, fiberconfig, yscut=2.5, deg=[0,10], sigma=5.0,
                              make_plot=True, make_obj_plots=True)
        except:
            import pdb; pdb.set_trace()
        done.append(flatfname)
    print("Fitting GHLB to flats took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-fitflat")

def m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract,
                            throughput_fname=None):
    if check_finished(workdir, "extract-horneghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    flatfnames2 = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, flatfname2, arcfname in zip(objfnames, flatfnames, flatfnames2, arcfnames):
        m2fs_horne_ghlb_extract(objfname, flatfname, flatfname2, arcfname, fiberconfig, Nextract=Nextract,
                                throughput_fname=throughput_fname)
    print("Horne GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-horneghlb")

def m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract,
                             throughput_fname=None):
    if check_finished(workdir, "extract-splineghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    flatfnames2 = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, flatfname2, arcfname in zip(objfnames, flatfnames, flatfnames2, arcfnames):
        m2fs_spline_ghlb_extract(objfname, flatfname, flatfname2, arcfname, fiberconfig, Nextract=Nextract,
                                 throughput_fname=throughput_fname)
    print("Spline GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-splineghlb")

#################################################
# script to run
#################################################
if __name__=="__main__":
    start = time.time()
    if False:
        dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSr.db"
        workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/r"
        calibconfigname = "nov2017run.txt"
        fiberconfigname = "data/Mg_wide_r.txt"
        throughput_fname = os.path.join(workdir,"Mg_wide_r_throughput.npy")
    else:
        #dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSb.db"
        #workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/b"
        #calibconfigname = "nov2017run.txt"
        #fiberconfigname = "data/Bulge_GC1_b.txt"
        #throughput_fname = os.path.join(workdir,"Bulge_GC1_b_throughput.npy")
        dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSb.db"
        workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/b_arcs"
        calibconfigname = "nov2017arcs.txt"
        fiberconfigname = "data/Bulge_GC1_b.txt"
        throughput_fname = os.path.join(workdir,"Bulge_GC1_b_throughput.npy")
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
    
    calibconfig = ascii.read(calibconfigname)
    fiberconfig = m2fs_parse_fiberconfig(fiberconfigname)
    objnums = get_obj_nums(calibconfig)
    
    ### Prep data
    m2fs_biastrim(dbname, workdir)
    m2fs_darksub(dbname, workdir)

    ### Throughput correction with twilight flats
    m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname)
    ### Trace flat
    m2fs_traceflat(dbname, workdir, fiberconfig, calibconfig)
    
    ### M2FS wavecal
    ## Find sources in 2D arc spectrum (currently a separate step running sextractor)
    m2fs_wavecal_find_sources(dbname, workdir, calibconfig)
    # NOTE: IF AN ARC HAS NOT BEEN IDENTIFIED, IT NEEDS TO BE DONE MANUALLY NOW
    ## Identify features in 2D spectrum with coherent point drift
    m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig, calibconfig)
    ## Use features to fit Xccd,Yccd(obj, order, lambda)
    m2fs_wavecal_fit_solution(dbname, workdir, fiberconfig, calibconfig)
    
    ### Scattered light subtraction
    m2fs_scattered_light(dbname, workdir, fiberconfig, calibconfig)
    
    ### Simple sum extraction
    m2fs_extract_sum_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract=4, throughput_fname=throughput_fname)
    ### Horne extraction with flat as profile
    m2fs_extract_horne_flat(dbname, workdir, fiberconfig, calibconfig, Nextract=4, throughput_fname=throughput_fname)
    
    ### GHLB fit flats as profiles
    m2fs_fit_flat_profiles(dbname, workdir, fiberconfig, calibconfig)
    ### Horne extraction with GHLB fit as profile
    m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5, throughput_fname=throughput_fname)
    ### Spline extraction with GHLB fit as profile
    m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5, throughput_fname=throughput_fname)
    
    print("Total time for {} objects: {:.1f}s".format(len(objnums), time.time()-start))

def m2fs_frame_by_frame_ghlb_extract(dbname, workdir, fiberconfig, calibconfig):
    """
    ### Frame-by-frame extraction
    ## Fit GHLB for each object individually
    ## This does NOT really work: the sky fibers are so faint that they have a really bad GHLB profile
    ## They get pulled around everywhere by cosmics, Littrow ghost, etc
    ## (The objects themselves seem fine!)
    ## Also, note that extended outliers (cosmic rays along a trace, bad lines/columns) are not removed
    ## Hopefully this can be remedied by fitting multiple exposures
    """
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    
    start = time.time()
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_ghlb_extract(objfname, flatfname, arcfname, fiberconfig, yscut=2.0, deg=[2,10], sigma=5.0,
                          make_plot=True, make_obj_plots=True)
    print("Fitting GHLB to objects took {:.1f}".format(time.time()-start))
