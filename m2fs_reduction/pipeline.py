from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time
from astropy.io import ascii

from m2fs_utils import read_fits_two, write_fits_two, m2fs_parse_fiberconfig
from m2fs_utils import m2fs_4amp
from m2fs_utils import m2fs_make_master_dark, m2fs_subtract_one_dark
from m2fs_utils import m2fs_make_master_flat, m2fs_trace_orders
from m2fs_utils import m2fs_wavecal_find_sources

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

#################################################
# Pipeline steps
#################################################
def m2fs_biastrim(dbname, workdir):
    if check_finished(workdir, "biastrim"): return
    
    tab = ascii.read(dbname)
    for row in tab:
        outfile = os.path.join(workdir, os.path.basename(row["FILE"])+".fits")
        m2fs_4amp(row["FILE"], outfile)
    mark_finished(workdir, "biastrim")
def m2fs_darksub(dbname, workdir):
    if check_finished(workdir, "darksub"): return
    
    ## Make master dark frame
    masterdarkname = os.path.join(workdir, "master_dark.fits")
    tab = ascii.read(dbname)
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
def m2fs_traceflat(dbname, workdir, fiberconfig):
    if check_finished(workdir, "traceflat"): return
    
    ## Make a master flat
    masterflatname = os.path.join(workdir, "master_flat.fits")
    tab = ascii.read(dbname)
    flattab = tab[tab["EXPTYPE"]=="Flat"]
    print("Found {} flatframes".format(len(flattab)))
    fnames = [get_file(x, workdir, "d") for x in flattab["FILE"]]
    m2fs_make_master_flat(fnames, masterflatname)
    
    m2fs_trace_orders(masterflatname, fiberconfig, make_plot=True)
    
    mark_finished(workdir, "traceflat")

def m2fs_wavecal(dbname, workdir, fiberconfig):
    if check_finished(workdir, "arcid"): return
    
    ## Get list of arcs to process
    tab = ascii.read(dbname)
    filter = np.unique(tab["FILTER"])[0]
    config = np.unique(tab["CONFIG"])[0]
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    print("Found {} arc frames".format(len(arctab)))
    fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    ## Read pre-identified coordinates
    print("Reading identifications from (xx todo)")
    identified_sources = None # TODO use dbname and fiberconfig to get this somehow
    #print("{} features identified".format(len(identified_sources)))
    
    for fname in fnames:
        ## Find sources
        m2fs_wavecal_find_sources(fname, workdir)
        
        ## Use CPD to identify points
        #m2fs_wavecal_identify_sources(fname, workdir, identified_sources)
        
        ## Fit wavelength solution and new trace from identified points
        #m2fs_wavecal_fit_solution(fname, workdir, fiberconfig)
        #break

#################################################
# Script to run
#################################################
if __name__=="__main__":
    dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSr.db"
    workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/r"
    fiberconfigname = "data/Mg_wide_r.txt"
    assert os.path.exists(dbname)
    assert os.path.exists(workdir)
    assert os.path.exists(fiberconfigname)
    
    ## I am assuming everything is part of the same setting
    tab = ascii.read(dbname)
    assert len(np.unique(tab["INST"])) == 1, np.unique(tab["INST"])
    assert len(np.unique(tab["CONFIG"])) == 1, np.unique(tab["CONFIG"])
    assert len(np.unique(tab["FILTER"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["SLIT"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["BIN"])) == 1, np.unique(tab["BIN"])
    assert len(np.unique(tab["SPEED"])) == 1, np.unique(tab["SPEED"])
    assert len(np.unique(tab["NAMP"])) == 1, np.unique(tab["NAMP"])
    
    fiberconfig = m2fs_parse_fiberconfig(fiberconfigname)
    
    m2fs_biastrim(dbname, workdir)
    m2fs_darksub(dbname, workdir)
    m2fs_traceflat(dbname, workdir, fiberconfig)
    
    
    # M2FS wavecal
    # Find sources in 2D arc spectrum (currently a separate step running sextractor)
    # NOTE: IF AN ARC HAS NOT BEEN IDENTIFIED, IT NEEDS TO BE DONE MANUALLY NOW
    # Identify features in 2D spectrum with coherent point drift
    # Use features to fit Xccd,Yccd(obj, order, lambda)
    m2fs_wavecal(dbname, workdir, fiberconfig)
    
    # M2FS profile
    # Fit g(obj, order, lambda)
    
    # M2FS extract
    # Associate arcs and flats to data
    # Forward Model Flux(obj,order,lambda)
    # Sigma clip outlier pixels (cosmic rays) when fitting
    
    # M2FS skysub
    
