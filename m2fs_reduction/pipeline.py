from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import glob, os, sys, time
from astropy.io import ascii

from m2fs_utils import m2fs_4amp, read_fits_two
from m2fs_utils import m2fs_make_master_dark, m2fs_subtract_one_dark
from m2fs_utils import m2fs_make_master_flat, m2fs_trace_orders

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
def m2fs_traceflat(dbname, workdir):
    if check_finished(workdir, "traceflat"): return
    
    ## Make a master flat
    masterflatname = os.path.join(workdir, "master_flat.fits")
    tab = ascii.read(dbname)
    flattab = tab[tab["EXPTYPE"]=="Flat"]
    print("Found {} flatframes".format(len(flattab)))
    fnames = [get_file(x, workdir) for x in flattab["FILE"]]
    m2fs_make_master_flat(fnames, masterflatname)
    
    #mark_finished(workdir, "traceflat")

#################################################
# Script to run
#################################################
if __name__=="__main__":
    dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSr.db"
    workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/r"
    assert os.path.exists(dbname)
    assert os.path.exists(workdir)
    
    ## I am assuming everything is part of the same setting
    tab = ascii.read(dbname)
    assert len(np.unique(tab["INST"])) == 1, np.unique(tab["INST"])
    assert len(np.unique(tab["CONFIG"])) == 1, np.unique(tab["CONFIG"])
    assert len(np.unique(tab["FILTER"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["SLIT"])) == 1, np.unique(tab["FILTER"])
    assert len(np.unique(tab["BIN"])) == 1, np.unique(tab["BIN"])
    assert len(np.unique(tab["SPEED"])) == 1, np.unique(tab["SPEED"])
    assert len(np.unique(tab["NAMP"])) == 1, np.unique(tab["NAMP"])
    
    m2fs_biastrim(dbname, workdir)
    m2fs_darksub(dbname, workdir)
    m2fs_traceflat(dbname, workdir)
    # Associate arcs and flats to data
    # M2FS wavecal
    # M2FS profile
    # M2FS extract
    # M2FS skysub
