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
from m2fs_utils import quick_1d_extract

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
    assert tab[ix]["EXPTYPE"]=="Object"
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
    mark_finished(workdir, "traceflat")
def m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname):
    pass

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

def m2fs_extract_sum_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract):
    """
    Simple sum extraction within an aperture of 2*Nextract+1 pixels around the trace
    """
    if check_finished(workdir, "extract-sum"): return
    
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_sum_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract, make_plot=True)
    
    mark_finished(workdir, "extract-sum")

def m2fs_extract_horne_flat(dbname, workdir, fiberconfig, calibconfig, Nextract):
    """
    Horne extraction using flat as object profile for 2*Nextract+1 pixels around the trace
    """
    if check_finished(workdir, "extract-horneflat"): return
    
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    for objfname, flatfname, arcfname in zip(objfnames, flatfnames, arcfnames):
        m2fs_horne_flat_extract(objfname, flatfname, arcfname, fiberconfig, Nextract=Nextract, make_plot=True)
    
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
        m2fs_ghlb_extract(flatfname2, flatfname, arcfname, fiberconfig, yscut=2.5, deg=[0,10], sigma=5.0,
                          make_plot=True, make_obj_plots=True)
        done.append(flatfname)
    print("Fitting GHLB to flats took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-horneflat")

def m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract):
    if check_finished(workdir, "extract-horneghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    flatfnames2 = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, flatfname2, arcfname in zip(objfnames, flatfnames, flatfnames2, arcfnames):
        m2fs_horne_ghlb_extract(objfname, flatfname, flatfname2, arcfname, fiberconfig, Nextract=Nextract)
    print("Horne GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-horneghlb")

def m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract):
    if check_finished(workdir, "extract-splineghlb"): return
    objnums = get_obj_nums(calibconfig)
    objfnames = [get_obj_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    flatfnames = [get_flat_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    flatfnames2 = [get_flat_file(objnum, dbname, workdir, calibconfig, "ds") for objnum in objnums]
    arcfnames = [get_arc_file(objnum, dbname, workdir, calibconfig, "d") for objnum in objnums]
    start = time.time()
    for objfname, flatfname, flatfname2, arcfname in zip(objfnames, flatfnames, flatfnames2, arcfnames):
        m2fs_spline_ghlb_extract(objfname, flatfname, flatfname2, arcfname, fiberconfig, Nextract=Nextract)
    print("Spline GHLB extract took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "extract-splineghlb")

#################################################
# script to run
#################################################
if __name__=="__main__":
    start = time.time()
    #dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSr.db"
    #workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/r"
    #calibconfigname = "nov2017run.txt"
    #fiberconfigname = "data/Mg_wide_r.txt"
    #throughput_fname = "./Mg_wide_r_throughput.npy"
    dbname = "/Users/alexji/M2FS_DATA/test_rawM2FSb.db"
    workdir = "/Users/alexji/M2FS_DATA/test_reduction_files/b"
    calibconfigname = "nov2017run.txt"
    fiberconfigname = "data/Bulge_GC1_b.txt"
    throughput_fname = "./Bulge_GC1_b_throughput.npy"
    assert os.path.exists(dbname)
    assert os.path.exists(workdir)
    assert os.path.exists(calibconfigname)
    assert os.path.exists(fiberconfigname)
    
    tab = load_db(dbname)
    ## I am assuming everything is part of the same setting
    #tab = ascii.read(dbname)
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
    #m2fs_throughput(dbname, workdir, fiberconfig, throughput_fname)
    Npixcut=13; sigma=3.0; deg=[5,5]
    Nextract=4
    ythresh=200; nthresh=1.9
    from astropy.io import ascii, fits
    from m2fs_utils import m2fs_get_trace_fnames
    
    tab = load_db(dbname)
    tab = tab[tab["EXPTYPE"]=="Thru"]
    Nthru = len(tab)
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    ordpixranges = fiberconfig[4]
    trueord = fiberconfig[2]
    
    allthrumat = np.zeros((Nthru, Nobj, Norder))
    start = time.time()
    for irow,row in enumerate(tab):
        start2 = time.time()
        thrufname = get_file(row["FILE"], workdir, "d")
        thrufname_scat = get_file(row["FILE"], workdir, "ds")
        print("--THRU: PROCESSING {}".format(thrufname))
        # Trace, subtract scattered light
        if not os.path.exists(m2fs_get_trace_fnames(thrufname)[0]): 
            m2fs_trace_orders(thrufname, fiberconfig, make_plot=True, ythresh=ythresh, nthresh=nthresh)
            m2fs_subtract_scattered_light(thrufname, thrufname, None, fiberconfig,
                                          make_plot=True, Npixcut=Npixcut, sigma=sigma, deg=deg)
        # 1D extract
        thru1dfname = os.path.join(workdir, os.path.basename(thrufname)[:-5]+".1d.ms.fits")
        # Maybe do this with profile extraction? Will only matter if there are significant
        #  fiber-to-fiber differences in the profile.
        quick_1d_extract(thrufname_scat, thrufname, fiberconfig, Nextract=Nextract,
                         outfname=thru1dfname)
        #quick_1d_extract(thrufname, thrufname, fiberconfig, Nextract=Nextract,
        #                 outfname=thru1dfname)
        # Load spectrum and calculate throughput medians
        with fits.open(thru1dfname) as hdul:
            data = hdul[0].data
        thrumat = np.zeros((Nobj,Norder)) + np.nan
        for iobj in range(Nobj):
            for iord in range(Norder):
                itrace = iord + iobj*Norder
                xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1)
                thrumat[iobj,iord] = np.median(data[itrace,xarr])
        allthrumat[irow] = thrumat
        print("--THRU: Took {:.1f}".format(time.time()-start2))
    print("--ALLTHRU: Took {:.1f}".format(time.time()-start))
    np.save(throughput_fname, allthrumat)
    ## One overall normalization for all fibers/orders
    #overall_norm = np.median(allthrumat.reshape((Nthru, Nobj*Norder)), axis=1)
    #norm_allthrumat = allthrumat / overall_norm[:,np.newaxis,np.newaxis]
    ## One overall normalization for each order
    overall_norm = np.median(allthrumat, axis=1)
    norm_allthrumat = allthrumat / overall_norm[:,np.newaxis,:]
    final_thrumat = np.median(norm_allthrumat, axis=0)
    #final_thrumat = np.zeros((Nobj,Norder))
    #norm = np.zeros((Nthru, Norder))
    #for irow in range(Nthru):
    #    for iord in range(Norder):
    #        norm[irow,iord] = np.median(allthrumat[irow,:,iord])
    #norm_allthrumat = allthrumat / norm[:, np.newaxis, :]
    for iobj in range(Nobj):
        for iord in range(Norder):
            #final_thrumat[iobj,iord] = np.median(norm_allthrumat[:,iobj,iord])
            print("obj={:2} ord={} thru = {:.2f} +/- {:.3f}".format(
                    iobj,iord,final_thrumat[iobj,iord],
                    np.std(norm_allthrumat[:,iobj,iord])))
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(Nthru,Norder,figsize=(5*Norder,5*Nthru))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for irow,row in enumerate(tab):
        thrufname = get_file(row["FILE"], workdir, "d")
        thru1dfname = os.path.join(workdir, os.path.basename(thrufname)[:-5]+".1d.ms.fits")
        with fits.open(thru1dfname) as hdul:
            data = hdul[0].data
        for iobj in range(Nobj):
            for iord in range(Norder):
                itrace = iord + iobj*Norder
                xarr = np.arange(fiberconfig[4][iord][0], fiberconfig[4][iord][1]+1)
                #axes[iord].plot(xarr, data[itrace,xarr]/norm_allthrumat[irow,iobj,iord], lw=.5)
                #axes[irow,iord].plot(xarr, data[itrace,xarr]/allthrumat[irow,iobj,iord], lw=.5)
                #axes[iord].plot(xarr, data[itrace,xarr], lw=.5)
                #axes[irow,iord].plot(xarr, data[itrace,xarr]/final_thrumat[iobj,2], lw=.5)
                #axes[irow,iord].plot(xarr, data[itrace,xarr]/final_thrumat[iobj,iord], lw=.5)
                axes[irow,iord].plot(xarr, data[itrace,xarr]/final_thrumat[iobj,4], lw=.5)
                axes[irow,iord].set_xlim(xarr[0], xarr[-1])
    fig.tight_layout()
    fig.savefig("thrutest.png")
    plt.show()
    
    fig, axes = plt.subplots(1,Norder,figsize=(6*Norder,6))
    objnumarr = np.arange(Nobj)
    for iord in range(Norder):
        ax = axes[iord]
        ax.axhline(1.0,color='k',linestyle=':')
        for i in range(Nthru):
            ax.plot(objnumarr, norm_allthrumat[i,:,iord], label=str(i))
        ax.set_xlabel("iobj")
        ax.set_title("Order {}".format(trueord[iord]))
    ylim = [np.inf, -np.inf]
    for ax in axes:
        ylim[0] = min(ylim[0], ax.get_ylim()[0])
        ylim[1] = max(ylim[1], ax.get_ylim()[1])
    for ax in axes:
        ax.set_ylim(ylim)
    fig.tight_layout()
    
    fig, axes = plt.subplots(1,Norder,figsize=(6*Norder,6))
    objnumarr = np.arange(Nobj)
    for iord in range(Norder):
        ax = axes[iord]
        ax.axhline(1.0,color='k',linestyle=':')
        for i in range(Nthru):
            ax.plot(objnumarr, norm_allthrumat[i,:,iord], label=str(i))
        ax.set_xlabel("iobj")
        ax.set_title("Order {}".format(trueord[iord]))
    ylim = [np.inf, -np.inf]
    for ax in axes:
        ylim[0] = min(ylim[0], ax.get_ylim()[0])
        ylim[1] = max(ylim[1], ax.get_ylim()[1])
    for ax in axes:
        ax.set_ylim(ylim)
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axhline(1.0,color='k',linestyle=':')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(Nthru):
        for iord in range(Norder):
            ax.plot(objnumarr, norm_allthrumat[i,:,iord], color=colors[iord % len(colors)],)
        ax.set_xlabel("iobj")
    fig.tight_layout()
    
    plt.show()

def tmp():    
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
    m2fs_extract_sum_aperture(dbname, workdir, fiberconfig, calibconfig, Nextract=4)
    ### Horne extraction with flat as profile
    m2fs_extract_horne_flat(dbname, workdir, fiberconfig, calibconfig, Nextract=4)
    ### GHLB fit flats as profiles
    m2fs_fit_flat_profiles(dbname, workdir, fiberconfig, calibconfig)
    ### Horne extraction with GHLB fit as profile
    m2fs_extract_horne_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5)
    ### Spline extraction with GHLB fit as profile
    m2fs_extract_spline_ghlb(dbname, workdir, fiberconfig, calibconfig, Nextract=5)
    
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
    

def tmp():
    ### M2FS extract
    ## Fit profile to flats
    ## TODO input a file that associates objects, flats, and arcs
    m2fs_fit_profile_ghlb(dbname, workdir, fiberconfig)
    ## TODO apply flat GHLB profiles to extract objects, including throughput correction
    
    # M2FS profile
    ysfunc, Lfunc = m2fs_get_pixel_functions(flatnames[-1],arcnames[-1],fiberconfig)
    #ysfunc, Lfunc = m2fs_get_pixel_functions(flatnames[-1],arcnames[-2],fiberconfig)
    R, eR, header = read_fits_two(flatnames[-1])
    shape = R.shape
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    deg = [3, 12]
    yscut = 2.0
    
    from m2fs_utils import fit_ghlb
    alloutput = []
    Nobj, Norder = fiberconfig[0], fiberconfig[1]
    Ntrace = Nobj*Norder
    used = np.zeros_like(R)
    modeled_flat = np.zeros_like(R)

    iobjs = range(Nobj)
    iords = range(Norder)

    start = time.time()
    import matplotlib.pyplot as plt
    
    #fig, axes = plt.subplots(Ntrace,4,figsize=(6*4,6*Ntrace))
    for iobj in iobjs:
        fig, axes = plt.subplots(Norder, 4, figsize=(6*4,6*Norder))
        for iord in iords:
            print("iobj={} iord={}".format(iobj,iord))
            itrace = iord + iobj*Norder
            output = fit_ghlb(iobj, iord, fiberconfig,
                              X, Y, R, eR,
                              ysfunc, Lfunc, deg,
                              pixel_spacing = 1,
                              yscut = yscut, maxiter1=5, maxiter2=5, sigma = 4.0)
            pfit, Rfit, Yarr, eYarr, Xmat, L, ys, mask, indices, Sprimefunc, Sprime, Sfunc, S, Pfit = output
            alloutput.append(output)
            used[indices] = used[indices] + 1
            modeled_flat[indices] += Rfit
            
            ax = axes[iord,0]
            ax.plot(L, Yarr, 'k,')
            #ax.plot(L, S, ',')
            Lplot = np.arange(L.min(), L.max()+0.1, 0.1)
            ax.plot(Lplot, Sfunc(Lplot), '-', lw=1)
            ax.set_xlabel("L"); ax.set_ylabel("S(L)")
            ax.set_title("iobj={} iord={}".format(iobj,iord))
            
            ax = axes[iord,1]
            ax.scatter(L,ys,c=Pfit, alpha=.1)
            ax.set_xlabel("L"); ax.set_ylabel("ys")
            ax.set_title("GHLB Profile")
            
            #ax = axes[iord,2]
            #ax.plot(L, Yarr - Rfit, '.')
            #ax.axhline(0, color='k', ls=':')
            #ax.set_xlabel("L"); ax.set_ylabel("R - Rfit")
            
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
            #ax.axvline(0)
            #ax.axhline(0, color='k', ls=':')
            ax.set_xlabel("(R - Rfit)/eR")
            ax.set_title("{}/{} points outside limits".format(Nbad, len(resid)))
        start2 = time.time()
        fig.savefig("GHLB_Obj{:03}.png".format(iobj), bbox_inches="tight")
        print("Saved figure: {:.1f}s".format(time.time()-start2))
        plt.close(fig)
    print("Finished huge fit: {:.1f}s".format(time.time()-start))
    #fig.savefig("huge_ghlb_flat.png", bbox_inches="tight")
    #plt.close(fig)
    #print("Saved figure: {:.1f}s".format(time.time()-start))
    #plt.show()
    
##def tmp():
##    ax = axes[2]
##    Rplot = R.copy()
##    unused = np.ones_like(Rplot, dtype=bool)
##    unused[indices] = False
##    Rplot[unused] = 0.0
##    ax.imshow(Rplot.T, origin="lower")
##    ax.set_title("Data for this trace")
##    
##    ax = axes[3]
##    fitdata = np.zeros_like(Rplot)
##    fitdata[indices] = Rfit
##    ax.imshow((Rplot - fitdata).T, origin="lower", cmap="coolwarm")
##    ax.set_title("Residuals")
##    plt.show()


##def plot_fit():
##    Pprime = np.exp(-ys**2/2.)
##    Yprime = Yarr/Pprime
##    Wprime = (Pprime/eYarr)
##    Wprime = Wprime/np.sum(Wprime)
##    fig, axes = plt.subplots(2,2,figsize=(10,10))
##    ax = axes[0,0]
##    ax.set_title("iobj={} iord={} color=ys".format(iobj,iord))
##    sc = ax.scatter(L, Yprime, c=ys, cmap="coolwarm", vmin=-yscut, vmax=yscut, alpha=.5)
##    Lmin, Lmax = L.min(), L.max()
##    Lplot = np.linspace(Lmin,Lmax,1000)
##    ax.plot(Lplot, Sprimefunc(Lplot), color='orange')
##    ax.set_xlabel("L")
##    ax.set_ylabel("R/P")
##    ax.set_ylim(0,np.percentile(Yprime,99))
##    
##    ax = axes[0,1]
##    ax.scatter(ys, Yprime, c=L, alpha=.5)
##    ax.set_xlabel("ys")
##    ax.set_ylabel("R/P")
##    ax.set_ylim(0,np.percentile(Yprime,99))
##    
##    ax = axes[1,0]
##    ax.scatter(L, ys, c=Wprime, alpha=.5, cmap="plasma")
##    ax.set_xlabel("L")
##    ax.set_ylabel("ys")
##    
##    ax = axes[1,1]
##    ax.scatter(ys, Wprime, c=L, alpha=.5)
##    ax.set_ylim(0,np.max(Wprime))
##    ax.set_xlabel("ys")
##    ax.set_ylabel("Wprime")
##    
##    plt.show()
##    
##def tmp():
    fig = plt.figure(figsize=(8,8))
    plt.imshow(used.T, origin="lower")
    plt.title("Pixels used in model")
    plt.colorbar(label="number times used")
    fig.savefig("model_used.png")

    fig = plt.figure(figsize=(8,8))
    plt.imshow(modeled_flat.T, origin="lower")
    plt.title("Model fit")
    plt.colorbar()
    fig.savefig("model_fit.png")

    fig = plt.figure(figsize=(8,8))
    plt.imshow((R - modeled_flat).T, origin="lower")
    plt.colorbar()
    plt.title("Data - model")
    fig.savefig("model_resid.png")
    
    plt.close("all")
    #plt.show()
    # M2FS extract
    # Associate arcs and flats to data
    # Forward Model Flux(obj,order,lambda)
    # Sigma clip outlier pixels (cosmic rays) when fitting
    
    # M2FS skysub
    
    print("Total time: {:.1f}".format(time.time()-start))
