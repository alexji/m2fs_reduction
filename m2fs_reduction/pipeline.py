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
from m2fs_utils import m2fs_get_pixel_functions
from m2fs_utils import m2fs_ghlb_extract

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
    
    m2fs_trace_orders(masterflatname, fiberconfig, trace_degree=7, stdev_degree=3, make_plot=True)
    
    for fname in fnames:
        m2fs_trace_orders(fname, fiberconfig, make_plot=True)
    
    mark_finished(workdir, "traceflat")

def m2fs_wavecal_find_sources(dbname, workdir):
    if check_finished(workdir, "wavecal-findsources"): return
    ## Get list of arcs to process
    tab = ascii.read(dbname)
    filter = np.unique(tab["FILTER"])[0]
    config = np.unique(tab["CONFIG"])[0]
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    print("Found {} arc frames".format(len(arctab)))
    fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    start = time.time()
    for fname in fnames:
        ## Find sources
        m2fs_wavecal_find_sources_one_arc(fname, workdir)
    print("Finding all sources took {:.1f}s".format(time.time()-start))
    mark_finished(workdir, "wavecal-findsources")
    
def m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig):
    if check_finished(workdir, "wavecal-identifysources"): return
    
    ## TODO use fiberconfig to get this somehow!!!
    identified_sources = ascii.read("data/Mg_Wide_r_id.txt")
    
    ## Get list of arcs to process
    tab = ascii.read(dbname)
    filter = np.unique(tab["FILTER"])[0]
    config = np.unique(tab["CONFIG"])[0]
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    print("Found {} arc frames".format(len(arctab)))
    fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    start = time.time()
    for fname in fnames:
        m2fs_wavecal_identify_sources_one_arc(fname, workdir, identified_sources)
    print("Identifying all sources took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "wavecal-identifysources")

def m2fs_wavecal_fit_solution(dbname, workdir, fiberconfig):
    if check_finished(workdir, "wavecal-fitsolution"): return
    
    ## Get list of arcs to process
    tab = ascii.read(dbname)
    filter = np.unique(tab["FILTER"])[0]
    config = np.unique(tab["CONFIG"])[0]
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    print("Found {} arc frames".format(len(arctab)))
    fnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    start = time.time()
    for fname in fnames:
        m2fs_wavecal_fit_solution_one_arc(fname, workdir, fiberconfig, make_plot=True)
    print("Fitting wavelength solutions took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "wavecal-fitsolution")

def m2fs_fit_profile_ghlb(dbname, workdir, fiberconfig):
    if check_finished(workdir, "flat-ghlb"): return
    
    tab = ascii.read(dbname)
    flattab = tab[tab["EXPTYPE"]=="Flat"]
    print("Found {} flats".format(len(flattab)))
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    ## HACK TODO need to do something about picking this, but for now....
    arctab = arctab[arctab["EXPTIME"] < 30]
    print("Found {} arcs".format(len(flattab)))
    
    #masterflatname = os.path.join(workdir, "master_flat.fits")
    flatfnames = [get_file(x, workdir, "d") for x in flattab["FILE"]]
    arcfnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    
    start = time.time()
    for flatfname, arcfname in zip(flatfnames, arcfnames):
        m2fs_ghlb_extract(flatfname, flatfname, arcfname, fiberconfig,
                          yscut=2.0, deg=[3,12], sigma=4.0,
                          make_plot=True, make_obj_plots=True)
    print("Fitting GHLB to flats took {:.1f}".format(time.time()-start))
    mark_finished(workdir, "flat-ghlb")

#################################################
# script to run
#################################################
if __name__=="__main__":
    start = time.time()
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
    arctab = tab[tab["EXPTYPE"]=="Comp"]
    flattab= tab[tab["EXPTYPE"]=="Flat"]
    objtab = tab[tab["EXPTYPE"]=="Object"]
    arcnames = [get_file(x, workdir, "d") for x in arctab["FILE"]]
    flatnames= [get_file(x, workdir, "d") for x in flattab["FILE"]]
    objnames = [get_file(x, workdir, "d") for x in objtab["FILE"]]
    
    fiberconfig = m2fs_parse_fiberconfig(fiberconfigname)
    
    m2fs_biastrim(dbname, workdir)
    m2fs_darksub(dbname, workdir)
    m2fs_traceflat(dbname, workdir, fiberconfig)
    
    ### M2FS wavecal
    ## Find sources in 2D arc spectrum (currently a separate step running sextractor)
    m2fs_wavecal_find_sources(dbname, workdir)
    # NOTE: IF AN ARC HAS NOT BEEN IDENTIFIED, IT NEEDS TO BE DONE MANUALLY NOW
    ## Identify features in 2D spectrum with coherent point drift
    m2fs_wavecal_identify_sources(dbname, workdir, fiberconfig)
    ## Use features to fit Xccd,Yccd(obj, order, lambda)
    m2fs_wavecal_fit_solution(dbname, workdir, fiberconfig)
    
    ### M2FS extract
    ## Fit profile to flats
    ## TODO input a file that associates objects, flats, and arcs
    m2fs_fit_profile_ghlb(dbname, workdir, fiberconfig)
    ## TODO apply GHLB profiles to extract objects, including throughput correction
    
    
def tmp():
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
