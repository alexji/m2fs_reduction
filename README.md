# m2fs_reduction
A reduction pipeline for M2FS data.

Under development.

# Author
* Alex Ji (Carnegie Observatories)

With major assistance from Dan Kelson (Carnegie Observatories).

Including some code adpated from T. Hansen, J. Simon, G. Blanc

# Requirements
* Attempted to be python 2/3 compatible, but only tested in python 3.6
* `numpy`, `scipy`, `astropy` libraries
* [Source extractor](https://www.astromatic.net/software/sextractor), callable from the command line

# To run
* Create database file, make sure it is formatted correctly (TODO include example)
* Create line identification file (example: `data/Mg_Wide_r_id.txt`)
* Create fiber configuration file (example: `data/Mg_Wide_r.txt`)
* Create calibration configuration file associating science frames, flat frames, arc frames (example: `nov2017run.txt`) (TODO generate automatically) 
* Edit `pipeline.py` for the database file, working directory, calibration configuration name, fiber configuration name, and some extraction parameters if needed (TODO make a configuration file for all this)
* Run `python pipeline.py`

# To be completed
- [ ] throughput correction
- [ ] basic sky subtraction
- [ ] test on other configurations
- [ ] refactor algorithm parameters better
- [ ] reduction configuration parameter file
- [ ] database file example

### Wish list
- [ ] CPD to match pre-identified arcs to actual arcs
- [ ] more line identifications and fiber configurations
- [ ] remove source extractor dependency
- [ ] solve for trace from the arcs only, not needing flats
- [ ] add wavelength calibration info (e.g. RMS) to output header
- [ ] ensure this works for non 2x2 binning
- [ ] some better stuff for trace peak checking (right now it just fails if it finds the wrong number of peaks)
- [ ] wavelength dependence in GHLB object profiles
- [ ] allow minor (linear) shifts in fiber locations relative to the flat when extracting
- [ ] simultaneous fit of all object profiles, rather than one-by-one
- [ ] simultaneous extraction of all objects on a frame, rather than one-by-one
- [ ] simultaneous extraction of multiple exposures (not sure if this is theoretically possible because sky is not subtracted before extraction...)
- [ ] routines to filter and mask images before extraction (e.g. removing or masking the Littrow ghosts)
- [ ] spline fit extraction using flat as object profile?
- [ ] quick plotting of reduced spectra

# Brief algorithm descriptions
* Bias subtraction is done with just the overscan regions
* Dark subtraction is simple exposure-time weighted scaling of a master dark frames
* Flats are traced by fitting Gaussians at a fixed pixel step, then refitting those parameters with a 7th degree Legendre polynimal (for trace) and 3rd degree polynomial (for stdev)
* Line identification is done by running source extractor on filtered arc frames to centroid line positions, then matching against the input line identifications (currently with a kdtree, so it has to be pretty close).
* Wavelength solution is a global fit of X, Y positions of all identified features (legendre polyomial of degree 3, 4, 5 in object, order, and wavelength). Includes including per-object offsets to account for relative drift to a global solution.
* Scattered light is fit to regions between tetrises with a 2D Legendre polynomial of degree (5,5)
* Object (fiber) profiles are fit with Gauss-Hermite Legendre basis (GHLB), see [Kelson 2005](http://code.obs.carnegiescience.edu/Algorithms/ghlb/view). Each fiber is fit separately. Degree is 0 for Legendre, 10 for Hermite. Note that degree=0 for Legendre means it is just a Gauss-Hermite ignoring the wavelength dependence of the profile (empirically, the ghosts and pixel sampling affect you too much to get a good fit with the wavelength dependence).
* Extraction is done multiple ways:
  * simple aperture sum (`*sum_specs.fits`)
  * Horne extraction with flat as object profile (`*horneflat_specs.fits`)
  * Horne extraction with GHLB as object profile (`*horneghlb_specs.fits`)
  * [Spline fit extraction](http://code.obs.carnegiescience.edu/Algorithms/ghlb/view) with GHLB as object profile (`*splineghlb_specs.fits`)

The recommended extraction procedure is the spline fit extraction. This is the most robust to cosmic rays and should have the best object profiles (minimally affected by ghosts, overlapping orders, and other weird things).
However I bet there will be cases causing the extraction model to be wrong, so make sure to examine the residuals, GHLB profile fits, etc. before using for science.
