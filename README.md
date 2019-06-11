# patch_maker

Simple tool to create patches for forcepho.

Structure of a patch:

patch
|
-- images
|    |
|    -- filters
|        |
|        -- exposures
|             |
|             -- sci stamp
|             -- rms stamp
|             -- mask stamp
|             -- hdr
|             -- wcs
|             -- photometry / zeropoint
|             -- PSF
|
-- mini scene

Requiered python modules:
os, numpy, h5py, json, sep, astropy, matplotlib

