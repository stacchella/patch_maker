
'''
construct patches for XDF.

'''

import os
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt

from patch_maker import make_patch as patcher


__all__ = ["image_reader", "get_wcs_info", "get_photometry"]


def image_reader(img_root):
    '''
    read image data
        hdr : header
        sci : science image
        wht : weight image
        mask : mask
    '''
    hdr = fits.getheader(os.path.join(img_path, img_root + "sci.fits"))
    sci = fits.getdata(os.path.join(img_path, img_root + "sci.fits"))
    wht = fits.getdata(os.path.join(img_path, img_root + "wht.fits"))
    mask = np.zeros(sci.shape)
    return(hdr, sci, wht, mask)


def get_wcs_info(image):
    '''
    crval: sky coordinates of the reference pixel
    crpix: pixel coordinates of the reference pixel
    CD: CD matrix
    W: W matrix
    dpix_dsky: matrix [dpix/dRA, dpix/dDec]
    scale: scale matrix (pixels per arcsecond)
    '''
    psize = np.array(image.shape)
    crpix = np.floor(0.5 * psize)
    crval = image.wcs.wcs_pix2world(crpix[None, :], 0)[0, :2]
    if image.wcs.wcs.has_cd():
        CD = image.wcs.wcs.cd
    elif image.wcs.wcs.has_pc():
        CD_11 = image.wcs.wcs.cdelt[0]*image.wcs.wcs.pc[0][0]
        CD_12 = image.wcs.wcs.cdelt[1]*image.wcs.wcs.pc[0][1]
        CD_21 = image.wcs.wcs.cdelt[0]*image.wcs.wcs.pc[1][0]
        CD_22 = image.wcs.wcs.cdelt[1]*image.wcs.wcs.pc[1][1]
        CD = np.array([[CD_11, CD_12], [CD_21, CD_22]])
    else:
        print('no CD or PC matrix in WCS')
    W = np.eye(2)
    W[0, 0] = np.cos(np.deg2rad(crval[-1]))
    dpix_dsky = np.matmul(np.linalg.inv(CD), W)
    scale = np.linalg.inv(CD * 3600.0)
    keys = ['crpix', 'crval', 'CD', 'W', 'dpix_dsky', 'scale']
    values = [crpix, crval, CD, W, dpix_dsky, scale]
    return(dict(zip(keys, values)))


def get_photometry(header, filter_name):
    '''
    photometric conversion, flux to image units
    '''
    return(1.0)



def get_coord_from_cat(catalog):
    '''
    write function that returns ra and dec in degrees (ICRS frame)
    '''
    return(None)


def get_cat_entries(catrow, filters):
    '''
    get catalog entries needed for scene:
        id      : id of object
        ra, dec : position on the sky
        q       : axis ratio
        pa      : position angle
        n       : Sersic index
        rh      : half-light radius
        fluxes  : fluxes in different filters (as provided)
    '''
    return(None)


########## read UDF ##########

# define paths

img_path = '/Volumes/Tacchella/Work/Data/UDF/XDF/'
patch_path = '/Volumes/Tacchella/Work/Data/tmp/'


# set up image and PSF lists

number_of_fakes = 1

psf_list = {"f435w": number_of_fakes*["gmpsf_30mas_hst_f814w_ng4.h5"],
            "f606w": number_of_fakes*["gmpsf_30mas_hst_f814w_ng4.h5"],
            "f775w": number_of_fakes*["gmpsf_30mas_hst_f814w_ng4.h5"],
            "f814w": number_of_fakes*["gmpsf_30mas_hst_f814w_ng4.h5"],
            "f850lp": number_of_fakes*["gmpsf_30mas_hst_f850lp_ng4.h5"],
            "f125w": number_of_fakes*["gmpsf_hst_f160w_ng3.h5"],
            "f140w": number_of_fakes*["gmpsf_hst_f160w_ng3.h5"],
            "f160w": number_of_fakes*["gmpsf_hst_f160w_ng3.h5"],
            }

image_list = {"f435w": number_of_fakes*["hlsp_xdf_hst_acswfc-30mas_hudf_f435w_v1_"],
              "f606w": number_of_fakes*["hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_"],
              "f775w": number_of_fakes*["hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_"],
              "f814w": number_of_fakes*["hlsp_xdf_hst_acswfc-30mas_hudf_f814w_v1_"],
              "f850lp": number_of_fakes*["hlsp_xdf_hst_acswfc-30mas_hudf_f850lp_v1_"],
              "f125w": number_of_fakes*["hlsp_xdf_hst_wfc3ir-60mas_hudf_f125w_v1_"],
              "f140w": number_of_fakes*["hlsp_xdf_hst_wfc3ir-60mas_hudf_f140w_v1_"],
              "f160w": number_of_fakes*["hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_"]
              }

zeropoint_list = {"f435w": 1.0/(3.142e-19),
                  "f606w": 1.0/(7.795e-20),
                  "f775w": 1.0/(9.934e-20),
                  "f814w": 1.0/(7.036e-20),
                  "f850lp": 1.0/(1.517e-19),
                  "f125w": 1.0/(2.25e-20),
                  "f140w": 1.0/(1.47e-20),
                  "f160w": 1.0/(1.93e-20)
                  }


# detection image

detection_img_hdr, detection_img_sci, detection_img_wht, detection_img_msk = image_reader(image_list["f160w"][0])
detection_img_wcs = WCS(detection_img_hdr)

# detection_img_hdr, detection_img1_sci, detection_img1_wht, detection_img1_msk = image_reader(image_list["f160w"][0])
# detection_img2_hdr, detection_img2_sci, detection_img2_wht, detection_img2_msk = image_reader(image_list["f140w"][0])
# detection_img3_hdr, detection_img3_sci, detection_img3_wht, detection_img3_msk = image_reader(image_list["f125w"][0])

# detection_img_sci = (detection_img1_wht*detection_img1_sci + detection_img2_wht*detection_img2_sci + detection_img3_wht*detection_img3_sci)/np.sqrt(detection_img1_wht**2 + detection_img2_wht**2 + detection_img3_wht**2)
# detection_img_wcs = WCS(detection_img_hdr)

# detection_img_sci[np.isnan(detection_img_sci)] = 0.0


# grid of centers

pix_scale_detection_img = 0.06  # arcsec per pixel
overlap = 1.8  # in arcsec
size_x, size_y = 6.0, 6.0  # in arcsec
wsize = np.array([size_x, size_y])


dist_centers_x = size_x-overlap
dist_centers_y = size_y-overlap

X = np.arange(0, detection_img_hdr['NAXIS1'], dist_centers_x/pix_scale_detection_img, dtype=int)
Y = np.arange(0, detection_img_hdr['NAXIS2'], dist_centers_y/pix_scale_detection_img, dtype=int)

xx, yy = np.meshgrid(X, Y)

x_values, y_values = xx.flatten(), yy.flatten()

x_cen, y_cen = [], []

for (x_i, y_i) in zip(x_values, y_values):
    if (np.nansum(detection_img_sci[x_i-int(size_x/pix_scale_detection_img):x_i+int(size_x/pix_scale_detection_img), y_i-int(size_y/pix_scale_detection_img):y_i+int(size_y/pix_scale_detection_img)]) != 0.0):
        x_cen.append(x_i)
        y_cen.append(y_i)


# just look at the data...
# plt.imshow(detection_img_sci, vmin=np.nanpercentile(detection_img_sci, 16), vmax=np.nanpercentile(detection_img_sci, 84))
# plt.plot(x_values, y_values, 'x', color='orange')
# plt.plot(x_cen, y_cen, '.', color='red')
# plt.show()


# convert centers to RA / DEC

pixcrd_cen = np.array(list(zip(x_cen, y_cen)), dtype=np.float_)
world_cen = detection_img_wcs.wcs_pix2world(pixcrd_cen, 1)


for ii_cen in range(len(world_cen))[::10]:
    center = world_cen[ii_cen]
    patch_name = os.path.join(patch_path, 'patch_udf_test_' + str(ii_cen) + '.h5')
    patcher(image_reader, get_wcs_info, get_photometry, get_coord_from_cat, get_cat_entries, patch_name, image_list, psf_list, center, wsize, catalog=None, bufferarcsec=0.0, max_sources=30, do_phot_on_fly=True, plot_img=False, detection_img=detection_img_sci, detection_wcs=detection_img_wcs)





'''
print('move patch to ascent \nscp -r', patch_name, 'stacchella@login1.ascent.olcf.ornl.gov://gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data/')
'''
'''
# move patch to ascent

scp -r /Users/sandrotacchella/Desktop/patch_construction/test_patch_large.h5 stacchella@login1.ascent.olcf.ornl.gov://gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data/

'''

