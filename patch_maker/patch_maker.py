
'''
script to construct a patch.

'''

import os
import numpy as np
import h5py
import json
import sep

from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


__all__ = ["do_phot", "make_patch"]


def do_phot(image, wcs, plot_img=False):

    # measure a spatially varying background on the image
    image = image.byteswap().newbyteorder()
    m, s = np.mean(image), np.std(image)
    bkg = sep.Background(image)

    # subtract the background
    data_sub = image - bkg
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)

    # aperture photometry
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         3.0, err=bkg.globalrms, gain=1.0)
    if plot_img:

        # show the first 10 objects results:
        for i in range(len(objects)):
            print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))

        # plot background-subtracted image
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub), np.std(data_sub)
        c = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                      vmin=m-s, vmax=m+s, origin='lower')

        # plot an ellipse for each object
        for i in range(len(objects)):
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6*objects['a'][i],
                        height=6*objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        plt.colorbar(c)
        plt.show()

    # Convert pixel coordinates to world coordinates
    pixcrd = np.array(list(zip(objects['x'], objects['y'])), dtype=np.float_)
    world = wcs.wcs_pix2world(pixcrd, 1)

    return(objects, world, flux)


def make_patch(image_reader, get_wcs_info, get_photometry, get_coord_from_cat, get_cat_entries,
               patch_name, img_list, psf_list, center, wsize,
               catalog=None, bufferarcsec=3.0, max_sources=None,
               do_phot_on_fly=False, get_phot_from_cat=False, plot_img=False,
               detection_img=None, detection_wcs=None):
    '''
    construct a patch
    '''

    # get sky position
    position = SkyCoord(center[0], center[1], unit="deg", frame="icrs")
    size = wsize * u.arcsec

    # set up hdf5 file
    if os.path.exists(patch_name):
        os.remove(patch_name)
    f = h5py.File(patch_name, 'w')

    # add image group

    grp_images = f.create_group('images')
    grp_images.attrs['filters'] = list(img_list.keys())
    grp_images.attrs['band_idx_list'] = range(len(list(img_list.keys())))

    # loop over filters
    for ii_filter, filter_name in enumerate(grp_images.attrs['filters']):
        grp_filter = f['images'].create_group(filter_name)
        grp_filter.attrs['exposures_original_names'] = img_list[filter_name]
        grp_filter.attrs['band_idx'] = ii_filter

        # loop over all images
        exp_name = []
        for idx_img in range(len(img_list[filter_name])):

            # read in image data
            hdr, sci, wht, msk = image_reader(img_list[filter_name][idx_img])
            wcs = WCS(hdr)

            # make cutouts
            image = Cutout2D(sci, position, size, wcs=wcs)
            # Update the FITS header with the cutout WCS
            weight = Cutout2D(wht, position, size, wcs=wcs)
            mask = Cutout2D(msk, position, size, wcs=wcs)
            im = np.ascontiguousarray(image.data)
            rms = np.ascontiguousarray(np.sqrt(1.0/weight.data))
            mk = np.ascontiguousarray(mask.data)

            # if plot_img:
            #     if (idx_img == 0):
            #         fig, ax = plt.subplots()
            #         m, s = np.mean(im), np.std(im)
            #         c = ax.imshow(im, interpolation='nearest', cmap='gray',
            #                       vmin=m-s, vmax=m+s, origin='lower')
            #         plt.colorbar(c)
            #         plt.show()

            # get filter name and other information
            #list_filter_name.append(filter_name)
            header_info = get_wcs_info(image)
            header_info['phot'] = get_photometry(hdr, filter_name)

            # add exposure
            exp_name.append("exp_" + str(idx_img))
            grp_exp = f['images'][filter_name].create_group("exp_" + str(idx_img))
            grp_exp.create_dataset('sci', data=im)
            grp_exp.create_dataset('rms', data=rms)
            grp_exp.create_dataset('mask', data=mk)
            grp_exp.attrs.update(header_info)
            grp_exp.create_dataset('header', data=json.dumps(dict(image.wcs.to_header())))
            grp_exp.create_dataset('psf_name', data=[np.string_(psf_list[filter_name][idx_img])])

        grp_filter.attrs['exposures'] = exp_name

    if do_phot_on_fly:
        print('do phot on', filter_name)

        # do photometry
        d_image = Cutout2D(detection_img, position, size, wcs=detection_wcs)
        d_im = np.ascontiguousarray(d_image.data)
        objects, world, flux = do_phot(d_im, d_image.wcs, plot_img=plot_img)
        flux = flux/1.0  # get_photometry(hdr, filter_name)
        print('detected ' + str(len(objects)) + ' galaxies')
        if (len(objects) == 0.0):
            f.close()
            os.remove(patch_name)
            return()
        # get parameters of galaxies, add them to hdf5
        sourcepars = []
        sourceflux = []
        sizes = np.clip(np.sqrt(objects["b"] / objects["a"])*objects["a"]*0.06, 0.03, 0.3)
        for ii_s in range(len(objects))[:max_sources]:
            fl = len(grp_images.attrs['filters'])*[flux[ii_s]]
            pr = [10, world[ii_s][0], world[ii_s][1], np.sqrt(objects[ii_s]["b"] / objects[ii_s]["a"]),
                  90.0 - objects[ii_s]["theta"] * 180. / np.pi, 2.0, sizes[ii_s]]
            sourcepars.append(pr)
            sourceflux.append(fl)

    if get_phot_from_cat:

        # select galaxies in patch
        ra, dec = get_coord_from_cat(catalog)
        coords = SkyCoord(ra * u.degree, dec * u.degree, frame="icrs")
        dist = coords.separation(position)
        sel = (dist < ((np.sqrt(2) * np.mean(wsize) + bufferarcsec) * u.arcsec))
        print('detected ' + str(len(objects)) + ' galaxies')
        if (np.sum(sel) == 0.0):
            f.close()
            os.remove(patch_name)
            return()

        # get parameters of galaxies, add them to hdf5
        sourcepars = []
        sourceflux = []
        for s in catalog[sel][:max_sources]:
            fl, pr = get_cat_entries(s, grp_images.attrs['filters'])
            sourcepars.append(pr)
            sourceflux.append(fl)

    # add mini scene info from catalog
    grp_mini = f.create_group("mini_scene")
    grp_mini.attrs['filters'] = grp_images.attrs['filters']
    grp_mini.create_dataset('sourcepars', data=sourcepars)
    grp_mini.create_dataset('sourceflux', data=sourceflux)

    f.close()


