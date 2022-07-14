from photutils.aperture import CircularAperture, aperture_photometry
from astropy.table import vstack
from astropy.io import fits
import fitsio
import numpy as np


class ApPhot(object):
    
    def __init__(self,fnames,x,y,aperture_radii=[1.,1.5,2.,2.5,3.,3.5,4.]):
        
        self.fnames=sorted(fnames)
        self.x=x
        self.y=y
        self.aperture_radii = aperture_radii
        self.apertures = None

    def _make_apertures(self, apradii, dx):

        x0,y0 = int(self.x), int(self.y)
                 
        circapers = [CircularAperture((self.x-x0+dx,self.y-y0+dx), aprad ) for aprad in self.aperture_radii]
        self.apertures = circapers
        
        return circapers
        

        
    def _do_photometry(self, fname, bkg_estimate=5.23, dx=8):

        x0,y0 = int(self.x), int(self.y)

        x_range = x0-dx, x0+dx
        y_range = y0-dx, y0+dx 
        
        #circapers = self._make_apertures(self.aperture_radii, dx)

        
        fits=fitsio.FITS(fname)
        
        img = fits[0][x_range[0]:x_range[1], y_range[0]:y_range[1]]
        err = fits[1][x_range[0]:x_range[1], y_range[0]:y_range[1]]
        hdr = fits[0].read_header()
                
        #(img,err),hdr = fitsio.read(fname, rows=x_range, columns=y_range, header=True)
                
        #err = fitsio.read(fname, rows=x_range, columns=y_range,).T
        
        #hdulist = fits.open(fname,mmap=False)
        #img=hdulist[0].data.T
        #err=hdulist[1].data.T
        #hdr=hdulist[0].header
        time = hdr['TSTART'] 
        
        result = aperture_photometry(img - bkg_estimate, self.apertures, error=err, method='subpixel')

        result['bjd'] = time
        
        #del img
        #del err
        #del hdr
        #hdulist.close()
        
        #del hdulist
        
        fits.close()
        del fits
        
        return result
    
    
    def get_lightcurve(self, dx=5):
        
        self.apertures = self._make_apertures(apradii=self.aperture_radii, dx=dx)

        all_results = vstack([ self._do_photometry(f,dx=dx) for f in self.fnames])
        
        [all_results.rename_column('aperture_sum_'+str(i), 'apflux_r_'+str(int(ap.r*10))) for i,ap in enumerate(self.apertures)]
        [all_results.rename_column('aperture_sum_err_'+str(i), 'apflux_r_'+str(int(ap.r*10))+'_err') for i,ap in enumerate(self.apertures)]
        
        return all_results
        
        
    
