from photutils.aperture import CircularAperture
import fitsio
import numpy as np
import pandas as pd



class ApPhot(object):

    def __init__(self,fnames,x,y,aperture_radii=[1.,1.5,2.,2.5,3.,3.5,4.], cache_fits=True):
        
        self.fnames=sorted(fnames)
        self.x=x
        self.y=y
        self.aperture_radii = aperture_radii
        self.apertures = None
        
        if cache_fits:
            self.fits_cache = [fitsio.FITS(f) for f in self.fnames ]
        else:
            self.fits_cache=None
            
    
    def _get_fits_object(self,f):
                
        if isinstance(f, str):
            return fitsio.FITS(f)
        else:
            return self.fits_cache[f]
        
        
    def _get_time(self, fname):
                
        fits=self._get_fits_object(fname)    
        hdr = fits[0].read_header()

        t=hdr['TSTART']

        if isinstance(fname,str):
            fits.close()
        
        return t
        
        

    def _make_apertures(self, dx=5, apradii=None, mask=True):

        if apradii is None:
            apradii = self.aperture_radii
        
        x0,y0 = int(self.x), int(self.y)
        
        if mask:
            circapers = [CircularAperture((self.x-x0+dx,self.y-y0+dx), aprad ).to_mask(method='exact') for aprad in self.aperture_radii]
        else:
            circapers = [CircularAperture((self.x-x0+dx,self.y-y0+dx), aprad ) for aprad in self.aperture_radii]
        
        self.apertures = circapers        
        return circapers


    def _get_cutout(self, fname, dx=5):
        
        x0,y0 = int(self.x), int(self.y)

        x_range = x0-dx, x0+dx
        y_range = y0-dx, y0+dx 
        
        fits = self._get_fits_object(fname)
        
        img = fits[0][x_range[0]:x_range[1]+1, y_range[0]:y_range[1]+1]
        err = fits[1][x_range[0]:x_range[1]+1, y_range[0]:y_range[1]+1]
        
        return img, err


    
    def _add_aperture_fluxes(self, img, err=None):
            
        weighted_cutouts = [apmask.multiply(img) for apmask in self.apertures]
        fluxes = [np.sum(c) for c in weighted_cutouts]
                
        if not(err is None):
            weighted_errs = [apmask.multiply(err**2.) for apmask in self.apertures]
            flux_errs = [np.sqrt(np.sum(c)) for c in weighted_errs]
            return fluxes + flux_errs
        
        return fluxes



    
    def _do_phot_faster(self, fname, bkg_estimate=5.23,dx=5):
        
        img,err = self._get_cutout(fname)
        fluxes = self._add_aperture_fluxes(img-bkg_estimate, err=err)
        
        return fluxes
        
    
    
    
    def get_lightcurve_faster(self, dx=5, use_cached_fits=True, return_numpy=False):
        
        self.apertures = self._make_apertures(apradii=self.aperture_radii, dx=dx)
        
        if use_cached_fits and not(self.fits_cache is None):            
            fnames = np.arange(len(self.fnames),dtype=int)
        else:
            fnames = self.fnames
        
        all_fluxes = np.array([self._do_phot_faster(f,dx=dx) for f in fnames] )
        times = [self._get_time(f) for f in fnames]
        
        data = np.c_[times, all_fluxes.round(2)]

        if return_numpy:
            return data
            
        
        flux_columns=['sapflux_r'+str(int(r*10)) for r in self.aperture_radii]
        flux_err_columns=['sapflux_r'+str(int(r*10))+'_err' for r in self.aperture_radii]
        columns = ['jd'] + flux_columns + flux_err_columns
        
        return pd.DataFrame(data, columns=columns)
