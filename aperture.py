from photutils.aperture import CircularAperture, aperture_photometry
from astropy.table import vstack
from astropy.io import fits




class PhotObject(object):
    
    def __init__(self,fnames,x,y,aperture_radii=[1.,1.5,2.,2.5,3.,3.5,4.]):
        
        self.fnames=sorted(fnames)
        self.x=x
        self.y=y
        self.apertures=self._make_apertures(aperture_radii)

    def _make_apertures(self, apradii):

        x0, y0 = self.x, self.y
        circapers = [CircularAperture((x0,y0), aprad ) for aprad in apradii]
        return circapers
        
        
    def _do_photometry(self, fname, bkg_estimate=5.23, sicid=1):
        
        hdulist = fits.open(fname,mmap=False)
        img=hdulist[0].data.T
        err=hdulist[1].data.T
        hdr=hdulist[0].header
        time = hdr['TSTART'] 
        
        result = aperture_photometry(img - bkg_estimate, self.apertures, error=err, )
        result['bjd'] = time
        
        del img
        del err
        del hdr
        hdulist.close()
        
        del hdulist
        
        return result
    
    
    def get_lightcurve(self,sicid=1):
        
        all_results = vstack( [self._do_photometry(f,sicid) for f in tqdm(self.fnames)] )
        
        [all_results.rename_column('aperture_sum_'+str(i), 'apflux_r_'+str(ap.r)) for i,ap in enumerate(self.apertures)]
        [all_results.rename_column('aperture_sum_err_'+str(i), 'apflux_r_'+str(ap.r)+'_err') for i,ap in enumerate(self.apertures)]
        
        return vstack(all_results)
    