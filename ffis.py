
from astropy.io import fits
import glob

from .utils import *
from .cutout import CutoutCollection




class FFIcollection(object):
    
    def __init__(self, img_data=None, sic=None):
        
        self.img_data=img_data
        self.sic = sic
        
    def get_image_data(self,data_dir=None,fname=None):
        
        img_file_list = glob.glob(data_dir+fname+'*.fits')

        
        self.img_data = [fits.getdata(f,mmap=False)[0].copy() for f in img_file_list] 
        
        headers = [fits.getheader(f,mmp=False) for f in img_file_list]
                
        #display(headers[0])    
        
        self.exptimes = np.array([h['EXPOSURE'] for h in headers])
        self.jd = np.array([h['TSTART'] for h in headers])+self.exptimes/2.
        
        
        return self


        
    def make_ffi_cutouts(self, sicbro_id, dx=5):

        
        x0 = int(self.sic['xcol'][self.sic.id.to_numpy()==sicbro_id])
        y0 = int(self.sic['ycol'][self.sic.id.to_numpy()==sicbro_id])

        cutouts = [ i[x0-dx:x0+dx+1,y0-dx:y0+dx+1] for i in self.img_data ]
            
                
        cutout_collection = CutoutCollection(cutouts,x0,y0,dx)
        cutout_collection.time = self.jd
        cutout_collection.exptimes = self.exptimes

        cutout_collection.mag = self.sic['mag'].loc[self.sic.id.to_numpy()==sicbro_id]

        
        sic_cut =  self.sic['xcol']>x0-dx
        sic_cut &= self.sic['xcol']<x0+dx
        sic_cut &= self.sic['ycol']>y0-dx
        sic_cut &= self.sic['ycol']<y0+dx
        sic_cut &= self.sic['mag'] < float(cutout_collection.mag)+5.
        
        cutout_collection.sicbro = self.sic[sic_cut]
        cutout_collection.sicbro_id = sicbro_id
        cutout_collection.mag = cutout_collection.sicbro['mag'].loc[cutout_collection.sicbro['id']==sicbro_id]
        
        return cutout_collection
