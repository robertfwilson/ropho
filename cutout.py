from astropy.io import fits

from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup, DBSCANGroup
from photutils.psf import (IterativelySubtractedPSFPhotometry, BasicPSFPhotometry,DAOPhotPSFPhotometry,PRFAdapter)
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.detection import find_peaks
#from astropy.visualization import simple_norm, SqrtStretch, SinhStretch, LinearStretch
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch

from photutils.aperture import CircularAperture, RectangularAperture, aperture_photometry


from astropy.modeling.fitting import LevMarLSQFitter, SLSQPLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table, vstack

from photutils.psf.models import EPSFModel
from tqdm import tqdm


from .utils import pd, np, plt



class CutoutCollection(object):
    
    def __init__(self, cutouts, x0, y0, dx):
        
        self.x0=int(x0)
        self.y0=int(y0)
        self.dx=dx
        self.cutouts = cutouts
        self.master=self._create_master_image()
        #self.psfmod = self._get_psf_model()
        
        
    def _get_psf_model(self, filtername='F146', spectype='M0V', sca=7, difference_image=False):

        fname = '/Users/rfwilso1/mypy/ropho/data/psf/'+filtername.lower()+'/wfi_psf_'+spectype.upper()+'_SCA{:02d}'.format(sca)+'.fits'
        
        psfdata = fits.open(fname)[0].data
        prfdata = fits.open(fname)[1].data

        oversampling = psfdata.shape[0]/prfdata.shape[0]
        midpoint = psfdata.shape[0]//2
        psf_beg = int(midpoint-self.dx*oversampling)
        psf_end = int(midpoint+self.dx*oversampling)

        psf_beg = max(0,psf_beg)
        psf_end = min(psfdata.shape[0], psf_end)
        
        psfmod = EPSFModel(psfdata[psf_beg:psf_end,psf_beg:psf_end],
                           oversampling=oversampling, normalize=True,)
        
        psfmod.x_0.bounds=(0,self.master.shape[0])
        psfmod.y_0.bounds=(0,self.master.shape[1])

        if not(difference_image):
            psfmod.flux.bounds=(0,np.inf)

        return psfmod
        
    def _create_master_image(self):
        
        master = sum(i for i in self.cutouts)/len(self.cutouts)
        
        self.master=master
        return master
    
    
    def _plot_nearby_stars(self):
        
        if self.master is None:
            self._create_master_image()
        
        dx=self.dx
        
        master=self.master
        cat = self.sicbro
        x0 = self.x0-dx
        y0 = self.y0-dx

        norm1 = ImageNormalize(master, interval=PercentileInterval(95), stretch=SqrtStretch())
        plt.imshow(master, origin='lower',  cmap='inferno', norm=norm1)
        plt.colorbar()

        plt.scatter(cat['ycol'].to_numpy()-y0,cat['xcol'].to_numpy()-x0, marker='o',
                    c='0.75', lw=1, edgecolor='k', s=(28.5-cat['mag'].to_numpy())**2,
                    alpha=0.75)
        
        plt.show()
        
        return None
    
            
        
    def set_up_photometry(self, sigma_psf=2., fix_centroid=False, aperture=2):
        
        bkgrms = MADStdBackgroundRMS()
        std = bkgrms(self.cutouts[0])
        
        daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
        mmm_bkg = MMMBackground()
        fitter = LevMarLSQFitter()
            
        
        self.psfmod.x_0.fixed=fix_centroid
        self.psfmod.y_0.fixed=fix_centroid
        
        
        phot = BasicPSFPhotometry(daogroup, mmm_bkg, self.psfmod, fitshape=(9,9), finder=None,aperture_radius=aperture) 
                
        return phot
    
    
    def get_star_positions(self, mag_lim=25.):
        
        x0=self.x0
        y0=self.y0
        dx=self.dx
        
        cat = self.sicbro
        
        mag_cut = cat['mag']<mag_lim

        y_guess = cat['xcol'].to_numpy()[mag_cut]-x0+dx
        x_guess = cat['ycol'].to_numpy()[mag_cut]-y0+dx
        
        known_positions = pd.DataFrame(np.array([x_guess, y_guess]).T, 
                               columns=['x_0','y_0'], )
        
        init_guesses = Table.from_pandas(known_positions)
        
        distance_argsort = np.argsort(np.sqrt((x_guess-dx)**2. + (y_guess-dx)**2.))
        
        init_guesses=init_guesses[distance_argsort]
        init_guesses['id'] = np.arange(len(init_guesses))+1
        
        
        return init_guesses
    
    def get_apr_lightcurve(self, fix_centroid=True, aperture_radius=2):


        initguesses = self.get_star_positions()
        x,y = initguesses['x_0'][0], initguesses['y_0'][0]
        
        #x,y=self.dx,self.dx
        x-= 0.25
        y-= 0.25

        
        circaper = CircularAperture((x,y), aperture_radius)
        rectaper = RectangularAperture((x,y), 3,3)
        
        phot_results=[]
        for i in self.cutouts:
            phot_results.append( aperture_photometry(i, [circaper, rectaper]) )
        
        return vstack(phot_results), circaper, rectaper
    
    def get_psf_lightcurve(self, fix_centroid=False, use_master=False, aperture=1.5):
                
        initguesses = self.get_star_positions()
        
        if use_master:
            
            masterimg = self.master
            master_phot = self.set_up_photometry(aperture=aperture)
            
            result_master = master_phot(image=masterimg, init_guesses=initguesses)
            mastguesses = Table()
            mastguesses['id'] = result_master['id']
            mastguesses['x_0'] = result_master['x_fit']#,'y_fit','flux_fit']
            mastguesses['y_0'] = result_master['y_fit']#,'y_fit','flux_fit'
            mastguesses['f_0'] = result_master['flux_fit']#,'y_fit','flux_fit']
            mastguesses = mastguesses[np.argsort(mastguesses['id'])]   
            
        else:
            mastguesses = initguesses
        
        phot = self.set_up_photometry(fix_centroid=fix_centroid)
        
        psf_results=[]
        
        for i in tqdm(self.cutouts):
            
            result_tab = phot(image=i, init_guesses=mastguesses)
            psf_results.append(result_tab[result_tab['id']==1])
    
        lcnew = vstack(psf_results)
        self.lc=lcnew
        
        return lcnew
        
    def correct_flux(lc):
    
        dx=lc['x_fit']-np.median(lc['x_fit'])
        dy=lc['y_fit']-np.median(lc['y_fit'])

        X = np.array([dx, dy, dx*dy, dx**2., dy**2.]).T
        Y_psf = np.array(norm(lc['flux_fit']) )
        Y_apr = np.array(norm(lc['flux_0']) )

        reg_psf = LinearRegression().fit(X,Y_psf)
        reg_apr = LinearRegression().fit(X,Y_apr)

        psf_corr=Y_psf/reg_psf.predict(X)
        apr_corr=Y_apr/reg_apr.predict(X)

        return psf_corr, apr_corr
    
    
    def plot_lc_summary(self):
        

        fig = plt.figure(tight_layout=True, figsize=(6,6))
        gs = gridspec.GridSpec(3, 2)

        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
        ax20 = fig.add_subplot(gs[2, 0], )
        ax21 = fig.add_subplot(gs[2, 1], )

        lc=self.lc

        ax0.scatter(range(len(lc)), norm(lc['flux_fit'] )-1. , color='brown', facecolor='w',
                    label='psf', marker='.')
        ax0.scatter(range(len(lc)), norm(lc['flux_0'], )-1., color='dodgerblue', facecolor='w', 
                    label='apr', marker='.')
        #ax0.scatter(range(len(lc)), f_psf_corr-1., label='psf' , color='brown', marker='.')
        #ax0.scatter(range(len(lc)), f_apr_corr-1., label='apr' , color='dodgerblue',marker='.' )

        ax0.legend(ncol=2)


        ax1.errorbar(range(len(lc)), lc['x_fit']-np.mean(lc['x_fit']),fmt='.',capsize=3, label='$\mathregular{\\delta x}$')
        ax1.errorbar(np.arange(len(lc))+.25, lc['y_fit']-np.mean(lc['y_fit']), fmt='.',capsize=3, label='$\mathregular{\\delta y}$')
        ax1.legend(ncol=2)



        ax20plot = ax20.scatter(lc['x_fit']-np.mean(lc['x_fit']), lc['y_fit']-np.mean(lc['y_fit']),
                   c=norm(lc['flux_fit']), marker='.' )

        #plt.colorbar(ax20plot, ax=ax20)

        ax21plot = ax21.scatter(lc['x_fit']-np.mean(lc['x_fit']), lc['y_fit']-np.mean(lc['y_fit']),
                   c=norm(lc['flux_0']), marker='.' )

        #plt.colorbar(ax21plot, ax=ax21)

        ax21.text(0.1, 0.1, 'apr', transform=ax21.transAxes)
        ax20.text(0.1, 0.1, 'psf', transform=ax20.transAxes)


        ax21.set_xlabel('$\mathregular{\\delta x}$ (pixel)')
        ax21.set_ylabel('$\mathregular{\\delta y}$ (pixel)')
        ax20.set_xlabel('$\mathregular{\\delta x}$ (pixel)')
        ax20.set_ylabel('$\mathregular{\\delta y}$ (pixel)')

        ax1.set_ylabel('Centroid (pixel)')
        ax1.set_xlabel('Cadence Number')
        ax0.set_xlabel('Cadence Number')
        ax0.set_ylabel('$\mathregular{\\delta f/f}$')

        ax0.set_title('sicbro id: {:.0f}'.format(self.sicbro_id), fontweight='bold')

        plt.tight_layout()
        fig.align_labels()
        
        plt.show()
    
    
