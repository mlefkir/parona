import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from .callds9 import start_ds9
from astroquery.heasarc import Heasarc
from astropy.time import Time
import heasoftpy as hsp
from .xmm_lc import get_lightcurve

def energy2nustarchannel(energy_keV):
    return (energy_keV - 1.6)/0.04


def nustarchannel2energy(channel_nb):
    return channel_nb*0.04+1.6


class ObservationNuSTAR:
    """
    Class for processing an XMM-Newton observation

    The folder tree is this one

    path/odf
    path/data/obsid_datadir
    path/obsid_workdir
    path/obsid_workdir/logs

    """

    def __init__(self, path, obsidentifier, src_name,slices=["all"]):
        self.ID = obsidentifier
        self.slices = slices
        self.nslices = len(self.slices)
        self.instruments = ["FPMA", "FPMB"]
        self.obs_files = {}
        self.regions = {}
        
        heasarc = Heasarc()
        self.src_name = src_name
        table = heasarc.query_object(src_name, mission='numaster')
        indx = np.take(np.where(table["OBSID"]==self.ID),0)
        self.date = Time(table[indx]["TIME"],format='mjd').iso[:10]
        
        for instr in self.instruments:
            self.obs_files[instr] = {}
            self.regions[instr] = {}

        self.energybands = [[3.0, 10.0], [10.0, 79.0]]
        self.energy_range = [np.min(np.array(self.energybands).flatten()), np.max(
            np.array(self.energybands).flatten())]

        self.check_repertories(path)
        self.replot = True

    def check_repertories(self, path):
        if not os.path.isdir(f'{path}/data'):
            os.mkdir(f'{path}/data')
        if not os.path.isdir(f'{path}/data/{self.ID}'):
            os.mkdir(f'{path}/data/{self.ID}')
        if not os.path.isdir(f'{path}/{self.ID}_workdir'):
            os.mkdir(f'{path}/{self.ID}_workdir')
        if not os.path.isdir(f'{path}/{self.ID}_workdir/logs'):
            os.mkdir(f'{path}/{self.ID}_workdir/logs')
        if not os.path.isdir(f'{path}/{self.ID}_workdir/plots'):
            os.mkdir(f'{path}/{self.ID}_workdir/plots')
        if not os.path.isdir(f'{path}/{self.ID}_workdir/products'):
            os.mkdir(f'{path}/{self.ID}_workdir/products')

        self.datadir = f'{path}/data/{self.ID}/'
        self.workdir = f'{path}/{self.ID}_workdir/'
        self.logdir = f'{path}/{self.ID}_workdir/logs/'
        self.plotdir = f'{path}/{self.ID}_workdir/plots/'

    def calibrate(self):
        """
        Calibrate the data files with nupipeline
        """
        os.chdir(self.workdir)
        if not len(glob.glob(f"{self.workdir}/*")) > 4:
            os.system(f"nupipeline indir='{self.datadir}' steminputs='nu{self.ID}' outdir='{self.workdir}' 2>&1  | tee '{self.logdir}/nupipeline.txt'")
        for instr in self.instruments:
            self.obs_files[instr]["gti"] = f"{self.workdir}/nu{self.ID}{instr[-1]}01_cl.evt"

    def select_regions(self, src_name):
        """

        Select the source and background regions with ds9

        """
        print(f'<  INFO  > : Selection of source and background regions with ds9')

        for i,instr in enumerate(self.instruments):
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            if i ==0 :
                try:
                    python_ds9.set("regions select all")
                except:
                    python_ds9 = start_ds9()
                    
            self.obs_files[instr]["image"] = f"{self.workdir}/nu{self.ID}{instr[-1]}01_cl.evt"

            python_ds9.set("regions select all")
            python_ds9.set("regions delete")
            python_ds9.set("file "+self.obs_files[instr]["image"])
            python_ds9.set("scale asinh")
            python_ds9.set("cmap b")

            if len(glob.glob(f"{self.workdir}/products/*{instr}.reg")) != 2:
                print(
                    f"Draw FIRST the region for the source save it as a src_{instr}.reg and bkg_{instr}.reg and THEN the background")
                input("Press Enter to continue...")
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")

            python_ds9.set("zoom to fit")
            # python_ds9.set(
                # f"saveimage png {self.plotdir}/{self.ID}_{src_name}{instr}_image.png")
    def extract_events(self,src_name,low=3.0,up=79.0):
        """Extract event files for source and background"""
        print('<  INFO  > : Generating event files')
        
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            # python_ds9.set(f"saveimage png {self.plotdir}/{self.ID}_{src_name}{instr}_image.png")


            evts_src_name = f"{self.workdir}/products/{src_name}_evts_src_{instr}.fits"
            evts_bkg_name = f"{self.workdir}/products/{src_name}_evts_bkg_{instr}.fits"
            spec_src_name = f"{self.workdir}/products/{src_name}_spec_src_{instr}.fits"
            spec_bkg_name = f"{self.workdir}/products/{src_name}_spec_bkg_{instr}.fits"
                
            if len(glob.glob(f"{self.workdir}/products/{src_name}_evts_*_{instr}.fits"))!=2:
                print(f'\t\t<  INFO  > : Extracting source and background events')

                if glob.glob(evts_src_name) == []:
                    hsp.extractor(
                    filename=f"{self.workdir}/nu{self.ID}A01_cl.evt", 
                    eventsout=evts_src_name,
                    regionfile=f"{self.workdir}/products/src_{instr}.reg",
                    imgfile='NONE',fitsbinlc='NONE',
                    binlc=500.0,
                    phafile='NONE',fitsbinpha='NONE',
                    timefile='NONE',xcolf='X', xcolh='X',ycolf='Y', ycolh='Y',
                    polwcol='NONE',stokes='NONE',tcol='TIME', ecol='PI'
                    );
                
                if glob.glob(evts_bkg_name) == []:
                    hsp.extractor(
                    filename=f"{self.workdir}/nu{self.ID}A01_cl.evt", 
                    eventsout=evts_bkg_name,
                    regionfile=f"{self.workdir}/products/bkg_{instr}.reg",
                    imgfile='NONE',fitsbinlc='NONE',
                    binlc=500.0,
                    phafile='NONE',fitsbinpha='NONE',
                    timefile='NONE',xcolf='X', xcolh='X',ycolf='Y', ycolh='Y',
                    polwcol='NONE',stokes='NONE',tcol='TIME', ecol='PI'
                    );
            print(f'\t\t<  INFO  > : Extracting source and background spectra')   
            if glob.glob(spec_src_name) == [] or glob.glob(spec_bkg_name) == []:
                os.system(f"""nuproducts indir='{self.workdir}' outdir='{self.workdir}/products' \
                    srcregionfile='{self.workdir}/products/src_{instr}.reg' \
                    bkgregionfile='{self.workdir}/products/bkg_{instr}.reg' \
                    instrument='{instr}' steminputs='nu{self.ID}' pilow='{int(energy2nustarchannel(low))}' \
                    pihigh='{int(energy2nustarchannel(up))}' \
                    lcfile=None bkglcfile=None imagefile=None rungrppha=no \
                    runmkarf=no runmkrmf=no \
                    bkgphafile='{spec_bkg_name}' \
                    phafile='{spec_src_name}' 2>&1  | tee '{self.logdir}/nuproducts_spectra_{instr}.txt'""")

            else:
                print(f'\t\t<  INFO  > : Source and background events already extracted')

    def gen_lightcurves_manual(self, src_name,energies=[[3.0,10.],[10.0,79.0]],input_timebin_size=500):
        """
        Generate light curves for source and background for all energy bands
        Generate background subtracted light-curve
        """

        print('<  INFO  > : Generating light-curves'    )
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            
            evts_src_name = f"{self.workdir}/products/{src_name}_evts_src_{instr}.fits"
            evts_bkg_name = f"{self.workdir}/products/{src_name}_evts_bkg_{instr}.fits"
            spec_src_name = f"{self.workdir}/products/{src_name}_spec_src_{instr}.fits"
            spec_bkg_name = f"{self.workdir}/products/{src_name}_spec_bkg_{instr}.fits"
                
            src_backscal = fits.open(spec_src_name)["SPECTRUM"].header["BACKSCAL"]
            bkg_backscal = fits.open(spec_bkg_name)["SPECTRUM"].header["BACKSCAL"]
            
            scale = src_backscal/bkg_backscal
            

            for en in energies:
                print(f"\t<  INFO  > : Processing energy range : {en[0]}-{en[1]} keV")

                PI_min = np.rint(energy2nustarchannel(en[0])).astype(int)
                PI_max = np.rint(energy2nustarchannel(en[1])).astype(int)
                print(f"\t\t<  INFO  > : PI_min : {PI_min} PI_max : {PI_max}")
            
                t, net, err, bg, bg_err, t_bin, clean_Frac_EXP, T0 = get_lightcurve(
                    evts_src_name,
                    evts_bkg_name,
                    scale,
                    user_defined_bti=None,
                    verbose=True,
                    min_Frac_EXP=0.3,
                    CCDNR='',
                    input_timebin_size=input_timebin_size,
                    PATTERN=4,
                    PI=[PI_min, PI_max],
                    t_clip_start=10,
                    t_clip_end=100,
                    suffix="",
                    is_nustar=True)

                arr = np.array([t, net, err, bg, bg_err], dtype=float).T
                frac = np.array([t_bin[:-1], clean_Frac_EXP], dtype=float).T
                print(en)
                np.savetxt(
                    f"{src_name}{instr}_{self.ID}_lc_{en[0]}-{en[1]}.txt",
                    arr,
                    header=f"T0 (TT): {T0}" + "\n T" + "\ntime net err bg bg_err",
                )
                np.savetxt(
                    f"{src_name}{instr}_{self.ID}_{en[0]}-{en[1]}_frac.txt",
                    frac,
                    header=f"T0: {T0}" + "\ntime clean_Frac_EXP",
                )

                fig, ax = plt.subplots(figsize=(15, 5))
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Rate (cts/s)")
                ax.errorbar(
                    t, net, yerr=err, color="k", label="Net", fmt="o", ms=2, mfc="w"
                )
                ax.errorbar(
                    t,
                    bg,
                    yerr=bg_err,
                    color="r",
                    label="Background",
                    fmt="o",
                    ms=2,
                    mfc="w",
                )
                # ax[1].errorbar(t,bg,yerr=bg_err,color='r',label='Background',fmt='o',ms=3,mfc='w')
                ax.legend()
                fig.savefig(
                    f"lightcurve_{instr}_{en[0]}-{en[1]}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

    def gen_lightcurves(self, src_name,bintime=500,correctlc=False):
        """
        Generate light curves for source and background for all energy bands
        Generate background subtracted light-curve
        """
        tag = f"_{bintime}"

        print('<  INFO  > : Generating light-curves')
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            for energy_range in self.energybands:
                low, up = energy_range
                lc_src_name = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_lc{tag}_src_{low}-{up}.fits"
                lc_bkg_name = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_lc{tag}_bkg_{low}-{up}.fits"

                if glob.glob(lc_src_name) == [] or glob.glob(lc_bkg_name) == []:
                    print(
                        f'\t\t<  INFO  > : Generate src and bkg light curves {low}-{up} keV')


            if len(glob.glob(f"{self.workdir}/products/*{src_name}_{instr}*lc_*")) != len(self.energybands):
                for energy_range in self.energybands:
                    low, up = energy_range
                    lc_src_name = f"{self.workdir}/{self.ID}_{src_name}_{instr}_lc{tag}_src_{low}-{up}.fits"
                    lc_bkg_name = f"{self.workdir}/{self.ID}_{src_name}_{instr}_lc{tag}_bkg_{low}-{up}.fits"

                    if glob.glob(lc_src_name) == [] or glob.glob(lc_bkg_name) == []:
                        print(
                            f'\t\t<  INFO  > : Generate src and bkg light curves {low}-{up} keV')

                        os.system(f"""nuproducts indir='{self.workdir}' outdir='{self.workdir}/products' \
                            srcregionfile='{self.workdir}/products/src_{instr}.reg' \
                            bkgregionfile='{self.workdir}/products/bkg_{instr}.reg' \
                            correctlc='{"yes" if correctlc else "no"}' \
                            instrument='{instr}' steminputs='nu{self.ID}' pilow='{int(energy2nustarchannel(low))}'  \
                            pihigh='{int(energy2nustarchannel(up))}'  binsize={bintime}   \
                            lcfile='{lc_src_name}' bkglcfile='{lc_bkg_name}'  \
                            imagefile=None runmkarf=no runmkrmf=no bkgphafile=NONE phafile=NONE 2>&1  | tee '{self.logdir}/nuproducts_{instr}_{low}_{up}.txt' """)
        self.plot_light_curves(src_name,tag)

    def plot_light_curves(self, src_name,tag):
        """
        Plot light_curves

        """

        fig, ax = plt.subplots(2, 1, figsize=(8, 9))
        workdir = f"{self.workdir}/products/"
        for j, (axis, energies) in enumerate(zip(ax, self.energybands)):
            if j == 0:
                axis.set_title(self.date)
            low, up = energies
            for instr in self.instruments:
                lc_src_name = f"{workdir}/{self.ID}_{src_name}_{instr}_lc{tag}_src_{low}-{up}.fits"
                lc_bkg_name = f"{workdir}/{self.ID}_{src_name}_{instr}_lc{tag}_bkg_{low}-{up}.fits"

                hdu_list_src = fits.open(lc_src_name)
                hdu_list_bkg = fits.open(lc_bkg_name)
                lc_data_src = hdu_list_src[1].data
                lc_data_bkg = hdu_list_bkg[1].data
                lc_data_clean = lc_data_src["RATE"]-lc_data_bkg["RATE"]
                lc_data_clean_error = np.sqrt(
                    lc_data_src["ERROR"]**2+lc_data_bkg["ERROR"]**2)
                if instr == "FPMA":
                    col = "#ffa600"
                else:
                    col = "#f4413e"
                axis.errorbar(lc_data_src["TIME"]/1000, lc_data_clean,
                              yerr=lc_data_clean_error, fmt="o", markersize=5., color=col, label=instr)
            label = f"{int(low)}-{int(up)}"
            axis.set_ylabel(f"{label} keV" + "\n"+"Rate "+r"$\mathrm{(count}~\mathrm{s}^{-1})$")
            axis.grid()
            ax[0].set_xticklabels([])
            axis.set_xlabel("Time (ks)")

        ax[-1].legend()
        # fig.subplots_adjust(bottom=0.15,top=0.8,left=0.05,right=0.9,hspace=0.6,wspace=-0.9)
        #fig.suptitle("Light curves NGC 7582",fontsize=30)
        fig.tight_layout()
        # fig.subplots_adjust(wspace=0.12,hspace=0.1,bottom=0.2)
        fig.savefig(f"{self.plotdir}/{self.ID}_{src_name}lightcurves.pdf")

    def gen_spectra(self, src_name, **kwargs):
        """
        Generate the spectral files
        """

        print(f'<  INFO  > : Generating spectra')
        if self.nslices > 1:
            assert ("start_nustar" in kwargs), "No start time in NuSTAR gen_spectra file ! "
            print(f'<  INFO  > : Generating GTI FILES')
            start_nustar = kwargs.get("start_nustar")
            self.create_nustar_gti(start_nustar)

        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')

            partnumber = 0
            for interval in self.slices:
                if interval == "all":
                    timeslice = ""
                    portion = 'all'
                    expression_gti = ""
                else:
                    partnumber += 1
                    portion = f"p{partnumber}"
                    expression_gti = f" usrgtifile='{self.workdir}/{self.ID}_{instr}_gti_{portion}.fits' "
                print(f'\t\t<  INFO  > : Processing portion : {portion}')

                low, up = self.energy_range

                src_spectrum = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_spectrum_src_{low}-{up}_{portion}.fits"
                bkg_spectrum = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_spectrum_bkg_{low}-{up}_{portion}.fits"
                output_rmf = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_{low}-{up}_{portion}.rmf"
                output_arf = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_{low}-{up}_{portion}.arf"
                grouped_spectrum = f"{self.workdir}/products/{self.ID}_{src_name}{instr}_grouped_spectrum_{low}-{up}_{portion}.fits"

                if glob.glob(grouped_spectrum) == []:
                    os.system(f"""nuproducts indir='{self.workdir}' outdir='{self.workdir}/products' \
                        srcregionfile='{self.workdir}/products/src_{instr}.reg' \
                        bkgregionfile='{self.workdir}/products/bkg_{instr}.reg' \
                        instrument='{instr}' steminputs='nu{self.ID}' pilow='{int(energy2nustarchannel(low))}' \
                        pihigh='{int(energy2nustarchannel(up))}' \
                        lcfile=None bkglcfile=None imagefile=None rungrppha=yes \
                        runmkarf=yes runmkrmf=yes grpmincounts=25 \
                        outrmffile='{output_rmf}' \
                        outarffile='{output_arf}' \
                        grppibadlow='{int(energy2nustarchannel(low))}' grppibadhigh='{int(energy2nustarchannel(up))}' \
                        bkgphafile='{bkg_spectrum}' \
                        phafile='{src_spectrum}' \
                        grpphafile='{grouped_spectrum}' {expression_gti} 2>&1  | tee '{self.logdir}/nuproducts_spectra_{instr}_{portion}.txt'""")

    def create_nustar_gti(self, start_nustar):
        """
        Create GTI for NuSTAR

        """

        for instr in self.instruments:
            instrument = instr[-1]

            hdu_gti = fits.open(
                f"{self.workdir}/nu{self.ID}{instrument}01_gti.fits")
            ct = 0
            gti_slices = [0]
            Data = []
            Lim_list = []
            last_iter_cut = False
            last_iter_split = False
            for trunc in range(len(self.slices)-1):
                lim = self.slices[trunc][1]*1e3+start_nustar
                Lim_list.append(lim)
                ct = 0
                for i in range(hdu_gti[1].header["NAXIS2"]):
                    if lim > hdu_gti[1].data["START"][i] and lim > hdu_gti[1].data["STOP"][i]:
                        if lim < hdu_gti[1].data["START"][i+1]:
                            gti_slices.append(i)
                            gti_slices.sort()
                            Data.append(
                                np.copy(hdu_gti[1].data[gti_slices[trunc]:1+gti_slices[trunc+1]]))
                            if last_iter_cut == True:
                                Data[trunc] = Data[trunc][1:]
                            if last_iter_split == True:
                                Data[trunc][0]["START"] = self.slices[trunc -
                                                                               1][1]*1e3+start_nustar
                                last_iter_split = False
                            last_iter_cut = True
                            break
                        else:
                            ct += 1
                    elif lim > hdu_gti[1].data["START"][i] and lim < hdu_gti[1].data["STOP"][i]:
                        last_iter_split == True
                        gti_slices.append(i)
                        gti_slices.sort()
                        Data.append(
                            np.copy(hdu_gti[1].data[gti_slices[trunc]:1+gti_slices[trunc+1]]))
                        if last_iter_split == True:
                            Data[trunc][0]["START"] = self.slices[trunc -
                                                                  1][1]*1e3+start_nustar
                        elif last_iter_cut == True:
                            Data[trunc] = Data[trunc][1:]
                            last_iter_cut = False
                        Data[trunc][-1]["STOP"] = lim
                        last_iter_split = True
                        break

            gti_slices.append(hdu_gti[1].header["NAXIS2"]-1)
            Data.append(np.copy(hdu_gti[1].data[gti_slices[trunc+1]:]))

            if last_iter_split == True:
                Data[-1][0]["START"] = self.slices[trunc][1]*1e3+start_nustar

            for i in range(len(self.slices)):
                hdu = fits.PrimaryHDU()
                hdul = fits.HDUList([hdu])
                stdgti = fits.BinTableHDU(Data[i])
                stdgti.name = "STDGTI"
                hdul.append(stdgti)
                if os.path.isfile(f"{self.workdir}/{self.ID}_{instr}_gti_p{i+1}.fits"):
                    os.remove(
                        f"{self.workdir}/{self.ID}_{instr}_gti_p{i+1}.fits")
                hdul.writeto(
                    f"{self.workdir}/{self.ID}_{instr}_gti_p{i+1}.fits")
