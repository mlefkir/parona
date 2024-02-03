import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from astropy.wcs import WCS
from regions import Regions
from matplotlib.colors import LogNorm
from pysas.wrapper import Wrapper as w
from astroquery.esa.xmm_newton import XMMNewton
from astropy.io import fits
from astropy.table import Table
import contextlib
import shutil
from .callds9 import start_ds9
from .xmm_lc import get_lightcurve


class ObservationXMM:
    """
    Processing an XMM-Newton observation

    The directories created are:

    path/odf
    path/data/obsid_datadir
    path/obsid_workdir
    path/obsid_workdir/logs
    path/obsid_workdir/plots


    Parameters
    ----------
    path : str
        Path to the folder where the repertories will be created
    obsidentifier : str
        Observation ID
    slices : list, optional
        List of slices to process, by default ['all'], in case we want reduce
        spectra for specific segments of the observation
    nslices : int, optional
        Number of slices to process, by default 0, in case we want reduce
        spectra for specific segments of the observation
    instruments : list, optional
        List of instruments to process, by default ["EPN", "EMOS1", "EMOS2"]
    energybands : list, optional
        List of energy bands to process, by default [[200, 3000], [3000, 12000]]
        2keV-3keV and 3keV-12keV
    replot : bool, optional
        Replot the light curves, by default True
    

    """
    
    def __init__(
        self,
        path,
        obsidentifier,
        slices=["all"],
        instruments=["EPN", "EMOS1", "EMOS2"],
        energybands=[[200, 3000], [3000, 12000]],replot=True,
    ):
        """

        Parameters
        ----------
        path : str
            Path to the folder where the repertories will be created
        obsidentifier : str
            Observation ID
        slices : list, optional
            List of slices to process, by default ['all'], in case we want reduce
            spectra for specific segments of the observation
        nslices : int, optional
            Number of slices to process, by default 0, in case we want reduce
            spectra for specific segments of the observation
        instruments : list, optional
            List of instruments to process, by default ["EPN", "EMOS1", "EMOS2"]
        energybands : list, optional
            List of energy bands to process, by default [[200, 3000], [3000, 12000]]
            2keV-3keV and 3keV-12keV
        """

        self.ID = obsidentifier
        print(f"<  INFO  > : Observation ID: {self.ID}")
        print(
            f"<  INFO  > : Retrieving observation information from the XSA TAP service"
        )
        query = XMMNewton.query_xsa_tap(
            f"select * from v_public_observations where observation_id='{obsidentifier}' "
        )
        query2 = XMMNewton.query_xsa_tap(
            f"select * from v_exposure where observation_id='{obsidentifier}' "
        )

        # find the observation date
        self.date = query["start_utc"][0][:10]
        self.slices = slices
        if self.slices == ["all"]:
            self.nslices = 0
        else:
            self.nslices = len(self.slices)

        self.instruments = instruments
        if isinstance(self.instruments, str):
            self.instruments = [self.instruments]
        self.instruments_mode = []
        filt_sci = query2["is_scientific"]

        for instr in self.instruments:
            self.instruments_mode.append(
                query2["mode_friendly_name"][filt_sci][query2["instrument"][filt_sci] == instr][0])
        self.modes = dict(zip(self.instruments, self.instruments_mode))
        
        print(f"<  INFO  > : Scientific modes: {self.modes}")
        
        self.obs_files = {}
        self.regions = {}

        for instr in self.instruments:
            self.obs_files[instr] = {}
            self.regions[instr] = {}

        self.energybands = energybands
        self.energy_range = [
            np.min(np.array(self.energybands).flatten()),
            np.max(np.array(self.energybands).flatten()),
        ]
        print(f"<  INFO  > : Energy range: {self.energy_range}")
        self.check_repertories(path)

        self.replot = replot
        # execute startsas which is is a wrapper for the SAS command line tools
        # cifbuild and odfingest, will set SAS_CCF and SAS_ODF environment variables
        # if already set then execute again but with different arguments
        if not glob.glob(f"{self.workdir}/ccf.cif") or not glob.glob(
            f"{self.workdir}/*SUM.SAS"
        ):
            print("<  INFO  > : Executing startsas for the first time")
            inargs = [f"odfid={self.ID}", f"workdir={self.workdir}"]
            with open(f"{self.logdir}/startsas.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("startsas", inargs).run()
        else:
            print("<  INFO  > : Executing startsas again")
            inargs = [
                f'sas_ccf={glob.glob(f"{self.workdir}/ccf.cif")[0]}',
                f'sas_odf={glob.glob(f"{self.workdir}/*SUM.SAS")[0]}',
                f"workdir={self.workdir}",
            ]
            with open(f"{self.logdir}/startsas_bis.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("startsas", inargs).run()

        os.chdir(self.workdir)

    def check_repertories(self, path):
        """Check the repertories and create them if they don't exist

        Following the folder tree described in the class docstring:

            path/odf
            path/data/obsid_datadir
            path/obsid_workdir
            path/obsid_workdir/logs
            path/obsid_workdir/plots

        Parameters
        ----------
        path : str
            Path to the folder where the repertories will be created
        """

        if not os.path.isdir(f"{path}/{self.ID}_workdir"):
            os.makedirs(f"{path}/{self.ID}_workdir")
        if not os.path.isdir(f"{path}/{self.ID}_workdir/logs"):
            os.makedirs(f"{path}/{self.ID}_workdir/logs")
        if not os.path.isdir(f"{path}/{self.ID}_workdir/plots"):
            os.makedirs(f"{path}/{self.ID}_workdir/plots")

        self.workdir = f"{path}/{self.ID}_workdir/"
        self.logdir = f"{path}/{self.ID}_workdir/logs/"
        self.plotdir = f"{path}/{self.ID}_workdir/plots/"
        os.environ["SAS_WORKDIR"] = self.workdir

    def gen_evts(self, with_MOS=True, with_RGS=False, with_OM=False):
        """
        Generate the event list the EPIC pn and MOS CCD

        will run epproc by default and emproc if the event list is not already present in the workdir

        Parameters
        ----------
        with_MOS : bool, optional
            If True, will run emproc to generate the event list for the EPIC MOS CCDs, by default True
        with_RGS : bool, optional
            If True, will run rgsproc to generate the event list for the RGS, by default False
        with_OM : bool, optional

        """

        os.chdir(self.workdir)
        if (
            glob.glob(f"{self.workdir}/*EPN*ImagingEvts*") == []
            and "EPN" in self.instruments
        ):
            print(
                f"<  INFO  > : Running epproc to generate events list for the EPIC-pn"
            )
            with open(f"{self.logdir}/epproc.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("epproc", []).run()
        if with_MOS and ("EMOS1" in self.instruments or "EMOS2" in self.instruments ):
            if glob.glob(f"{self.workdir}/*MOS*ImagingEvts*") == [] and (
                "EMOS1" in self.instruments or "EMOS2" in self.instruments
            ):
                print(
                    f"<  INFO  > : Running emproc to generate events list for the EPIC-MOS 1 & 2"
                )
                with open(f"{self.logdir}/emproc.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("emproc", []).run()
        if with_RGS and ("R1" in self.instruments or "R2" in self.instruments):
            with open(f"{self.logdir}/rgsproc.log", "w+") as f:
                print(
                    f"<  INFO  > : Running rgsproc to generate events list for the RGS 1 & 2"
                )
                with contextlib.redirect_stdout(f):
                    w("rgsproc", []).run()

        if with_OM and ("OM" in self.instruments):
            with open(f"{self.logdir}/omichain.log", "w+") as f:
                print(
                    f"<  INFO  > : Running omichain to generate events list for the OM"
                )
                with contextlib.redirect_stdout(f):
                    w("omichain", []).run()

        for instr in self.instruments:
            self.obs_files[instr]["evts"] = self.find_eventfile(instr)
            print(
                f'<  INFO  > : The raw event list selected for {instr} is: {self.obs_files[instr]["evts"].replace(self.workdir,"")}'
            )

    def find_RGS_eventfiles(self):
        """
        Return the raw event list file

        If there is only one event list file in the workdir, it will return it.
        Else, it will return the biggest file in the workdir.
        """
        for instr in ["R1", "R2"]:
            self.obs_files[instr] = {}
            buff = glob.glob(f"{self.workdir}/*{instr}*EVENLI*")
            size = 0
            if len(buff) == 1:
                input_eventfile = buff[0]
            else:
                for i, res in enumerate(buff):
                    if os.path.getsize(res) > size:
                        size = os.path.getsize(res)
                        input_eventfile = buff[i]
                        break
            self.obs_files[instr]["evts"] = input_eventfile

    def find_eventfile(self, instr):
        """
        Return the raw event list file

        If there is only one event list file in the workdir, it will return it.
        Else, it will return the biggest file in the workdir.
        
        Parameters
        ----------
        instr : str
            Instrument name, e.g. "EPN", "EMOS1", "EMOS2"
        """
        buff = glob.glob(f"{self.workdir}/*{instr}_*ImagingEvts*")
        size = 0
        if len(buff) == 1:
            input_eventfile = buff[0]
        else:
            for i, res in enumerate(buff):
                if os.path.getsize(res) > size:
                    size = os.path.getsize(res)
                    input_eventfile = buff[i]

        return input_eventfile

    def gen_flarelc(self):
        """

        Generate the flare background rate file

        """
        print("<  INFO  > : Generating flares light curves")

        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr][
                "flare"
            ] = f"{self.workdir}/{self.ID}_{instr}_FlareBKGRate.fits"

            if glob.glob(self.obs_files[instr]["flare"]) == []:
                if "PN" in instr:
                    expression = "#XMMEA_EP&&(PI>10000.&&PI<12000.)&&(PATTERN==0.)"
                else:
                    expression = "#XMMEA_EM&&(PI>10000)&&(PATTERN==0.)"

                inargs = [
                    f'table={self.obs_files[instr]["evts"]}',
                    "withrateset=Y",
                    f'rateset={self.obs_files[instr]["flare"]}',
                    "maketimecolumn=Y",
                    "timebinsize=100",
                    "makeratecolumn=Y",
                    f"expression={expression}",
                ]

                with open(f"{self.logdir}/{instr}_flares.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
            # -- Plot the light curve for the flares --
            if (
                glob.glob(f"{self.plotdir}/{self.ID}_{instr}_FlareBKGRate.pdf") == []
                or self.replot == True
            ):
                self.plot_flares_lc(instr)

    def plot_flares_lc(self, instr):
        """

        Plot the light curve of the flares

        """

        hdu_list = fits.open(self.obs_files[instr]["flare"], memmap=True)
        lc_data = Table(hdu_list[1].data)
        lc_data["TIME"] -= lc_data["TIME"][0]
        fig, axis = plt.subplots(1, 1, figsize=(8, 6))
        axis.plot(lc_data["TIME"] / 1e3, lc_data["RATE"])
        if "PN" in instr:
            axis.hlines(
                0.4,
                lc_data["TIME"][0] / 1e3,
                lc_data["TIME"][-1] / 1e3,
                color="red",
                label="0.4 cts/s",
            )
        else:
            axis.hlines(
                0.35,
                lc_data["TIME"][0] / 1e3,
                lc_data["TIME"][-1] / 1e3,
                color="red",
                label="0.35 cts/s",
            )
        axis.legend()
        axis.set_xlabel("Time (ks)")
        axis.set_ylabel("count rate (cts/s)")

        fig.suptitle(f"{self.ID} {instr}\n flares background light-curve")
        fig.tight_layout()
        fig.savefig(f"{self.plotdir}/{self.ID}_{instr}_FlareBKGRate.pdf")

    def gen_gti(self):
        """

        Generate the GTI good time intervals

        """
        print(f"<  INFO  > : Generating the GTI")
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr]["gti"] = f"{self.workdir}/{self.ID}_{instr}_GTI.fits"
            if glob.glob(self.obs_files[instr]["gti"]) == []:
                if "PN" in instr:
                    expression = "RATE<=0.4"
                else:
                    expression = "RATE<=0.35"

                inargs = [
                    f'table={self.obs_files[instr]["flare"]}',
                    f'gtiset={self.obs_files[instr]["gti"]}',
                    f"expression={expression}",
                ]
                print(f"\t<  INFO  > : Running tabgtigen")
                with open(f"{self.logdir}/{instr}_tabgtigen.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("tabgtigen", inargs).run()

    def gen_clean_evts(self):
        """

        Generating cleaned event list

        """
        print(f"<  INFO  > : Generating clean event lists")
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr][
                "clean_evts"
            ] = f"{self.workdir}/{self.ID}_{instr}_evts_clean.fits"

            if glob.glob(self.obs_files[instr]["clean_evts"]) == []:
                if "PN" in instr:
                    expression = f'#XMMEA_EP && gti( {self.obs_files[instr]["gti"]} , TIME ) && (PI >150)'
                else:
                    expression = f'#XMMEA_EM && gti( {self.obs_files[instr]["gti"]} , TIME ) && (PI >150)'

                inargs = [
                    f'table={self.obs_files[instr]["evts"]}',
                    "withfilteredset=Y",
                    f'filteredset={self.obs_files[instr]["clean_evts"]}',
                    "destruct=Y",
                    "keepfilteroutput=T",
                    f"expression={expression}",
                ]

                print(f"<  INFO  > : Filtering flares to produce events list")
                with open(f"{self.logdir}/{instr}_filtering_flares.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()

    def gen_images(self):
        """

        Generate images from the cleaned event list

        """
        print(f"<  INFO  > : Generating images")
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr][
                "image"
            ] = f"{self.workdir}/{self.ID}_{instr}_image.fits"
            if glob.glob(self.obs_files[instr]["image"]) == []:
                inargs = [
                    f'table={self.obs_files[instr]["evts"]}',
                    "imagebinning=binSize",
                    f'imageset={self.obs_files[instr]["image"]}',
                    "withimageset=yes",
                    "xcolumn=X",
                    "ycolumn=Y",
                    "ximagebinsize=80",
                    "yimagebinsize=80",
                ]

                print(f"\t<  INFO  > : Generate an image")
                with open(f"{self.logdir}/{instr}_image.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()

    def plot_image_region(self,hdu, region_file, instruments_mode ):
        """
        Plot an image with regions overlaid.
        """
        fig,ax = plt.subplots(figsize=(9,8))#,subplot_kw={'projection':wcs})


        wcs = WCS(hdu.header)
        M = hdu.data

        iy = np.all(M==0,axis=1)
        ix = np.all(M==0,axis=0)
        ysplit = np.split(iy, np.where(iy == 0.0)[0])
        xsplit = np.split(ix, np.where(ix == 0.0)[0])

        ystart = ysplit[0].sum()
        xstart = xsplit[0].sum()
        ystop = len(iy)-ysplit[-1].sum()
        xstop = len(ix)-xsplit[-1].sum()
        
        
        im = ax.imshow(M, origin='lower', cmap='cividis', norm=LogNorm(vmin=5))
        fig.colorbar(im, ax=ax)
        myreg = Regions.read(region_file, format='ds9')
        for reg in myreg:
            reg.to_pixel(wcs).plot(ax=ax)
        
        # if instruments_mode == 'Small Window':
        ax.update({'xlim':(xstart,xstop),'ylim':(ystart,ystop)})
        fig.tight_layout()
        
        return fig, ax
    
    def select_regions(self, src_name, show_regions=False):
        """

        Select the source and background regions with ds9

        """
        print(f"<  INFO  > : Selection of source and background regions with ds9")

        for i,instr in enumerate(self.instruments):
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr][
                "positions"
            ] = f"{self.workdir}/{self.ID}_{src_name}{instr}_positions.txt"
            
            if glob.glob(self.obs_files[instr]["positions"]) == []:                
                if i == 0:
                    try:
                        python_ds9.set("regions select all")
                    except:
                        python_ds9 = start_ds9()

                python_ds9.set("regions select all")
                python_ds9.set("regions delete")
                python_ds9.set("file " + self.obs_files[instr]["evts"])
                python_ds9.set("bin to fit")
                python_ds9.set("scale log")
                python_ds9.set("cmap b")
                print("Draw FIRST the region for the source and THEN the background")
                input("Press Enter to continue...")
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")
                self.regions[instr]["src"] = python_ds9.get("regions").split("\n")[0]
                self.regions[instr]["bkg"] = python_ds9.get("regions").split("\n")[1]
                python_ds9.set("regions format ds9")
                python_ds9.set("regions system wcs")
                python_ds9.set("regions select all")
                python_ds9.set(f"regions save {self.workdir}/plot_{instr}.reg")
                
                np.savetxt(
                    self.obs_files[instr]["positions"],
                    np.array([self.regions[instr]["src"], self.regions[instr]["bkg"]]),
                    fmt="%s",
                )
                # doesn't work anymore, I don't know why...
                #python_ds9.set(f"saveimage png {self.plotdir}/{self.ID}_{src_name}{instr}_image.png")


            else:
                print(f"\t<  INFO  > : Reading regions from {self.obs_files[instr]['positions']}")
                self.regions[instr]["src"], self.regions[instr]["bkg"] = np.genfromtxt(
                    self.obs_files[instr]["positions"], dtype="str"
                )
            if show_regions or not os.path.isfile(f"{self.workdir}/plot_{instr}.reg") :
                if i == 0:
                    try:
                        python_ds9.set("regions select all")
                    except:
                        python_ds9 = start_ds9()

                python_ds9.set("regions select all")
                python_ds9.set("regions delete")
                python_ds9.set("file " + self.obs_files[instr]["evts"])
                python_ds9.set("bin to fit")
                python_ds9.set("scale log")
                python_ds9.set("cmap b")
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")
                python_ds9.set(f'regions load {self.obs_files[instr]["positions"]}')
                python_ds9.set("regions format ds9")
                python_ds9.set("regions system wcs")
                python_ds9.set("regions select all")
                python_ds9.set(f"regions save {self.workdir}/plot_{instr}.reg")
                
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")
                python_ds9.set("zoom to fit")
            hdu = fits.open(self.obs_files[instr]["image"])[0]
            fig,ax = self.plot_image_region(hdu, f"{self.workdir}/plot_{instr}.reg", self.modes[instr])
            fig.savefig(f"{self.plotdir}/{self.ID}_{src_name}{instr}_image.png")
            plt.close(fig)
                # python_ds9.set(f"saveimage png {self.plotdir}/{self.ID}_{src_name}{instr}_image.png")



    def check_pileup(self, src_name):
        """

        Check pileup with epatplot

        """
        print("<  INFO  > : Checking pile-up during observation")
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            self.obs_files[instr]["epatplot"] = f"{self.ID}_{src_name}{instr}_pat.ps"
            self.obs_files[instr][
                "clean_filt"
            ] = f"{self.workdir}/{self.ID}_{instr}_clean_filtered.fits"

            if len(glob.glob(f"{self.plotdir}/*{src_name}*pat.ps")) != 3:
                inargs = [
                    f'table={self.obs_files[instr]["evts"]}',
                    "withfilteredset=yes",
                    f'filteredset={self.obs_files[instr]["clean_filt"]}',
                    "keepfilteroutput=yes",
                    f'expression=((X,Y) in  {self.regions[instr]["src"]})'# && gti({self.obs_files[instr]["gti"]} ,TIME)',
                ]

                print(f"\t<  INFO  > : Generate an event list for pile-up")
                with open(f"{self.logdir}/{src_name}{instr}_pileup.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
                inargs = [
                    f'set={self.obs_files[instr]["clean_filt"]}',
                    f'plotfile={self.obs_files[instr]["epatplot"]}',
                ]
                print(f"\t<  INFO  > : Running epatplot to evaluate pile-up")
                with open(f"{self.logdir}/{src_name}{instr}_epatplot.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("epatplot", inargs).run()
                shutil.move(
                    self.obs_files[instr]["epatplot"],
                    f"{self.plotdir}/{self.obs_files[instr]['epatplot']}",
                )

    def gen_lightcurves(
        self, src_name, binning, absolute_corrections=False, write_bin_tag=False
    ):
        """

        Generate light curves for source and background for all energy bands
        Generate background subtracted light-curve

        https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/epicmode.html

        Parameters
        ----------
        src_name: str
            Name of the source
        binning: list or float
            Binning for each instrument,
        absolute_corrections: bool, optional
            Apply absolute corrections to the light-curves, by default False
        write_bin_tag: bool, optional
            Write the binning tag in the light-curve file name, by default False

        """
        if isinstance(binning, list) and len(binning) != len(self.instruments):
            raise ValueError(
                f"Length of binning list must be equal to the number of instruments ({len(self.instruments)})"
            )
        if isinstance(binning, float) or isinstance(binning, int):
            binning = [binning] * len(self.instruments)

        if write_bin_tag:
            tags = [f"_bin{b}" for b in binning]

        binsize = dict(zip(self.instruments, binning))

        print("<  INFO  > : Generating light-curves")

        # instru
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            if write_bin_tag:
                tag = tags[self.instruments.index(instr)]
            else:
                tag = ""
            for name_tag in ["src", "bkg"]:
                for energy_range in self.energybands:
                    low, up = energy_range

                    self.obs_files[instr][
                        f"clean_{name_tag}_{low/1000}_{up/1000}"
                    ] = f"{self.workdir}/{self.ID}_{instr}_clean_{name_tag}_{low/1000}_{up/1000}_evts_clean.fits"
                    lc_name = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_{name_tag}_{low/1000}-{up/1000}.fits"
                    self.obs_files[instr][
                        f"{src_name}_lc_src_{low/1000}_{up/1000}"
                    ] = lc_name

                    if glob.glob(lc_name) == []:
                        if "PN" in instr:
                            expression = f"#XMMEA_EP && (PATTERN<=4) && (PI in [{low}:{up}]) && ((X,Y) IN {self.regions[instr][name_tag]})"
                        else:
                            expression = f"#XMMEA_EM && (PATTERN<=12) && (PI in [{low}:{up}]) && ((X,Y) IN {self.regions[instr][name_tag]})"

                        inargs = [
                            f'table={self.obs_files[instr]["clean_evts"]}',
                            "withfilteredset=yes",
                            f'filteredset={self.obs_files[instr][f"clean_{name_tag}_{low/1000}_{up/1000}"]}',
                            "keepfilteroutput=yes",
                            f"expression={expression}",
                        ]

                        print(
                            f"\t\t<  INFO  > : Generate {name_tag} event list {low/1000}-{up/1000} keV"
                        )
                        with open(
                            f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_lc{tag}_light_curve.log",
                            "w+",
                        ) as f:
                            with contextlib.redirect_stdout(f):
                                w("evselect", inargs).run()

                        inargs = [
                            f'table={self.obs_files[instr]["clean_evts"]}',
                            "withrateset=yes",
                            f"rateset={lc_name}",
                            f"timebinsize={binsize[instr]}",
                            "maketimecolumn=yes",
                            "makeratecolumn=yes",
                            f"expression={expression}",
                        ]

                        print(
                            f"\t\t<  INFO  > : Generate {name_tag} light curves {low/1000}-{up/1000} keV"
                        )
                        with open(
                            f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_lc{tag}_light_curve.log",
                            "w+",
                        ) as f:
                            with contextlib.redirect_stdout(f):
                                w("evselect", inargs).run()
            # ---light curves corrected from background
            for energy_range in self.energybands:
                low, up = energy_range
                lc_clean = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_clean_{low/1000}-{up/1000}.fits"
                if glob.glob(lc_clean) == []:
                    lc_src = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_src_{low/1000}-{up/1000}.fits"
                    lc_bkg = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_bkg_{low/1000}-{up/1000}.fits"
                    inargs = [
                        f"srctslist={lc_src}",
                        f'eventlist={self.obs_files[instr]["clean_evts"]}',
                        f"outset={lc_clean}",
                        f"bkgtslist={lc_bkg}",
                        "withbkgset=yes",
                        f"applyabsolutecorrections={absolute_corrections}",
                    ]
                    print(
                        f"\t<  INFO  > : Running epiclccorr to correct light curves {low/1000}-{up/1000} keV"
                    )
                    with open(
                        f"{self.logdir}/{src_name}{instr}_{low/1000}-{up/1000}_{tag}epiclccorr.log",
                        "w+",
                    ) as f:
                        with contextlib.redirect_stdout(f):
                            w("epiclccorr", inargs).run()

        self.plot_lightcurves(src_name, tag)

    def plot_lightcurves(self, src_name, tag):
        list_energies = []
        for energy_range in self.energybands:
            low, up = energy_range
            list_energies.append(f"{low/1000}-{up/1000}")
        fig, ax = plt.subplots(2, 1, figsize=(8, 9))
        for axis, energies in zip(ax, list_energies):
            for instr in self.instruments:
                hdu_list = fits.open(
                    f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_clean_{energies}.fits"
                )
                lc_data = Table(hdu_list[1].data)
                lc_data["TIME"] -= lc_data["TIME"][0]
                axis.errorbar(
                    lc_data["TIME"] / 1000,
                    lc_data["RATE"],
                    yerr=lc_data["ERROR"],
                    fmt="o",
                    markersize=5.0,
                    label=instr,
                )
                axis.legend(loc="upper left", bbox_to_anchor=(1, 0.9))
            axis.set_title(f"{energies} keV")
            axis.set_xlabel("Time (ks)")
            axis.set_ylabel("Rate (cts/s)")
        fig.suptitle(f"{self.ID} - Light curves - {self.date}", fontsize=20)
        fig.tight_layout()
        fig.savefig(f"{self.plotdir}/{self.ID}_{src_name}lightcurves.pdf")
        plt.close(fig)
        
    def find_start_time(self):
        """

        Find the start time for the slices

        """
        incidents_MOS1_CCD = [[6, "2005-03-09"], [3, "2012-12-11"]]
        print("<  INFO  > : Try to find the start time for the slices")

        Start = []
        for counter, instr in enumerate(self.instruments):
            if instr == "EPN":
                if self.instruments_mode[counter] == "Small Window":
                    print("\t<  INFO  > : EPN in Small window mode")
                    nCCD = 1
                    list_CCD = [4]
                elif self.instruments_mode[counter] == "Full Frame":
                    print("\t<  INFO  > : EPN in Full frame mode")
                    nCCD = 12
                    list_CCD = range(1, nCCD + 1)
                else:
                    raise NotImplementedError(
                        f"EPN mode {self.instruments_mode[counter]} not implemented"
                    )
            else:
                nCCD = 7
                list_CCD = range(1, nCCD + 1)
            hdu_list = fits.open(self.obs_files[instr]["clean_evts"])
            L = []
            if instr == "EMOS1":
                if date.fromisoformat(self.date) > date.fromisoformat(
                    incidents_MOS1_CCD[0][1]
                ) and date.fromisoformat(self.date) < date.fromisoformat(
                    incidents_MOS1_CCD[1][1]
                ):
                    list_CCD = [1, 2, 3, 4, 5, 7]
                elif date.fromisoformat(self.date) > date.fromisoformat(
                    incidents_MOS1_CCD[1][1]
                ):
                    list_CCD = [1, 2, 4, 5, 7]
                else:
                    list_CCD = range(1, nCCD + 1)
            for i in list_CCD:
                if i < 10:
                    string = f"0{i}"
                else:
                    string = f"{i}"
                L.append(hdu_list[f"STDGTI{string}"].data["START"][0])
            Start.append(max(L))
        self.start_time = max(Start)
        print("Instrument start : ", self.instruments[np.argmax(Start)])
        return self.start_time

    def get_backscale_value(self, instr, src_name):
        """ Return the backscale value for the source and background regions
        
        Extract the backscale value from the spectrum of the source and background regions
        
        Parameters
        ----------
        instr : str
            Instrument name, e.g. "EPN", "EMOS1", "EMOS2"
        src_name : str
            Name of the source
        
        Returns
        -------
        float
            backscale value for the source region divided by the backscale value for the background region
        """
        backscale = []
        print(f"\t<  INFO  > : Generating spectra to get backscale value for {instr}")
        if instr == "EPN":
            specchanmax = 20479
            pattern = 4
            flag = "(FLAG==0) "
        else:
            specchanmax = 11999
            pattern = 12
            flag = "#XMMEA_EM "

        for name_tag in ["src", "bkg"]:
            # ---generate the spectrum
            output_spectrum = f"{self.workdir}/{self.ID}_{src_name}{instr}_spectrum_{name_tag}_BACKSCALE.fits"
            if instr == "EPN":
                flag = "(FLAG==0) "
            elif instr == "EMOS1" or instr == "EMOS2":
                flag = "#XMMEA_EM "
            expression = f"{flag} && (PATTERN <={pattern}) && ((X,Y) IN {self.regions[instr][name_tag]})"

            inargs = [
                f'table={self.obs_files[instr]["clean_evts"]}',
                "withspectrumset=yes",
                f"spectrumset={output_spectrum}",
                "energycolumn=PI",
                "spectralbinsize=5",
                "withspecranges=yes",
                "specchannelmin=0",
                f"specchannelmax={specchanmax}",
                f"expression={expression}",
            ]

            if glob.glob(output_spectrum) == []:
                print(f"<  INFO  > : Generate {name_tag} spectrum for BACKSCALE")
                with open(
                    f"{self.logdir}/{src_name}{instr}_{name_tag}_spectrum_BACKSCALE.log",
                    "w+",
                ) as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
                # run backscale
                inargs = [f"spectrumset={output_spectrum}"]
                with open(
                    f"{self.logdir}/{src_name}{instr}_{name_tag}_backscale_BACKSCALE.log",
                    "w+",
                ) as f:
                    with contextlib.redirect_stdout(f):
                        w("backscale", inargs).run()
            backscale.append(fits.open(output_spectrum)[1].header["BACKSCAL"])
            print(fits.open(output_spectrum)[1].header["BACKSCAL"])
        return backscale[0] / backscale[1]

    def get_scale_value(self, src_name, src_event_file, bkg_event_file):
        backscale = []
        instr = fits.open(src_event_file)["PRIMARY"].header["INSTRUME"]
        instr_bkg = fits.open(bkg_event_file)["PRIMARY"].header["INSTRUME"]
        assert (
            instr == instr_bkg
        ), "The source and background event files are not from the same instrument"
        print(f"\t<  INFO  > : Generating spectra to get backscale value for {instr}")

        if instr == "EPN":
            specchanmax = 20479
            pattern = 4
            flag = "(FLAG==0) "
        else:
            specchanmax = 11999
            pattern = 12
            flag = "#XMMEA_EM "

        for event_file, name_tag in zip(
            [src_event_file, bkg_event_file], ["src", "bkg"]
        ):
            # ---generate the spectrum
            output_spectrum = f"{self.workdir}/{self.ID}_{src_name}{instr}_spectrum_{name_tag}_BACKSCALE.fits"
            if instr == "EPN":
                flag = "(FLAG==0) "
            elif instr == "EMOS1" or instr == "EMOS2":
                flag = "#XMMEA_EM "
            expression = f"{flag} && (PATTERN <={pattern})"

            inargs = [
                f"table={event_file}",
                "withspectrumset=yes",
                f"spectrumset={output_spectrum}",
                "energycolumn=PI",
                "spectralbinsize=5",
                "withspecranges=yes",
                "specchannelmin=0",
                f"specchannelmax={specchanmax}",
                f"expression={expression}",
            ]

            if glob.glob(output_spectrum) == []:
                print(f"<  INFO  > : Generate {name_tag} spectrum for BACKSCALE")
                with open(
                    f"{self.logdir}/{src_name}{instr}_{name_tag}_spectrum_BACKSCALE.log",
                    "w+",
                ) as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
                # run backscale
                inargs = [f"spectrumset={output_spectrum}"]
                with open(
                    f"{self.logdir}/{src_name}{instr}_{name_tag}_backscale_BACKSCALE.log",
                    "w+",
                ) as f:
                    with contextlib.redirect_stdout(f):
                        w("backscale", inargs).run()
            backscale.append(fits.open(output_spectrum)[1].header["BACKSCAL"])
            # print(fits.open(output_spectrum)[1].header["BACKSCAL"])
            
        if backscale[1] == 0:
            raise ValueError("Backscale value for the background is 0")
        if backscale[0] == 0:
            raise ValueError("Backscale value for the source is 0")
        
        return backscale[0] / backscale[1]

    def gen_lightcurves_manual(
        self,
        src_name,
        binning,
        user_defined_bti=None,
        verbose=False,
        min_Frac_EXP=0.3,
        CCDNR=4,
        PATTERN=4,
        PI=[200, 10000],
        t_clip_start=10,
        t_clip_end=100,
    ):
        """
        Extract light-curves manually from the event files from both the source and background regions
        
        Parameters
        ----------
        src_name : str
            Name of the source
        binning : float 
            Binning time in seconds
        user_defined_bti : list, optional
            User defined good time intervals, by default None
        verbose : bool, optional
            Verbose, by default False
        min_Frac_EXP : float, optional
            Minimum fraction of exposure time, by default 0.3
        CCDNR : int, optional
            CCD number, by default 4, only for EPIC-PN
        PATTERN : int, optional
            Pattern, by default 4, only for EPIC-PN
        PI : list, optional
            Energy range, by default [200,10000]
        t_clip_start : int, optional
            Clip the first seconds, of the light-curve, by default 10
        t_clip_end : int, optional
            Clip the last seconds, of the light-curve, by default 100       
        
        """
        src_name += "_"
        print(f"\t<  INFO  > : Generating light-curves manually")
        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")
            event_file = self.obs_files[instr]["evts"]

            event_file_filtered = f"{instr}_barely_filtered.fits"

            if instr == "EPN":
                expression = "#XMMEA_EP && FLAG == 0"
            else:
                raise NotImplementedError("Only EPN implemented")
                expression = "#XMMEA_EM && FLAG == 0"

            # generate a barely filtered event file
            inargs = [
                f"table={event_file}",
                "withfilteredset=Y",
                f"filteredset={event_file_filtered}",
                f"expression={expression}",
            ]
            if glob.glob(event_file_filtered) == []:
                w("evselect", inargs).run()

            # generate the source and background event files
            src_event_file = f"{src_name}{instr}_evts_src.fits"
            bkg_event_file = f"{src_name}{instr}_evts_bkg.fits"
            output = [src_event_file, bkg_event_file]
            regions = [self.regions[instr]["src"], self.regions[instr]["bkg"]]

            for i, reg in enumerate(regions):
                inargs = [
                    f"table={event_file_filtered}",
                    "withfilteredset=Y",
                    f"filteredset={output[i]}",
                    f"expression=((X,Y) in  {reg})",
                ]
                if glob.glob(output[i]) == []:
                    w("evselect", inargs).run()
                    
            # open the event lists to check the CCNR
            src_hdu = fits.open(src_event_file)
            print(f"The source events are located on CCD: {np.unique(src_hdu['EVENTS'].data['CCDNR'])}")
            
            if not np.all(src_hdu["EVENTS"].data["CCDNR"] ==4.):
                raise ValueError("Not all events are in the CCDNR 4 in the source event file")
            bkg_hdu = fits.open(bkg_event_file)
            print(f"The background events are located on CCD: {np.unique(bkg_hdu['EVENTS'].data['CCDNR'])}")
            if not np.all(bkg_hdu["EVENTS"].data["CCDNR"] ==4.):
                print(np.unique(bkg_hdu["EVENTS"].data["CCDNR"]))
                raise ValueError("Not all events are in the CCDNR 4 in the background event file")
            
            # get the backscale value
            scale = self.get_scale_value(src_name, src_event_file, bkg_event_file)

            if type(PI[0]) == int:
                energies = [PI]
            else:
                energies = PI

            for PI in energies:
                print(
                    f"\t<  INFO  > : Processing energy range : {PI[0]/1000}-{PI[1]/1000} keV"
                )

                t, net, err, bg, bg_err, t_bin, clean_Frac_EXP, T0 = get_lightcurve(
                    src_event_file,
                    bkg_event_file,
                    scale,
                    user_defined_bti=user_defined_bti,
                    verbose=verbose,
                    min_Frac_EXP=min_Frac_EXP,
                    CCDNR=CCDNR,
                    input_timebin_size=binning,
                    PATTERN=PATTERN,
                    PI=PI,
                    t_clip_start=t_clip_start,
                    t_clip_end=t_clip_end,
                )

                # save the lightcurve
                # arr = np.array([t, net,err,bg,bg_err,Frac_EXP],dtype=float).T
                arr = np.array([t, net, err, bg, bg_err], dtype=float).T
                frac = np.array([t_bin[:-1], clean_Frac_EXP], dtype=float).T
                np.savetxt(
                    f"{src_name}{instr}_{self.ID}_lc_{PI[0]/1000}-{PI[1]/1000}.txt",
                    arr,
                    header=f"T0 (TT): {T0}" + "\n T" + "\ntime net err bg bg_err",
                )
                np.savetxt(
                    f"{src_name}{instr}_{self.ID}_{PI[0]/1000}-{PI[1]/1000}_frac.txt",
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
                    f"lightcurve_{PI[0]/1000}-{PI[1]/1000}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

    def correct_GTI_timeseries(self, time_src, start_GTI, stop_GTI, dt):
        maxi = len(time_src) - 1
        k = 0
        L = []

        time = time_src - dt / 2  # time_src is the middle of the time bin

        for i in range(maxi):
            if time[i] < start_GTI[k]:
                print(
                    "Start of the time bin is before the start of the good time interval"
                )
                if time[i + 1] < start_GTI[k]:
                    print(
                        "1. End of the time bin is before the start of the good time interval"
                    )
                    print(f"i={i} Time bin is not in any good time interval\n")
                    L.append(0)
                else:
                    print(
                        "2. End of the time bin is after the start of the good time interval"
                    )
                    if time[i + 1] < stop_GTI[k]:
                        print(
                            "2.1. End of the time bin is before the end of the good time interval"
                        )
                        fracexp = (time[i + 1] - start_GTI[k]) / (time[i + 1] - time[i])
                    else:
                        print(
                            "2.2. End of the time bin is after the end of the good time interval"
                        )
                        fracexp = (stop_GTI[k] - start_GTI[k]) / (time[i + 1] - time[i])

                        j = k + 1
                        while time[i + 1] > start_GTI[j]:
                            print(f"2.2.1 j = {j}")
                            if time[i + 1] < stop_GTI[j]:
                                fracexp += (time[i + 1] - start_GTI[j]) / (
                                    time[i + 1] - time[i]
                                )
                                k = j
                                print(f"BREAK: {k}")
                                break
                            fracexp += (stop_GTI[j] - start_GTI[j]) / (
                                time[i + 1] - time[i]
                            )
                            j += 1
                        k = j
                    print(fracexp)
                    print(f"i={i} Time bin is partially in the good time interval\n")
                    L.append(fracexp)
                    # k = k+1
            else:
                if time[i + 1] < stop_GTI[k]:
                    print(f"i={i} 3. Time bin is fully in the good time interval\n")
                    L.append(1)

                else:
                    print(
                        f"4. End of the time bin is after the end of the good time interval"
                    )

                    fracexp = (stop_GTI[k] - time[i]) / (time[i + 1] - time[i])
                    j = k + 1
                    while time[i + 1] > start_GTI[j]:
                        print(f"j = {j}")
                        if time[i + 1] < stop_GTI[j]:
                            fracexp += (time[i + 1] - start_GTI[j]) / (
                                time[i + 1] - time[i]
                            )
                            k = j
                            print(f"BREAK: {j}")
                            break
                        fracexp += (stop_GTI[j] - start_GTI[j]) / (
                            time[i + 1] - time[i]
                        )
                        print(f"iterate {j} {fracexp}")
                        j += 1

                    k = j
                    print(fracexp)
                    print(f"i={i} Time bin is partially in the good time interval\n")

                    L.append(fracexp)

        if time[-1] >= start_GTI[k]:
            if time[-1] + dt < stop_GTI[k]:
                L.append(1)
            else:
                fracexp = (stop_GTI[k] - time[-1]) / (dt)
                L.append(fracexp)
        else:
            L.append(0)

        return np.array(L)

    def rescale_rate(
        self,
        time_src,
        start_GTI,
        stop_GTI,
        rate_src,
        rate_err_src,
        rate_bkg,
        rate_err_bkg,
        dt,
        scale,
    ):
        corr_gti = self.correct_GTI_timeseries(time_src, start_GTI, stop_GTI, dt)
        rate_corrected = np.copy(rate_src) - np.copy(rate_bkg) * scale
        nonzero_corr = np.where(corr_gti != 0)
        rate_corrected[nonzero_corr] = (
            rate_src[nonzero_corr] - rate_bkg[nonzero_corr]
        ) / corr_gti[nonzero_corr]
        rate_err_corrected = np.sqrt(rate_err_src**2 + (rate_err_bkg * scale) ** 2)
        return rate_corrected, rate_err_corrected, corr_gti

    def write_fits(self, time_src, rate, rate_err, corr_gti, filename):
        PRIMARY = fits.PrimaryHDU()
        RATE_TABLE = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="TIME", format="D", array=time_src),
                fits.Column(name="RATE", format="D", array=rate),
                fits.Column(name="ERROR", format="D", array=rate_err),
                fits.Column(name="FRACEXP", format="D", array=corr_gti),
            ]
        )
        RATE_TABLE.name = "RATE"
        hdu_list = fits.HDUList([PRIMARY, RATE_TABLE])
        print(f"Writing {filename}...")
        print(hdu_list.info())
        print(hdu_list[1].header)
        print(hdu_list[1].data)
        hdu_list.writeto(f"manual_{filename}_lightcurve.fits", overwrite=True)
        return hdu_list

    def create_lightcurve(self, instr, src_name):
        for energy_range in self.energybands:
            low, up = energy_range
            filename = f"{src_name}_lc_{instr}_{low/1000}_{up/1000}"

            src_lc = fits.open(
                f"{self.workdir}/{self.ID}_{src_name}{instr}_lc_src_{low/1000}-{up/1000}.fits"
            )
            bkg_lc = fits.open(
                f"{self.workdir}/{self.ID}_{src_name}{instr}_lc_bkg_{low/1000}-{up/1000}.fits"
            )
            pn_gti = fits.open(self.obs_files[instr]["gti"])

            start_GTI, stop_GTI = (
                pn_gti["STDGTI"].data["START"],
                pn_gti["STDGTI"].data["STOP"],
            )
            rate_src, rate_err_src, time_src = (
                src_lc["RATE"].data["RATE"],
                src_lc["RATE"].data["ERROR"],
                src_lc["RATE"].data["TIME"],
            )
            rate_bkg, rate_err_bkg, time_bkg = (
                bkg_lc["RATE"].data["RATE"],
                bkg_lc["RATE"].data["ERROR"],
                bkg_lc["RATE"].data["TIME"],
            )
            dt = time_src[1] - time_src[0]
            scale = self.get_backscale_value(instr, src_name)
            rate_corrected, rate_err_corrected, corr_gti = self.rescale_rate(
                time_src,
                start_GTI,
                stop_GTI,
                rate_src,
                rate_err_src,
                rate_bkg,
                rate_err_bkg,
                dt,
                scale,
            )
            hdulist = self.write_fits(
                time_src, rate_corrected, rate_err_corrected, corr_gti, filename
            )
            return hdulist

    def RGS_disp(self):
        """    
        
        rgsimplot endispset='my_pi.fit' spatialset='my_spatial.fit' \
     srcidlist='1' srclistset='PxxxxxxyyyyRrzeeeSRCLI_0000.FIT' \
     device=/xs
  
    """
        self.find_RGS_eventfiles()
        for instr in ["R1", "R2"]:
            srcli = glob.glob(f"P*{instr}*SRCLI*")[0]

            if glob.glob(f"{self.ID}_{instr}_spatial.fits") == []:
                print(f"<  INFO  > : Generate spatial image for {instr}")
                inargs = [
                    f'table={self.obs_files[instr]["evts"]}:EVENTS',
                    "xcolumn=M_LAMBDA",
                    "ycolumn=XDSP_CORR",
                    f"imageset={self.ID}_{instr}_spatial.fits",
                ]
                with open(f"{self.logdir}/{instr}_spatial.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()

            if glob.glob(f"{self.ID}_{instr}_PI_image.fits") == []:
                print(f"<  INFO  > : Generate PI image for {instr}")
                inargs = [
                    f'table={self.obs_files[instr]["evts"]}:EVENTS',
                    "xcolumn=M_LAMBDA",
                    "ycolumn=PI",
                    "yimagemin=0",
                    "yimagemax=3000",
                    f"imageset={self.ID}_{instr}_PI_image.fits",
                    f"expression=REGION({srcli}:RGS{instr[-1]}_SRC1_SPATIAL,M_LAMBDA,XDSP_CORR)",
                ]
                with open(f"{self.logdir}/{instr}_PI_image.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
            if glob.glob(f"{self.ID}_{instr}_disp.png") == []:
                inargs = [
                    f"endispset={self.ID}_{instr}_PI_image.fits",
                    f"spatialset={self.ID}_{instr}_spatial.fits",
                    f"srclistset={srcli}",
                    "colour=3",
                    "invert=no",
                    f"plotfile={self.ID}_{instr}_disp.png",
                    f"device=/png",
                ]

                with open(f"{self.logdir}/{instr}_rgsimplot.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("rgsimplot", inargs).run()

    def RGS_background_lightcurve(self, bin_size=100):
        """      evselect table=PxxxxxxyyyyRrzeeeEVENLI0000.FIT timebinsize=100 \
     rateset=my_rgs1_background_lc.fit \
     makeratecolumn=yes maketimecolumn=yes \
     expression='(CCDNR==9)&&(REGION(PxxxxxxyyyyRrzeeeSRCLI_0000.FIT:RGS1_BACKGROUND,M_LAMBDA,XDSP_CORR))'"""
        self.find_RGS_eventfiles()
        print(" INFO  : Generate background lightcurves for RGS")
        for instr in ["R1", "R2"]:
            srcli = glob.glob(f"P*{instr}*SRCLI*")[0]

            if glob.glob(f"{self.ID}_{instr}_background_lc.fits") == []:
                print(f" INFO  : Generate background lightcurves for {instr}")

                inargs = [
                    f'table={self.obs_files[instr]["evts"]}',
                    f"rateset={self.ID}_{instr}_background_lc.fits",
                    f"timebinsize={bin_size}",
                    f"makeratecolumn=yes",
                    f"maketimecolumn=yes",
                    f"expression=CCDNR==9&&REGION({srcli}:RGS{instr[-1]}_BACKGROUND,M_LAMBDA,XDSP_CORR)",
                ]
                with open(f"{self.logdir}/{instr}_bkg_lc.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
            # plot the lightcurve
        if glob.glob(f"{self.plotdir}/{self.ID}_RGS_background_lc.pdf") and self.replot:
            print(f" INFO  : Plot background lightcurves for {instr}")
            hdu1 = fits.open(f"{self.ID}_R1_background_lc.fits")[1].data
            hdu2 = fits.open(f"{self.ID}_R2_background_lc.fits")[1].data
            fig, ax = plt.subplots(2, 1, figsize=(13, 9))
            ax[0].errorbar(
                hdu1["TIME"] - hdu1["TIME"][0],
                hdu1["RATE"],
                yerr=hdu1["ERROR"],
                fmt="o",
                color="k",
            )
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Rate (cts/s)")
            ax[0].set_title("RGS1", fontsize=18)
            ax[1].errorbar(
                hdu2["TIME"] - hdu2["TIME"][0],
                hdu2["RATE"],
                yerr=hdu2["ERROR"],
                fmt="o",
                color="k",
            )
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("Rate (cts/s)")
            ax[1].set_title("RGS2", fontsize=18)
            fig.tight_layout()
            fig.savefig(f"{self.plotdir}/{self.ID}_RGS_background_lc.pdf")

    def RGS_plot_lightcurve(self):
        print(" INFO  : Generate lightcurves for RGS")
        for order in [1, 2]:
            if (
                glob.glob(f"{self.plotdir}/{self.ID}_RGS_O{order}_lc.pdf") == []
                or self.replot
            ):
                print(f" INFO  : Plot RGS lightcurves for order {order}")
                src_1 = glob.glob(f"P{self.ID}R1*SRTSR_{order}*")[0]
                src_2 = glob.glob(f"P{self.ID}R2*SRTSR_{order}*")[0]
                bkg_1 = glob.glob(f"P{self.ID}R1*BGTSR_{order}*")[0]
                bkg_2 = glob.glob(f"P{self.ID}R2*BGTSR_{order}*")[0]
                hdu1 = fits.open(src_1)[1].data
                hdu2 = fits.open(src_2)[1].data
                hdu3 = fits.open(bkg_1)[1].data
                hdu4 = fits.open(bkg_2)[1].data
                fig, ax = plt.subplots(2, 1, figsize=(13, 9))
                ax[0].errorbar(
                    hdu1["TIME"] - hdu1["TIME"][0],
                    hdu1["RATE"],
                    yerr=hdu1["ERROR"],
                    fmt="o",
                    color="k",
                    label="Source",
                )
                # similar to the background lightcurve,
                # ax[0].errorbar(hdu1['TIME']-hdu1['TIME'][0],hdu1['BACKV'],yerr=hdu1['BACKE'],fmt='o',color='g',label='Source-BKG')
                ax[0].errorbar(
                    hdu3["TIME"] - hdu3["TIME"][0],
                    hdu3["RATE"],
                    yerr=hdu3["ERROR"],
                    fmt="o",
                    color="r",
                    label="Background",
                )
                ax[0].set_xlabel("Time (s)")
                ax[0].set_ylabel("Rate (cts/s)")
                ax[0].set_title("RGS1", fontsize=18)
                ax[0].legend()
                ax[1].errorbar(
                    hdu2["TIME"] - hdu2["TIME"][0],
                    hdu2["RATE"],
                    yerr=hdu2["ERROR"],
                    fmt="o",
                    color="k",
                    label="Source",
                )
                # ax[1].errorbar(hdu2['TIME']-hdu2['TIME'][0],hdu2['BACKV'],yerr=hdu2['BACKE'],fmt='o',color='g',label='Source-BKG')
                ax[1].errorbar(
                    hdu4["TIME"] - hdu4["TIME"][0],
                    hdu4["RATE"],
                    yerr=hdu4["ERROR"],
                    fmt="o",
                    color="r",
                    label="Background",
                )
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Rate (cts/s)")
                ax[1].set_title("RGS2", fontsize=18)
                # ax[1].legend()
                fig.suptitle(f"Light curve RGS order{order}", fontsize=20)
                fig.tight_layout()
                fig.savefig(f"{self.plotdir}/{self.ID}_RGS_O{order}_lc.pdf")

    def gen_EPIC_spectra(self, src_name, **kwargs):
        """

        Generate the spectral files

        """
        if os.getcwd() != self.workdir:
            os.chdir(self.workdir)

        if "withminSN=yes" in kwargs:
            grouping = kwargs.get("withminSN", "withminSN=yes")
            min_group = kwargs.get("minSN", "minSN=5")
        else:
            grouping = kwargs.get("withCounts", "withCounts=yes")
            min_group = kwargs.get("mincounts", "mincounts=25")
        oversample = kwargs.get("oversample", "oversample=3.0")
        abscor = kwargs.get("abscor", False)

        print(f"<  INFO  > : Generating spectra")

        for instr in self.instruments:
            print(f"\t<  INFO  > : Processing instrument : {instr}")

            if instr == "EPN":
                specchanmax = 20479
                pattern = 4
                flag = "(FLAG==0) "
            else:
                specchanmax = 11999
                pattern = 12
                flag = "#XMMEA_EM "
            partnumber = 0

            # if several slices are defined, we split the spectrum in several parts
            for interval in self.slices:
                if interval == "all":
                    timeslice = ""
                    portion = "all"
                else:
                    start = self.find_start_time()
                    if 0 in interval:
                        timeslice = f" && (TIME <={start+interval[1]*1e3})"
                    elif not -1 in interval:
                        timeslice = f" && (TIME in [{start+interval[0]*1e3}:{start+interval[1]*1e3}])"
                    else:
                        timeslice = f" && (TIME >={start+interval[0]*1e3})"
                    partnumber += 1
                    portion = f"p{partnumber}"

                low, up = self.energy_range

                for name_tag in ["src", "bkg"]:
                    # ---generate the spectrum
                    output_spectrum = f"{self.workdir}/{self.ID}_{src_name}{instr}_spectrum_{name_tag}_{low/1000}-{up/1000}_{portion}.fits"
                    # if instr == "EPN":
                    #     flag = "(FLAG==0) "
                    # elif instr == "EMOS1" or instr == "EMOS2":
                    #     flag = "#XMMEA_EM "
                    expression = (
                        f"{flag} && (PATTERN <={pattern}) && ((X,Y) IN {self.regions[instr][name_tag]})"
                        + timeslice
                    )

                    inargs = [
                        f'table={self.obs_files[instr]["clean_evts"]}',
                        "withspectrumset=yes",
                        f"spectrumset={output_spectrum}",
                        "energycolumn=PI",
                        "spectralbinsize=5",
                        "withspecranges=yes",
                        "specchannelmin=0",
                        f"specchannelmax={specchanmax}",
                        f"expression={expression}",
                    ]

                    if glob.glob(output_spectrum) == []:
                        print(
                            f"<  INFO  > : Generate {name_tag} spectrum {low/1000}-{up/1000} keV"
                        )
                        with open(
                            f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_spectrum.log",
                            "w+",
                        ) as f:
                            with contextlib.redirect_stdout(f):
                                w("evselect", inargs).run()

                        # ---calculate the area of the regions
                        inargs = [
                            f"spectrumset={output_spectrum}",
                            f'badpixlocation={self.obs_files[instr]["clean_evts"]}',
                        ]

                        print(
                            f"<  INFO  > : Running backscale on {name_tag} spectrum {low/1000}-{up/1000} keV"
                        )
                        with open(
                            f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_backscale.log",
                            "w+",
                        ) as f:
                            with contextlib.redirect_stdout(f):
                                w("backscale", inargs).run()

                src_spectrum = f"{self.ID}_{src_name}{instr}_spectrum_src_{low/1000}-{up/1000}_{portion}.fits"
                bkg_spectrum = f"{self.ID}_{src_name}{instr}_spectrum_bkg_{low/1000}-{up/1000}_{portion}.fits"

                # ----generate the redistribution matrix
                output_rmf = (
                    f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.rmf"
                )
                if glob.glob(output_rmf) == []:
                    print(
                        f"<  INFO  > : Running rmfgen to generate the response matrix {low/1000}-{up/1000} keV"
                    )
                    inargs = [f"spectrumset={src_spectrum}", f"rmfset={output_rmf}"]

                    with open(
                        f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_rmfgen.log",
                        "w+",
                    ) as f:
                        with contextlib.redirect_stdout(f):
                            w("rmfgen", inargs).run()
                # ----generate the ancillary file
                output_arf = (
                    f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.arf"
                )
                if glob.glob(output_arf) == []:
                    print(
                        f"<  INFO  > : Running arfgen to generate the ancillary file {low/1000}-{up/1000} keV"
                    )
                    inargs = [
                        f"spectrumset={src_spectrum}",
                        f"arfset={output_arf}",
                        f'applyabsfluxcorr={"yes"if abscor else "no"}',
                        "withrmfset=yes",
                        f"rmfset={output_rmf}",
                        f'badpixlocation={self.obs_files[instr]["clean_evts"]}',
                        "detmaptype=psf",
                    ]

                    with open(
                        f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_arfgen.log",
                        "w+",
                    ) as f:
                        with contextlib.redirect_stdout(f):
                            w("arfgen", inargs).run()

                # ----grouping the spectrum
                grouped_spectrum = f"{self.ID}_{src_name}{instr}_grouped_spectrum_{low/1000}-{up/1000}_{portion}.fits"
                rmf_name = (
                    f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.rmf"
                )
                arf_name = (
                    f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.arf"
                )

                if glob.glob(grouped_spectrum) == []:
                    print(
                        f"<  INFO  > : Running specgroup to group the spectral files {low/1000}-{up/1000} keV"
                    )
                    inargs = [
                        f"spectrumset={src_spectrum}",
                        oversample,
                        "withoversampling=yes",
                        grouping,
                        min_group,
                        "withbgdset=yes",
                        "witharfset=yes",
                        "withrmfset=yes",
                        f"rmfset={rmf_name}",
                        f"arfset={arf_name}",
                        f"backgndset={bkg_spectrum}",
                        f"groupedset={grouped_spectrum}",
                    ]

                    with open(
                        f"{self.logdir}/{src_name}{instr}_{low/1000}-{up/1000}_{portion}_specgroup.log",
                        "w+",
                    ) as f:
                        with contextlib.redirect_stdout(f):
                            w("specgroup", inargs).run()
