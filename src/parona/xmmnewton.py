import os
import sys
import tarfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from pysas.wrapper import Wrapper as w
from astropy.io import fits
from astropy.table import Table
import contextlib
import shutil
from .callds9 import start_ds9


class ObservationXMM:
    """
    Class for processing an XMM-Newton observation

    The folder tree is this one

    path/odf
    path/data/obsid_datadir
    path/obsid_workdir
    path/obsid_workdir/logs
    path/obsid_workdir/plots

    """

    def __init__(self, path, obsidentifier, date, **kwargs):
        self.ID = obsidentifier
        self.date = date
        self.slices = kwargs.get('slices', ["all"])
        self.nslices = kwargs.get('nslices', 0)
        self.instruments = ["EPN", "EMOS1", "EMOS2"]
        self.obs_files = {}
        self.regions = {}

        for instr in self.instruments:
            self.obs_files[instr] = {}
            self.regions[instr] = {}

        self.energybands = kwargs.get("energybands",[[500, 3000], [3000, 10000]])
        self.energy_range = [np.min(np.array(self.energybands).flatten()), np.max(
            np.array(self.energybands).flatten())]
        self.check_repertories(path)
        self.get_odf()
        self.extract_odf()

        self.replot = True

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
        
        
        if not os.path.isdir(f'{path}/odf'):
            os.mkdir(f'{path}/odf')

        if not os.path.isdir(f'{path}/data'):
            os.mkdir(f'{path}/data')
        if not os.path.isdir(f'{path}/data/{self.ID}_datadir'):
            os.mkdir(f'{path}/data/{self.ID}_datadir')
        if not os.path.isdir(f'{path}/{self.ID}_workdir'):
            os.mkdir(f'{path}/{self.ID}_workdir')
        if not os.path.isdir(f'{path}/{self.ID}_workdir/logs'):
            os.mkdir(f'{path}/{self.ID}_workdir/logs')
        if not os.path.isdir(f'{path}/{self.ID}_workdir/plots'):
            os.mkdir(f'{path}/{self.ID}_workdir/plots')

        self.odfdir = f'{path}/odf'
        self.datadir = f'{path}/data/{self.ID}_datadir/'
        self.workdir = f'{path}/{self.ID}_workdir/'
        self.logdir = f'{path}/{self.ID}_workdir/logs/'
        self.plotdir = f'{path}/{self.ID}_workdir/plots/'
        os.environ["SAS_WORKDIR"] = self.workdir

    def get_odf(self):
        """
        Get the observation data file (ODF) archive from the repertory odf or from
        the XMM-Newton XSA database.
        
        If the ODF is not in the odf repertory, will download it from the XSA database using
        the astroquery package.
        
        
        """
        if not glob.glob(f"{self.odfdir}/{self.ID}.*") == []:
            self.obs_files["ODF"] = glob.glob(f"{self.odfdir}/{self.ID}.*")[0]
            print("<  INFO  > : ODF already downloaded")
        else:
            print("<  INFO  > : Downloading ODF")
            from astroquery.esa.xmm_newton import XMMNewton

            XMMNewton.download_data(
                f'{self.ID}', level="ODF", filename=f"{self.odfdir}/{self.ID}.tar")
            self.obs_files["ODF"] = glob.glob(f"{self.odfdir}/{self.ID}.*")[0]

    def extract_odf(self):
        """
        Extract the ODF archive into the datadir folder.
        """
        if not os.listdir(self.datadir):
            import tarfile
            print(f'<  INFO  > : Extracting the ODF')
            tf = tarfile.open(self.obs_files["ODF"], mode='r')
            if os.listdir(path=self.datadir) == []:
                tf.extractall(path=self.datadir, numeric_owner=True)
            if glob.glob(f'{self.datadir}/*TAR') != [] or len(glob.glob(f'{self.datadir}/*')) < 10:
                tf = tarfile.open(
                    glob.glob(f'{self.datadir}/*TAR')[0], mode='r')
                tf.extractall(path=self.datadir, numeric_owner=True)
                os.remove(glob.glob(f'{self.datadir}/*TAR')[0])

    def calibrate(self):
        """

        Calibrate the ODF with cifbuild and odfingest

        """
        os.chdir(self.workdir)
        os.environ["SAS_ODF"] = self.datadir
        if not len(glob.glob(f"{self.logdir}/cifbuild.log")) > 0:
            print(f'<  INFO  > : Running cifbuild to calibrate observation')
            with open(f"{self.logdir}/cifbuild.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("cifbuild", []).run()
            os.chdir(self.workdir)
        os.environ["SAS_CCF"] = f'{self.workdir}/ccf.cif'

        if not len(glob.glob(f'{self.workdir}/*SUM.SAS')) > 0:
            print(f'<  INFO  > : Running odfingest')
            with open(f"{self.logdir}/odfingest.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("odfingest", [
                      f'outdir={self.workdir}', f'odfdir={self.datadir}']).run()
        os.environ["SAS_ODF"] = glob.glob(f'{self.workdir}/*SUM.SAS')[0]

    def gen_evts(self):
        """
        Generate the event list the EPIC pn and MOS CCD

        will run epproc and emproc if the event list is not already present in the workdir
        """

        os.chdir(self.workdir)
        if glob.glob(f"{self.workdir}/*EPN*ImagingEvts*") == [] and "EPN" in self.instruments:
            print(f'<  INFO  > : Running epproc to generate events list for the EPIC-pn')
            with open(f"{self.logdir}/epproc.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("epproc", []).run()
        if glob.glob(f"{self.workdir}/*MOS*ImagingEvts*") == [] and ("EMOS1" in self.instruments or "EMOS2" in self.instruments):
            print(f'<  INFO  > : Running emproc to generate events list for the EPIC-MOS 1 & 2')
            with open(f"{self.logdir}/emproc.log", "w+") as f:
                with contextlib.redirect_stdout(f):
                    w("emproc", []).run()

        for instr in self.instruments:
            self.obs_files[instr]["evts"] = self.find_eventfile(instr)

    def find_eventfile(self, instr):
        """
        Return the event list file
        """
        buff = glob.glob(f"{self.workdir}/*{instr}_*ImagingEvts*")
        if len(buff) == 1:
            input_eventfile = buff[0]
        else:
            for res in buff:
                if "_S" in res:
                    input_eventfile = res
        return input_eventfile

    def gen_flarelc(self):
        """

        Generate the flare background rate file

        """
        print('<  INFO  > : Generating flares light curves')

        for instr in self.instruments:

            print(f'\t<  INFO  > : Processing instrument : {instr}')
            self.obs_files[instr]["flare"] = f"{self.workdir}/{self.ID}_{instr}_FlareBKGRate.fits"

            if glob.glob(self.obs_files[instr]["flare"]) == []:
                print(
                    f'\t<  INFO  > : Using event list : {self.obs_files[instr]["evts"]}')

                if "PN" in instr:
                    expression = '#XMMEA_EP&&(PI>10000.&&PI<12000.)&&(PATTERN==0.)'
                else:
                    expression = '#XMMEA_EM&&(PI>10000)&&(PATTERN==0.)'
                    
                inargs = [f'table={self.obs_files[instr]["evts"]}', 
                          'withrateset=Y', 
                          f'rateset={self.obs_files[instr]["flare"]}',
                          'maketimecolumn=Y', 
                          'timebinsize=100',
                          'makeratecolumn=Y',
                          f'expression={expression}']
                
                with open(f"{self.logdir}/{instr}_flares.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
            # -- Plot the light curve for the flares --
            if glob.glob(f"{self.plotdir}/{self.ID}_{instr}_FlareBKGRate.pdf") == [] or self.replot == True:
                self.plot_flares_lc(instr)

    def plot_flares_lc(self, instr):
        """

        Plot the light curve of the flares

        """

        hdu_list = fits.open(self.obs_files[instr]['flare'], memmap=True)
        lc_data = Table(hdu_list[1].data)
        lc_data["TIME"] -= lc_data["TIME"][0]
        fig, axis = plt.subplots(1, 1, figsize=(8  , 6))
        axis.plot(lc_data["TIME"]/1e3, lc_data["RATE"])
        if "PN" in instr:
            axis.hlines(
                0.4, lc_data["TIME"][0]/1e3, lc_data["TIME"][-1]/1e3, color="red", label="0.4 cts/s")
        else:
            axis.hlines(
                0.35, lc_data["TIME"][0]/1e3, lc_data["TIME"][-1]/1e3, color="red", label="0.35 cts/s")
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
        print(f'<  INFO  > : Generating the GTI')
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            self.obs_files[instr]["gti"] = f"{self.workdir}/{self.ID}_{instr}_GTI.fits"
            if glob.glob(self.obs_files[instr]["gti"]) == []:
                if "PN" in instr:
                    expression = 'RATE<=0.4'
                else:
                    expression = 'RATE<=0.35'
                    
                inargs = [f'table={self.obs_files[instr]["flare"]}',
                          f'gtiset={self.obs_files[instr]["gti"]}', 
                          f'expression={expression}']
                print(f'\t<  INFO  > : Running tabgtigen')
                with open(f"{self.logdir}/{instr}_tabgtigen.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("tabgtigen", inargs).run()

    def gen_clean_evts(self):
        """

        Generating cleaned event list

        """
        print(f'<  INFO  > : Generating clean event lists')
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            self.obs_files[instr]['clean_evts'] = f"{self.workdir}/{self.ID}_{instr}_evts_clean.fits"

            if glob.glob(self.obs_files[instr]['clean_evts']) == []:

                if "PN" in instr:
                    expression = f'#XMMEA_EP && gti( {self.obs_files[instr]["gti"]} , TIME ) && (PI >150)'
                else:
                    expression = f'#XMMEA_EM && gti( {self.obs_files[instr]["gti"]} , TIME ) && (PI >150)'

                inargs = [f'table={self.obs_files[instr]["evts"]}', 
                          'withfilteredset=Y', 
                          f'filteredset={self.obs_files[instr]["clean_evts"]}',
                          'destruct=Y',
                          'keepfilteroutput=T',
                          f'expression={expression}']
                
                print(f'<  INFO  > : Filtering flares to produce events list')
                with open(f"{self.logdir}/{instr}_filtering_flares.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()

    def gen_images(self):
        """

        Generate images from the cleaned event list

        """
        print(f'<  INFO  > : Generating images')
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            self.obs_files[instr]["image"] = f"{self.workdir}/{self.ID}_{instr}_image.fits"
            if glob.glob(self.obs_files[instr]["image"]) == []:
                
                inargs = [f'table={self.obs_files[instr]["clean_evts"]}', 
                          'imagebinning=binSize',
                          f'imageset={self.obs_files[instr]["image"]}',
                          'withimageset=yes', 
                          'xcolumn=X', 
                          'ycolumn=Y', 
                          'ximagebinsize=80',
                          'yimagebinsize=80']
                
                print(f'\t<  INFO  > : Generate an image')
                with open(f"{self.logdir}/{instr}_image.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()

    def select_regions(self, src_name):
        """

        Select the source and background regions with ds9

        """
        print(f'<  INFO  > : Selection of source and background regions with ds9')

        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')
            try:
                python_ds9.set("regions select all")
            except:
                python_ds9 = start_ds9()

            python_ds9.set("regions select all")
            python_ds9.set("regions delete")
            python_ds9.set("file "+self.obs_files[instr]["image"])
            python_ds9.set("scale log")
            python_ds9.set("cmap b")
            self.obs_files[instr]["positions"] = f"{self.workdir}/{self.ID}_{src_name}{instr}_positions.txt"
            if glob.glob(self.obs_files[instr]["positions"]) == []:
                print("Draw FIRST the region for the source and THEN the background")
                input("Press Enter to continue...")
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")
                self.regions[instr]["src"] = python_ds9.get(
                    "regions").split("\n")[0]
                self.regions[instr]["bkg"] = python_ds9.get(
                    "regions").split("\n")[1]
                np.savetxt(self.obs_files[instr]["positions"], np.array(
                    [self.regions[instr]["src"], self.regions[instr]["bkg"]]), fmt="%s")
            else:
                self.regions[instr]["src"], self.regions[instr]["bkg"] = np.genfromtxt(
                    self.obs_files[instr]["positions"], dtype='str')
                python_ds9.set(
                    f'regions load {self.obs_files[instr]["positions"]}')
                python_ds9.set("regions edit yes")
                python_ds9.set("regions format ciao")
                python_ds9.set("regions system physical")
            python_ds9.set("zoom to fit")
            #python_ds9.set(f"saveimage png {self.plotdir}/{self.ID}_{src_name}{instr}_image.png")

    def check_pileup(self, src_name):
        """

        Check pileup with epatplot

        """
        print('<  INFO  > : Checking pile-up during observation')
        for instr in self.instruments:

            print(f'\t<  INFO  > : Processing instrument : {instr}')
            self.obs_files[instr]["epatplot"] = f"{self.ID}_{src_name}{instr}_pat.ps"
            self.obs_files[instr]["clean_filt"] = f"{self.workdir}/{self.ID}_{instr}_clean_filtered.fits"

            if len(glob.glob(f"{self.plotdir}/*{src_name}*pat.ps")) != 3:
                
                inargs = [f'table={self.obs_files[instr]["clean_evts"]}', 
                          'withfilteredset=yes',
                          f'filteredset={self.obs_files[instr]["clean_filt"]}',
                          'keepfilteroutput=yes', 
                          f'expression=((X,Y) in  {self.regions[instr]["src"]}) && gti({self.obs_files[instr]["gti"]} ,TIME)']
                
                print(f'\t<  INFO  > : Generate an event list for pile-up')
                with open(f"{self.logdir}/{src_name}{instr}_pileup.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("evselect", inargs).run()
                inargs = [f'set={self.obs_files[instr]["clean_filt"]}',
                          f'plotfile={self.obs_files[instr]["epatplot"]}']
                print(f'\t<  INFO  > : Running epatplot to evaluate pile-up')
                with open(f"{self.logdir}/{src_name}{instr}_epatplot.log", "w+") as f:
                    with contextlib.redirect_stdout(f):
                        w("epatplot", inargs).run()
                shutil.move(self.obs_files[instr]["epatplot"],
                            f"{self.plotdir}/{self.obs_files[instr]['epatplot']}")

    def gen_lightcurves(self, src_name):
        """

        Generate light curves for source and background for all energy bands
        Generate background subtracted light-curve

        """
        self.nobin = False
        if self.nobin:
            # Time resolution of the EPIC Camera
            # https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/epicmode.html
            tag = "nobin"
            binsize = {"EPN": 73.4e-3, "EMOS1": 2.6, "EMOS2": 2.6}
        else:
            tag = ""
            binsize = {"EPN": 1000, "EMOS1": 1000, "EMOS2": 1000}

        print('<  INFO  > : Generating light-curves')
        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')

            if len(glob.glob(f"{self.workdir}/*{src_name}{instr}*lc_src*")) != len(self.energybands):
                for name_tag in ['src', 'bkg']:
                    for energy_range in self.energybands:
                        low, up = energy_range
                        lc_name = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_{name_tag}_{low/1000}-{up/1000}.fits"
                        if glob.glob(lc_name) == []:
                            if "PN" in instr:
                                expression = f'#XMMEA_EP && (PATTERN<=4) && (PI in [{low}:{up}]) && ((X,Y) IN {self.regions[instr][name_tag]})'
                            else:
                                expression = f'#XMMEA_EM && (PATTERN<=12) && (PI in [{low}:{up}]) && ((X,Y) IN {self.regions[instr][name_tag]})'
                                
                            inargs = [f'table={self.obs_files[instr]["clean_evts"]}', 
                                      'withrateset=yes',
                                      f'rateset={lc_name}',
                                      f'timebinsize={binsize[instr]}',
                                      'maketimecolumn=yes',
                                      'makeratecolumn=yes',
                                      f'expression={expression}']
                            
                            print(
                                f'\t\t<  INFO  > : Generate {name_tag} light curves {low/1000}-{up/1000} keV')
                            with open(f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_lc{tag}_light_curve.log", "w+") as f:
                                with contextlib.redirect_stdout(f):
                                    w("evselect", inargs).run()

            # ---light curves corrected from background
            for energy_range in self.energybands:
                low, up = energy_range
                lc_clean = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_clean_{low/1000}-{up/1000}.fits"
                if glob.glob(lc_clean) == []:
                    lc_src = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_src_{low/1000}-{up/1000}.fits"
                    lc_bkg = f"{self.workdir}/{self.ID}_{src_name}{instr}_lc{tag}_bkg_{low/1000}-{up/1000}.fits"
                    inargs = [f'srctslist={lc_src}', f'eventlist={self.obs_files[instr]["clean_evts"]}', f'outset={lc_clean}',
                              f'bkgtslist={lc_bkg}', 'withbkgset=yes', 'applyabsolutecorrections=yes']
                    print(
                        f'\t<  INFO  > : Running epiclccorr to correct light curves {low/1000}-{up/1000} keV')
                    with open(f"{self.logdir}/{src_name}{instr}_{low/1000}-{up/1000}_{tag}epiclccorr.log", "w+") as f:
                        with contextlib.redirect_stdout(f):
                            w("epiclccorr", inargs).run()
        if (not self.nobin) and self.replot:
            self.plot_lightcurves(src_name)

    def plot_lightcurves(self, src_name):
        # --- pdf light curves for energies 0.5-3 3-10 keV

        fig, ax = plt.subplots(2, 1, figsize=(8, 9))
        for (axis, energies) in zip(ax, ["0.5-3.0", "3.0-10.0"]):
            for instr in self.instruments:
                hdu_list = fits.open(
                    f"{self.workdir}/{self.ID}_{src_name}{instr}_lc_clean_{energies}.fits")
                lc_data = Table(hdu_list[1].data)
                lc_data["TIME"] -= lc_data["TIME"][0]
                axis.errorbar(lc_data["TIME"]/1000, lc_data["RATE"],
                              yerr=lc_data["ERROR"], fmt="o", markersize=5., label=instr)
                axis.legend(loc='upper left', bbox_to_anchor=(1, 0.9))
            axis.set_title(f"{energies} keV")
            axis.set_xlabel("Time (ks)")
            axis.set_ylabel("Rate (cts/s)")
        fig.suptitle(
            f"{self.ID} - Light curves - {self.date}", fontsize=20)
        fig.tight_layout()
        fig.savefig(f"{self.plotdir}/{self.ID}_{src_name}lightcurves.pdf")

    def find_start_time(self):
        """

        Find the start time for the slices

        """
        incidents_MOS1_CCD = [[6, "2005-03-09"], [3, "2012-12-11"]]

        Start = []
        for instr in self.instruments:
            if instr == "EPN":
                nCCD = 12
                list_CCD = range(1, nCCD+1)
            else:
                nCCD = 7
                list_CCD = range(1, nCCD+1)
            hdu_list = fits.open(self.obs_files[instr]["clean_evts"])
            L = []
            if instr == "EMOS1":
                if date.fromisoformat(self.date) > date.fromisoformat(incidents_MOS1_CCD[0][1]) and date.fromisoformat(self.date) < date.fromisoformat(incidents_MOS1_CCD[1][1]):
                    list_CCD = [1, 2, 3, 4, 5, 7]
                elif date.fromisoformat(self.date) > date.fromisoformat(incidents_MOS1_CCD[1][1]):
                    list_CCD = [1, 2, 4, 5, 7]
                else:
                    list_CCD = range(1, nCCD+1)
            for i in list_CCD:
                if i < 10:
                    string = f"0{i}"
                else:
                    string = f"{i}"
                L.append(hdu_list[f"STDGTI{string}"].data["START"][0])
            Start.append(max(L))
        self.start_time = max(Start)
        print("Instrument start : ",self.instruments[np.argmax(Start)])
        return self.start_time

    def gen_spectra(self, src_name, **kwargs):
        """

        Generate the spectral files

        """
        if 'withminSN=yes' in kwargs:
            grouping = kwargs.get("withminSN", 'withminSN=yes')
            min_group = kwargs.get("minSN", 'minSN=5')
        else:
            grouping = kwargs.get("withCounts", 'withCounts=yes')
            min_group = kwargs.get("mincounts", 'mincounts=25')
        oversample = kwargs.get("oversample", 'oversample=3.0')
        abscor = kwargs.get("abscor", True)

        print(f'<  INFO  > : Generating spectra')

        for instr in self.instruments:
            print(f'\t<  INFO  > : Processing instrument : {instr}')

            if instr == "EPN":
                specchanmax = 20479
                pattern = 4
                flag = "(FLAG==0) "
            else:
                specchanmax = 11999
                pattern = 12
                flag = "#XMMEA_EM "
            partnumber = 0
            for interval in self.slices:
                if interval == "all":
                    timeslice = ""
                    portion = 'all'
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
                    if instr == "EPN":
                        flag = "(FLAG==0) "
                    elif instr == "EMOS1" or instr == "EMOS2":
                        flag = "#XMMEA_EM "
                    expression = f"{flag} && (PATTERN <={pattern}) && ((X,Y) IN {self.regions[instr][name_tag]})"+timeslice
                    
                    inargs = [f'table={self.obs_files[instr]["clean_evts"]}',
                              'withspectrumset=yes',
                              f'spectrumset={output_spectrum}',
                              'energycolumn=PI', 
                              'spectralbinsize=5',
                              'withspecranges=yes', 
                              'specchannelmin=0',
                              f'specchannelmax={specchanmax}', 
                              f'expression={expression}']
                    
                    if glob.glob(output_spectrum) == []:
                        print(f'<  INFO  > : Generate {name_tag} spectrum {low/1000}-{up/1000} keV')
                        with open(f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_spectrum.log", "w+") as f:
                            with contextlib.redirect_stdout(f):
                                w("evselect", inargs).run()

                        # ---calculate the area of the regions
                        inargs = [ f'spectrumset={output_spectrum}', 
                                  f'badpixlocation={self.obs_files[instr]["clean_evts"]}']
                        
                        print(f'<  INFO  > : Running backscale on {name_tag} spectrum {low/1000}-{up/1000} keV')
                        with open(f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_backscale.log", "w+") as f:
                            with contextlib.redirect_stdout(f):
                                w("backscale", inargs).run()

                src_spectrum = f"{self.ID}_{src_name}{instr}_spectrum_src_{low/1000}-{up/1000}_{portion}.fits"
                bkg_spectrum = f"{self.ID}_{src_name}{instr}_spectrum_bkg_{low/1000}-{up/1000}_{portion}.fits"
                
                # ----generate the redistribution matrix
                output_rmf = f"{self.workdir}/{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.rmf"
                if glob.glob(output_rmf) == []:
                    print( f'<  INFO  > : Running rmfgen to generate the response matrix {low/1000}-{up/1000} keV')
                    inargs = [f'spectrumset={src_spectrum}', 
                              f'rmfset={output_rmf}']
                    
                    with open(f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_rmfgen.log", "w+") as f:
                        with contextlib.redirect_stdout(f):
                            w("rmfgen", inargs).run()
                # ----generate the ancillary file
                output_arf = f"{self.workdir}/{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.arf"
                if glob.glob(output_arf) == []:
                    print(f'<  INFO  > : Running arfgen to generate the ancillary file {low/1000}-{up/1000} keV') 
                    inargs = [f'spectrumset={src_spectrum}', 
                              f'arfset={output_arf}',
                              f'applyabsfluxcorr={"yes"if abscor else "no"}',
                              'withrmfset=yes', 
                              f'rmfset={output_rmf}',
                              f'badpixlocation={self.obs_files[instr]["clean_evts"]}', 
                              'detmaptype=psf']
                    
                    with open(f"{self.logdir}/{src_name}{instr}_{name_tag}_{low/1000}-{up/1000}_{portion}_arfgen.log", "w+") as f:
                        with contextlib.redirect_stdout(f):
                            w("arfgen", inargs).run()

                # ----grouping the spectrum
                grouped_spectrum = f"{self.ID}_{src_name}{instr}_grouped_spectrum_{low/1000}-{up/1000}_{portion}.fits"
                rmf_name = f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.rmf"
                arf_name = f"{self.ID}_{src_name}{instr}_{low/1000}-{up/1000}_{portion}.arf"

                if glob.glob(grouped_spectrum) == []:
                    print(f'<  INFO  > : Running specgroup to group the spectral files {low/1000}-{up/1000} keV')
                    inargs = [f'spectrumset={src_spectrum}', 
                              f'setbad=0:{low/1000},{up/1000}:26.0', 
                              'units=KEV', 
                              oversample,
                              'withoversampling=yes', 
                              grouping,
                              min_group, 
                              'withbgdset=yes',
                              'witharfset=yes', 
                              'withrmfset=yes',
                              f'rmfset={rmf_name}',
                              f'arfset={arf_name}',
                              f'backgndset={bkg_spectrum}',
                              f'groupedset={grouped_spectrum}']
                    
                    with open(f"{self.logdir}/{src_name}{instr}_{low/1000}-{up/1000}_{portion}_specgroup.log", "w+") as f:
                        with contextlib.redirect_stdout(f):
                            w("specgroup", inargs).run()

