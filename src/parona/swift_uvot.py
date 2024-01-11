import os
import sys
from shutil import copytree
import warnings

# astropy and astroquery
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.time import Time
from astroquery.simbad import Simbad
from regions import CirclePixelRegion, CircleSkyRegion


# heasoft
import heasoftpy as hsp
import pyvo as vo

# Ignore unimportant warnings
warnings.filterwarnings(
    "ignore",
    ".*Unknown element mirrorURL.*",
    vo.utils.xml.elements.UnknownElementWarning,
)

uvotsource = hsp.HSPTask("uvotsource")


class UVOTSwift:
    """To build a long-term light curve of a source using UVOT data from the Swift archive.

    Parameters
    ----------
    source_name : str
        Name of the source to be queried in Simbad
    filters : list
        List of filters to be used. Default is all filters.
    on_sciserver : bool
        If True, the files will be copied to the SciServer temporary directory. If False, the files will be downloaded to the local machine.

    """

    def __init__(
        self,
        source_name: str,
        filters=["w1", "w2", "m2", "bb", "uu", "vv"],
        on_sciserver=True,
        username="",
    ):
        self.source_name = source_name
        self.filters = filters
        self.on_sciserver = on_sciserver
        self.temp_path = f"/home/idies/workspace/Temporary/{username}/scratch/"

        self._make_dirs()
        if not self.on_sciserver:
            raise NotImplementedError("Running on SciServer only for now")
            # warnings.warn(
            # f"<  INFO  > : Not running on SciServer, lots of files will be downloaded to your local machine! Brace yourself!"
            # )
        else:
            from jdaviz import Imviz

            warnings.warn(
                f"<  INFO  > : Running on SciServer, files will be copied to your SciServer Temporary directory"
            )
        print(f"<  INFO  > : Loading TAP services from Heasarc")
        self._load_services()
        print(f"<  INFO  > : Getting coordinates of {self.source_name}")
        self._get_coords()
        print(f"<  INFO  > : Getting observations of {self.source_name}")
        self.observations = self.get_observations()
        self.n_obs = len(self.observations)
        print(f"<  INFO  > : Found {self.n_obs} observations")
        if self.on_sciserver:
            print(f"<  INFO  > : Getting directories of observations")
            self.obs_dirs = self.get_directories_sciserver(self.observations)
        else:
            pass
            # print(f"<  INFO  > : Getting directories of observations")
            # self.obs_dirs = self.get_directories(self.observations)

    def _get_image_by_index(self, index):
        """Get the image of a given observation index"""
        obsid = self.observations["obsid"][index]
        current_dir = f"{self.working_dir}/"
        images = [f"sw{obsid}u{filt}_sk.img.gz" for filt in self.filters]
        # check if the images exist
        located = []
        for image in images:
            if not os.path.exists(f"{self.working_dir}/images/{image}"):
                warnings.warn(f"{image} not found in {self.working_dir}/images/{image}")
            else:
                located.append(f"{current_dir}/images/{image}")
        return located

    def check_all_images(self):
        """Check if all the images are located"""
        for i in range(self.n_obs):
            self._get_image_by_index(i)

    def _make_dirs(self, path=""):
        """Make directories to store the files

        Parameters
        ----------
        path : str
            Path to the directory to be created
        """
        if path == "":
            path = self.temp_path
            # path = f"{os.getcwd()}/"
        os.makedirs(f"{path}{self.source_name}", exist_ok=True)
        os.makedirs(f"{path}{self.source_name}/images", exist_ok=True)
        os.makedirs(f"{path}{self.source_name}/fluxes", exist_ok=True)
        self.working_dir = f"{path}{self.source_name}/"

    def _get_coords(self):
        """
        Get coordinates of source_name from Simbad
        """
        coords = Simbad.query_object(self.source_name)
        ra, dec = coords["RA"][0], coords["DEC"][0]
        print(
            f"<  INFO  > : Coordinates of {self.source_name} are {coords['RA'][0]} {coords['DEC'][0]}"
        )
        self.coord = coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        self.ra = self.coord.ra.deg
        self.dec = self.coord.dec.deg

    def _load_services(self):
        """Load the TAP services from the Heasarc"""
        self.tap_services = vo.regsearch(servicetype="tap", keywords=["heasarc"])
        self.heasarc_tables = self.tap_services[0].service.tables

    def get_observations(
        self,
    ):
        """Given a source, find all the swift observations"""

        query = f"""SELECT name, obsid, start_time, uvot_exposure,xrt_exposure, ra, dec 
            FROM public.swiftmastr as cat 
            where 
            contains(point('ICRS',cat.ra,cat.dec),circle('ICRS',{self.ra},{self.dec},0.4))=1 
            order by cat.start_time
            """
        results = self.tap_services[0].search(query).to_table()
        ascii.write(
            results,
            f"{self.working_dir}{self.source_name}_allobs.txt",
            format="basic",
            overwrite=True,
        )
        return results

    def get_directories_sciserver(self, results):
        """Locate the observation files on the SCIserver"""
        path = "/FTP/swift/data/obs/"

        mjd = Time(results["start_time"], format="mjd")
        dirname = [
            mjd.to_value("iso")[k][:7].replace("-", "_") for k in range(len(mjd))
        ]
        not_found = []
        dirs = []
        # check if the directory exists
        for k in range(len(mjd)):
            if not os.path.exists(path + dirname[k] + "/" + results["obsid"][k]):
                not_found.append(k)
            else:
                dirs.append(path + dirname[k] + "/" + results["obsid"][k])

        # in case the directory is not found show the user the obsid and dirname
        if len(not_found) > 0:
            warnings.warn(f"{len(not_found)} observation files not located see below")
            for k in not_found:
                print(f"{results['obsid'][k]} - {dirname[k]}")

        # save the directories to a file
        with open(f"{self.working_dir}{self.source_name}_dirslist.txt", "w") as f:
            for di in dirs:
                f.write(di + "\n")
        return dirs

    def list_swift_catalogues(self, heasarc_tables):
        """List all the swift catalogues available in the Heasarc"""
        for tablename in heasarc_tables.keys():
            if "swift" in tablename:
                print(
                    " {:20s} {}".format(
                        tablename, heasarc_tables[tablename].description
                    )
                )
        for c in heasarc_tables["swiftmastr"].columns:
            print("{:20s} {}".format(c.name, c.description))

    def copy_files_in_temp(self):
        """Copy the files in the temporary directory of SciServer"""
        # copy the files to the temporary directory
        print(f"<  INFO  > : Copying files to temporary directory")
        no_image = []
        for i in range(self.n_obs):
            print(f"<  INFO  > : Copying files for obs {i+1}/{self.n_obs}", end="\r")
            sys.stdout.flush()
            obsid = self.observations["obsid"][i]
            original_dir = f"{self.obs_dirs[i]}/uvot/image/"
            dest = f"{self.working_dir}/images/{obsid}"

            if not os.path.exists(original_dir):
                print(f"<  INFO  > : No image found for {obsid}")
                no_image.append(obsid)
            else:
                if not os.path.exists(dest):
                    copytree(original_dir, dest)
        print(f"<  INFO  > : Done copying files to temporary directory")
        if len(no_image) > 0:
            warnings.warn(f"{len(no_image)} observations have no images")
            for obsid in no_image:
                print(f"{obsid}")

    def plot_image(self, image):
        """Plot the image of a given observation index and filter"""

        if self.on_sciserver:
            self.imviz = Imviz()
            self.viewer = self.imviz.default_viewer
            self.imviz.load_data(image)
            self.imviz.show()
            # customize the viewer
            plot_options = self.imviz.plugins["Plot Options"]
            plot_options.select_all()

            # Preset
            plot_options.stretch_preset = "Min/Max"

            self.viewer.set_colormap("Gray")
            self.viewer.stretch = "log"
            self.viewer.zoom_level = 1

        else:
            raise NotImplementedError("Plotting only on SciServer for now")

    def set_regions(self, ra_bkg, dec_bkg, radius_bkg=30.0 * u.arcsec):
        """Set the source and background regions for the photometry

        Parameters
        ----------
        ra_bkg : float
            Right ascension of the background region, in degrees
        dec_bkg : float
            Declination of the background region, in degrees
        radius_bkg : float
            Radius of the background region
        """
        ra_bkg = ra_bkg * u.deg
        dec_bkg = dec_bkg * u.deg

        src_reg = CircleSkyRegion(center=self.coord, radius=5 * u.arcsecond)
        bkg_coords = coord.SkyCoord(
            ra=ra_bkg, dec=dec_bkg, unit=(u.deg, u.deg), frame="icrs"
        )

        bkg_reg = CircleSkyRegion(center=bkg_coords, radius=radius_bkg)
        print("< INFO  > : Setting source and background regions")
        self.imviz.load_regions([src_reg, bkg_reg])

        obsid = self.observations["obsid"][i]
        path = f"{self.working_dir}/images/{obsid}"

        regions = imviz.get_interactive_regions()
        print(f"<  INFO  > : Saving regions to {self.working_dir}")
        labels = ["src", "bkg"]
        for filt in self.filters:
            for i, subset in enumerate(range(1, 3)):
                rad = str(regions[f"Subset {subset}"].to_sky(w).radius).replace(
                    " arcsec", '"'
                )
                with open(f"{labels[i]}_{filt}.txt", "w") as fl:
                    fl.write(
                        f'fk5;circle({np.float64(regions[f"Subset {subset}"].to_sky(w).center.ra)},{np.float64(regions[f"Subset {subset}"].to_sky(w).center.dec)},{rad})'
                    )

    def get_flux(self):
        path = f"{self.working_dir}/images/"
        obsid = self.observations["obsid"][i]

        for filt in self.filters:
            image = f"sw{obsid}u{filt}_sk.img.gz"

            res = uvotsource(
                image=f"{self.working_dir}/{image}",
                srcreg=f"{path}/src_{filt}.txt",
                bkgreg=f"{path}/bkg_{filt}.txt",
                outfile=f"{self.working_dir}/fluxes/{obsid}_{filt}.fits",
                sigma=3.0,
                skipreason="",
                chatter=0,
                clobber="true",
            )
            if not res.returncode == 0:
                print(f"<  INFO  > : Error in {image}")
                print(res.stderr)
                print(res.stdout)
                with open(f"{self.working_dir}/fluxes/{obsid}_{filt}.log", "w") as f:
                    f.write(res.stderr)
                    f.write(res.stdout)
