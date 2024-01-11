import os
import sys
import glob
import requests
from distutils.dir_util import copy_tree
import warnings

# astropy and astroquery
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits,ascii
from astropy.wcs import WCS
from astropy.time import Time
from astroquery.simbad import Simbad


# heasoft 
import heasoftpy as hsp
import pyvo as vo
# Ignore unimportant warnings
warnings.filterwarnings('ignore', '.*Unknown element mirrorURL.*', 
                        vo.utils.xml.elements.UnknownElementWarning)

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
    ):
        self.source_name = source_name
        self.filters = filters
        self.on_sciserver = on_sciserver
        self._make_dirs()
        if not self.on_sciserver:
            warnings.warn(
                f"<  INFO  > : Not running on SciServer, lots of files will be downloaded to your local machine! Brace yourself!"
            )
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
        if self.on_sciserver:
            print(f"<  INFO  > : Getting directories of observations")
            self.obs_dirs = self.get_directories_sciserver(self.observations)

    def _make_dirs(self,path=""):
        """Make directories to store the files
        
        Parameters
        ----------
        path : str
            Path to the directory to be created
        """
        if path=="": path = f"{os.getcwd()}/"
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
        ascii.write(results,f"{self.working_dir}{self.source_name}_allobs.txt",format="basic")
        return results

    def get_directories_sciserver(self, results):
        """Locate the observation files on the SCIserver
        
        """
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
        with open(f"{self.working_dir}{self.source_name}_dirslist.txt","w") as f:
            for di in dirs:
                f.write(di+"\n")
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