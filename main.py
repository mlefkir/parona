from parona import ObservationNuSTAR
from parona import ObservationXMM
import os

obsid_xmm =  ""
obsid_nu = ""
observ= ""
date =  ""
intervals = ["all"]
src_name = ""

loc = ""

if not os.path.isdir(f'{loc}/{observ}'):
    os.mkdir(f'{loc}/{observ}')

path_nu = loc+"/NuSTAR/"
path_xmm =  loc+"/XMM-Newton/"


#obs_nustar = ObservationNuSTAR(path_nu,obsid_nu, date,slices=intervals)
obs_xmm = ObservationXMM(path_xmm,obsid_xmm,date,slices=intervals)

#obs_nustar.calibrate()
obs_xmm.calibrate()
obs_xmm.gen_evts()
obs_xmm.gen_flarelc()
obs_xmm.gen_gti()
obs_xmm.gen_clean_evts()
obs_xmm.gen_images()
obs_xmm.find_start_time()

#start_time_xmm = obs_xmm.start_time
#plot_gti_nustar_xmm_portions(start_time_xmm,loc)

tag = f"{src_name}_"
obs_xmm.select_regions(tag)
obs_xmm.check_pileup(tag)
obs_xmm.gen_lightcurves(tag)

#obs_nustar.select_regions(tag)
#obs_nustar.gen_lightcurves(tag)
#obs_nustar.gen_spectra(tag,start_nustar=xmm2nu(start_time_xmm))
obs_xmm.gen_spectra(tag)
