from parona import ObservationXMM
import os
import matplotlib.pyplot as plt
plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")


src_name = 'ESO_141-G55'
if not os.path.isdir(src_name):
    os.mkdir(src_name)
path_xmm = os.getcwd()+f'/{src_name}'

obsid_xmm = ["0503750101",
"0503750501","0913190101"]

for ID in obsid_xmm:
    obs_xmm = ObservationXMM(path_xmm,ID,instruments=['EPN'])
    obs_xmm.gen_evts(with_MOS=False)
    obs_xmm.gen_images()
    obs_xmm.select_regions(src_name)
    obs_xmm.check_pileup(src_name)
    user_defined_bti = None
    obs_xmm.gen_lightcurves_manual(src_name,binning=150,
                                   user_defined_bti=user_defined_bti,
                                   PI=[[300,10000],[300,1500],[1500,10000]],
                                   PATTERN=4,CCDNR=4)