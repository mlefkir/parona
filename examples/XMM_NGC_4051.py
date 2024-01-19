from parona import ObservationXMM
import os
import matplotlib.pyplot as plt
plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs_colblind.mplstyle")

src_name = 'NGC_4051'
if not os.path.isdir(src_name):
    os.mkdir(src_name)
path_xmm = os.getcwd()+f'/{src_name}'
obsid_xmm = ['0606320101','0606320201','0606320301','0606320401','0606321301','0606321401',
             '0606321501','0606321601','0606321701','0606321801','0606321901','0606322001',
             '0606322101','0606322201','0606322301','0830430201','0830430801']

for ID in obsid_xmm:
    obs_xmm = ObservationXMM(path_xmm,ID,instruments=['EPN'])
    obs_xmm.gen_evts(with_MOS=False)
    obs_xmm.select_regions(src_name)
    user_defined_bti = None
    obs_xmm.gen_lightcurves_manual(src_name,binning=150,
                                   user_defined_bti=user_defined_bti,
                                   PI=[[300,10000],[300,1500],[1500,10000]],
                                   PATTERN=4,CCDNR=4)
