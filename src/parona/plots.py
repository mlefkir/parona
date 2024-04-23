from astropy.io import fits
from matplotlib.patches import Rectangle
from .converters import xmm2nu,nu2xmm

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")


def plot_gti_nustar_xmm_portions(obs_xmm,obs_nustar,T0,path,intervals=None):
    

    fig, ax = plt.subplots(len(obs_xmm.instruments),1,figsize=(12,1.65*len(obs_xmm.instruments)), sharex=True)
    if len(obs_xmm.instruments) == 1 : ax = [ax]
    xmm_col = "#4BB3FD" 
    nu_col = "#FFA600"

    for c,instr in enumerate(obs_xmm.instruments) :

        # hdu_list = fits.open(f"{}_{instr}_lc_clean_bin100_3.0-10.0.fits")  
        # lc_data = Table(hdu_list[1].data)
        # ax[c].errorbar(lc_data["TIME"], lc_data["RATE"]/np.max(np.nan_to_num(lc_data["RATE"],-1)), yerr=lc_data["ERROR"], fmt="o",color="blue", markersize=5., label=instr)

        #-- plot xmm gti --
        gti = fits.open(obs_xmm.obs_files[instr]["gti"])
        x = np.sort(np.concatenate((gti[1].data["START"],gti[1].data["STOP"])))
        ax[c].vlines(x-T0,ymin=0,ymax=1,color="r")
        for i in range(len(gti[1].data["START"])-1):
            ax[c].add_patch(Rectangle((gti[1].data["START"][i]-T0, 0),gti[1].data["STOP"][i] -gti[1].data["START"][i],1,edgecolor = None,alpha=0.55,facecolor = xmm_col,fill=True))
        ax[c].set_ylabel(instr)
        ax[c].add_patch(Rectangle((gti[1].data["START"][i+1]-T0, 0),gti[1].data["STOP"][i+1] -gti[1].data["START"][i+1],1,edgecolor = None,alpha=0.55,facecolor = xmm_col,fill=True,label="XMM-Newton"))


        #-- plot nustar gti --
        gti = fits.open(obs_nustar.obs_files["FPMA"]["gti"])
        x = np.sort(np.concatenate((gti["GTI"].data["START"],gti["GTI"].data["STOP"])))
        x = nu2xmm(x)-T0
        ax[c].vlines(x,ymin=0,ymax=1,color="b")
        for i in range(len(gti["GTI"].data["START"])-1):
            ax[c].add_patch(Rectangle((nu2xmm(gti["GTI"].data["START"][i])-T0, 0),gti["GTI"].data["STOP"][i] -gti["GTI"].data["START"][i],1,edgecolor = None,alpha=0.55,facecolor = nu_col,fill=True))
        ax[c].add_patch(Rectangle((nu2xmm(gti["GTI"].data["START"][i+1])-T0, 0),gti["GTI"].data["STOP"][i+1] -gti["GTI"].data["START"][i+1],1,edgecolor = None,alpha=0.55,facecolor = nu_col,fill=True,label="NuSTAR"))

        ax[c].set_ylabel(instr)
        ax[c].set_yticks([])
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['left'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        L = []
        #-- plot intervals --
        ax[c].tick_params(axis="x", direction="in", color="k")

    #     if c==0: length=1.4
    #     else : length = 1.2
    #     if intervals != None:
    #         for part,inter in enumerate(intervals):
    #             ax[c].axvline(x=inter[0]*1e3,ymin=0,ymax=length,color="black",linestyle="--",linewidth=2.5, clip_on=False)
    #             L.append(inter[0]*1e3)
    #             if c==0: 
    #                 ax[c].text((inter[0]+inter[1])*0.5e3-3e3,1.1,f"Part " + r'''\raisebox{.5pt}{\textcircled{\raisebox{-2pt} {'''+str(part+1)+"}}}")
    #         L.append(inter[0]*1e3)
    #         ax[c].set_xticks(L)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9,bottom=0.25,hspace=0.2)
    leg = ax[-1].legend(title=r"Good Time Intervals :",bbox_to_anchor=(0.35, -0.55, 0.5, .2),ncol=2,frameon=False)
    leg.get_title().set_position((-285, -27))
    ax[-1].set_xlabel("Observation time (s)")

    fig.savefig(f'{path}/simultaneous_XMMNUSTAR.pdf')