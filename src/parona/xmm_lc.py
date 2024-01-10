import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def get_lightcurve(
    src_event_file,
    bkg_event_file,
    scale,
    user_defined_bti=None,
    verbose=False,
    min_Frac_EXP=0.3,
    CCDNR=4,
    input_timebin_size=50,
    PATTERN=4,
    PI=[200, 10000],
    t_clip_start=10,
    t_clip_end=100,
):
    """

    Parameters
    ----------
    src_event_file : str
        Path to the source event file.
    bkg_event_file : str
        Path to the background event file.
    scale : float
        Scale factor to apply to the background.
    user_defined_bti : array
        User defined bad time intervals.
    verbose : bool
        Verbose mode.
    min_Frac_EXP : float
        Minimum fraction of exposure to consider a bin good.
    CCDNR : int
        CCD number. Default is 4 for pn.
    input_timebin_size : float
        Time bin size in seconds.
    PATTERN : int
        Pattern to use. Default is <=4.
    PI : list
        PI range to use. Default is [200,10000].
    t_clip_start : float
        Time to clip the start of the light curve in seconds. Default is 10.
    t_clip_end : float
        Time to clip the end of the light curve in seconds. Default is 100.


    Returns
    -------
    t : array
        Array of the time at the start of each time bin
    net : array
        Array of the net counts in each time bin
    err : array
        Array of the error on the net counts in each time bin
    bg : array
        Array of the background counts in each time bin
    bg_err : array
        Array of the error on the background counts in each time bin
    T0 : float
        Starting time of the observation

    """
    # input_timebin_size  in s is the time bin size for the light curve chosen by the user
    CCDNR = f"{CCDNR:02d}"

    hdu = fits.open(src_event_file)

    FRAME_TIME = hdu[f"STDGTI{CCDNR}"].header["FRMTIME"]  # in ms
    # in practice we want a timebinsize which is a multiple of the frame time
    N_frames_per_bin = np.ceil(input_timebin_size * 1000 / FRAME_TIME).astype(
        int
    )  # integer number of frames per bin
    timebin = N_frames_per_bin * FRAME_TIME / 1000  # s
    if verbose:
        print(f"N_frames_per_bin = {N_frames_per_bin}")
    if verbose:
        print(f"timebin = {timebin} s")

    times = []
    t_bins = []
    counts = []
    btis = []
    frac_exposures = []
    clean_frac_exposures = []
    T0 = 0

    for iter, item in enumerate([src_event_file, bkg_event_file]):
        hdu = fits.open(item)
        if verbose:
            print(f'Number of events = {len(hdu["EVENTS"].data)}')
        event_table = hdu["EVENTS"].data

        # mask for the pattern and the energy
        mask_PATTERN = event_table["PATTERN"] <= PATTERN
        mask_energy = (PI[0] < event_table["PI"]) & (event_table["PI"] < PI[1])
        mask = mask_PATTERN & mask_energy

        # apply the mask
        event_table = event_table[mask]
        if verbose:
            print(f"Number of events after filtering = {len(event_table)}")
        # get the time column
        time = event_table["TIME"]
        # clip the start and the end of the light curve
        if iter == 0:  # for the source
            t_start = time[0] + t_clip_start
            t_end = time[-1] - t_clip_end
            bin_list = np.arange(t_start, t_end, timebin)
        # get the counts in each bin
        count, t_bin = np.histogram(
            time, bins=bin_list, range=(t_start, t_end)
        )  # counts and time at the start of each bin
        if verbose:
            print(f"Number of bins = {len(t_bin)-1}")
        t = t_bin[:-1]  # remove the last element of the array

        T0 = t[0]
        t_bin = t_bin - T0

        ## get the GTIs
        GTI = hdu[f"STDGTI{CCDNR}"].data
        if verbose:
            print(f"Number of GTIs = {len(GTI)}")
        if len(GTI) == 1:
            BTI = np.array([[T0, GTI["START"][0] - T0], [GTI["STOP"][0] - T0, t[-1]]])
        else:
            BTI = get_bad_time_intervals(GTI, T0)
        Frac_EXP = calculate_Frac_EXP(t_bin, BTI)

        if np.any(Frac_EXP > 1) or np.any(Frac_EXP < 0):
            raise ValueError("Frac_EXP should be between 0 and 1")

        clean_Frac_EXP = np.where(Frac_EXP < min_Frac_EXP, 0, Frac_EXP)
        clean_frac_exposures.append(clean_Frac_EXP)

        times.append(t - T0)
        counts.append(count)
        frac_exposures.append(Frac_EXP)
        btis.append(BTI)
        t_bins.append(t_bin)

    plot_raw_lc(times[0], counts, frac_exposures, btis, filename="raw_lc")

    # get the user defined bad time intervals
    clean_Frac_EXP = np.minimum(clean_frac_exposures[0], clean_frac_exposures[1])
    if user_defined_bti is not None:
        frac_exp = calculate_Frac_EXP(t_bin, user_defined_bti)
        clean_Frac_EXP = np.minimum(clean_Frac_EXP, frac_exp)
        plot_raw_lc(
            times[0],
            counts,
            [frac_exp, frac_exp],
            [user_defined_bti, user_defined_bti],
            filename="user_lc",
        )

    # np.savetxt('frac_exp.txt',clean_Frac_EXP)

    # get the cleaned light curves and variance by rescaling the exposures
    cleaned_src = np.divide(
        counts[0].astype(float),
        clean_Frac_EXP,
        out=np.zeros_like(counts[0].astype(float)),
        where=clean_Frac_EXP != 0,
    )
    cleaned_src2 = np.divide(
        counts[0].astype(float),
        clean_Frac_EXP**2,
        out=np.zeros_like(counts[0].astype(float)),
        where=clean_Frac_EXP != 0,
    )

    cleaned_bkg = np.divide(
        counts[1].astype(float),
        clean_Frac_EXP,
        out=np.zeros_like(counts[1].astype(float)),
        where=clean_Frac_EXP != 0,
    )
    cleaned_bkg2 = np.divide(
        counts[1].astype(float),
        clean_Frac_EXP**2,
        out=np.zeros_like(counts[1].astype(float)),
        where=clean_Frac_EXP != 0,
    )

    if verbose:
        print(
            f"Min counts/bin in the src lc:",
            np.min(np.delete(cleaned_src, np.where(cleaned_src <= 0))),
        )

    src = [cleaned_src, cleaned_src2]
    bkg = [cleaned_bkg, cleaned_bkg2]

    dt = timebin
    # remove the bins with zero counts
    zeros_index = np.where(src[0] <= 0)

    src[0] = np.delete(src[0], zeros_index)
    src[1] = np.delete(src[1], zeros_index)
    bkg[0] = np.delete(bkg[0], zeros_index)
    bkg[1] = np.delete(bkg[1], zeros_index)
    t = np.delete(times[0], zeros_index)

    if verbose:
        print(f"Number of bins = {len(t)}")

    # scale the background
    bkg[0] = bkg[0] * scale
    bkg[1] = bkg[1] * scale**2

    # get the net counts and error
    net = (src[0] - bkg[0]) / dt
    err = np.sqrt(src[1] + bkg[1]) / dt
    bg = bkg[0] / dt
    bg_err = np.sqrt(bkg[1]) / dt

    return t, net, err, bg, bg_err, t_bin, clean_Frac_EXP, T0


def get_bad_time_intervals(gtis, T0):
    """Compute the bad time intervals from the good time intervals

    Parameters
    ----------
    gtis : array
        Good time intervals, with columns START and STOP
    T0 : float
        Starting time of the observation
    """

    n_gtis = len(gtis)
    j = 0
    if n_gtis > 1:
        btis = np.zeros((n_gtis - 1, 2))
        for j in range(n_gtis - 1):
            btis[j, 0] = gtis["STOP"][j] - T0
            btis[j, 1] = gtis["START"][j + 1] - T0

        return btis
    else:
        print("Only one good time interval")


def calculate_Frac_EXP(t_bin, btis):
    """Calculate the fraction of the time bin which is not in the Bad Time Interval

    Parameters
    ----------
    t_bin : array
        Array of the time at the start of each time bin, including the end of the last time bin
    btis : array
        Array of the start and stop time of each Bad Time Interval

    Returns
    -------
    Frac_EXP : array
        Array of the fraction of the time bin which is not in the Bad Time Interval

    """
    timebin = t_bin[1] - t_bin[0]
    Frac_EXP = np.ones(len(t_bin)) * timebin
    nbtis = len(btis)
    for i in range(nbtis):
        bti_start = np.searchsorted(t_bin, btis[i][0]) - 1
        bti_end = np.searchsorted(t_bin, btis[i][1]) - 1
        bti_start = max(0, bti_start)

        if bti_end >= 0:
            size = bti_end - bti_start

            if size == 0:
                Frac_EXP[bti_start] = Frac_EXP[bti_start] - (btis[i][1] - btis[i][0])
            if size > 1:
                Frac_EXP[bti_start + 1 : bti_end] = 0
            if size > 0:
                Frac_EXP[bti_start] = Frac_EXP[bti_start] - (
                    t_bin[bti_start + 1] - btis[i][0]
                )
                Frac_EXP[bti_end] = Frac_EXP[bti_end] - (btis[i][1] - t_bin[bti_end])

    Frac_EXP = Frac_EXP[:-1]
    Frac_EXP = Frac_EXP / timebin
    return Frac_EXP


def plot_raw_lc(t, y, fr, btis, filename):
    cols = ["C1", "C4"]
    labels = ["source", "background"]
    for k in range(2):
        fig, ax = plt.subplots(2, 1, figsize=(15, 7.5))
        ax[0].step(t, y[k], color=cols[k], label=labels[k], where="post")
        # ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel("Count rate (cts/s)")
        ax[0].legend()
        ax[0].set_title("Light curve (Raw)")

        ax[1].step(t, fr[k], color=cols[k], label=labels[k], where="post")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Frac EXP")
        ax[1].legend()
        ax[1].set_title("Fraction of exposure")

        ax[1].sharex(ax[0])
        for i in range(2):
            for bti in btis[k]:
                ax[i].axvspan(bti[0], bti[1], alpha=0.15, color="C3")
        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig(f"{filename}_{labels[k]}.png", dpi=300)
