import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .utils import energy2nustarchannel, nustarchannel2energy


def get_lightcurve_add_NuSTAR(
    src_event_files,
    bkg_event_files,
    scales,
    user_defined_bti=None,
    verbose=False,
    min_Frac_EXP=0.3,
    input_timebin_size=50,
    PATTERN=4,
    PI=[200, 10000],
    t_clip_start=10,
    t_clip_end=100,
    suffix="",
):
    """

    Parameters
    ----------
    src_event_file : list of str
        Paths to the source event files.
    bkg_event_file : str or list of str
        Paths to the background event files.
    scales : list of float
        Scale factor to apply to the background.
    user_defined_bti : array
        User defined bad time intervals.
    verbose : bool
        Verbose mode.
    min_Frac_EXP : float
        Minimum fraction of exposure to consider a bin good.
    CCDNR : int,str
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
    CCDNR = ""

    hdu = [fits.open(src_evt) for src_evt in src_event_files]

    FRAME_TIME = 0.1  # in ms
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

    # find the first instrument to start the observation
    first_instr = 0
    t_start = [
        fits.open(src_event_files[i])["EVENTS"].header["TSTART"]
        for i in range(len(src_event_files))
    ]
    t_stop = [
        fits.open(src_event_files[i])["EVENTS"].header["TSTOP"]
        for i in range(len(src_event_files))
    ]
    # define the start and end of the observation as the minimum and maximum of the start and stop times
    t0 = np.max(t_start)
    tm = np.min(t_stop)
    t_start = t0 + t_clip_start
    t_end = tm - t_clip_end
    bin_list = np.arange(t_start, t_end, timebin)

    for iter, item in enumerate(src_event_files + bkg_event_files):
        hdu = fits.open(item)
        if verbose:
            print(f'Number of events = {len(hdu["EVENTS"].data)}')
        event_table = hdu["EVENTS"].data

        # mask for the pattern and the energy
        pattern_name = "GRADE"

        mask_PATTERN = event_table[pattern_name] <= PATTERN
        mask_energy = (PI[0] < event_table["PI"]) & (event_table["PI"] < PI[1])
        mask = mask_PATTERN & mask_energy

        # apply the mask
        event_table = event_table[mask]
        if verbose:
            print(f"Number of events after filtering = {len(event_table)}")
        # get the time column
        time = event_table["TIME"]

        # get the counts in each bin
        count, t_bin = np.histogram(
            time, bins=bin_list, range=(t_start, t_end)
        )  # counts and time at the start of each bin
        if verbose:
            print(f"Number of bins = {len(t_bin)-1}")
        t = t_bin[:-1]  # remove the last element of the array

        T0 = t[0]
        if verbose:
            print(f"T0 = {T0}")
        t_bin = t_bin - T0

        ## get the GTIs
        GTI = hdu[f"STDGTI{CCDNR}"].data
        if verbose:
            print(f"Number of GTIs = {len(GTI)}")
            # print(f"GTI start = {GTI['START']}")
            # print(f"GTI stop = {GTI['STOP']}")
            # print(f"GTI start - T0 = {GTI['START'] - T0}")
        if len(GTI) == 1:
            print("Only one good time interval")
            if GTI["START"][0] < T0:
                print("The start of the GTI is before the start of the observation")
                if GTI["STOP"][0] > t[-1]:
                    print("The GTI is longer than the observation")
                    BTI = None
                else:
                    BTI = np.array([[GTI["STOP"][0] - T0, t[-1] - T0]])
            else:
                BTI = np.array(
                    [[T0, GTI["START"][0] - T0], [GTI["STOP"][0] - T0, t[-1] - T0]]
                )
        else:
            BTI = get_bad_time_intervals(GTI, T0)
        if BTI is None:
            Frac_EXP = np.ones(len(t))
        else:
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

    # for i, label in enumerate(["FPMA", "FPMB"]):

    #     if btis[i] is None and btis[i + 2] is None:
    #         btis_ = None
    #     else:
    #         btis_ = [btis[i], btis[i + 2]]
    #     plot_raw_lc(
    #         times[i],
    #         [counts[i], counts[i + 2]],
    #         [frac_exposures[i], frac_exposures[i + 2]],
    #         btis_,
    #         filename=f"raw_{label}_lc_{nustarchannel2energy(PI[0])}-{nustarchannel2energy(PI[1])}{suffix}",
    #     )

    src = []
    bkg = []
    src2 = []
    bkg2 = []
    clean_Frac_EXPs = []

    # get the user defined bad time intervals
    for i in range(2):
        clean_Frac_EXP = np.minimum(
            clean_frac_exposures[i], clean_frac_exposures[i + 2]
        )

        if user_defined_bti is not None:
            raise NotImplementedError(
                "User defined bad time intervals not implemented for NuSTAR"
            )

        # get the cleaned light curves and variance by rescaling the exposures
        cleaned_src = np.divide(
            counts[i].astype(float),
            clean_Frac_EXP,
            out=np.zeros_like(counts[i].astype(float)),
            where=clean_Frac_EXP != 0,
        )
        cleaned_src2 = np.divide(
            counts[i].astype(float),
            clean_Frac_EXP**2,
            out=np.zeros_like(counts[i].astype(float)),
            where=clean_Frac_EXP != 0,
        )

        cleaned_bkg = np.divide(
            counts[i + 2].astype(float),
            clean_Frac_EXP,
            out=np.zeros_like(counts[i + 2].astype(float)),
            where=clean_Frac_EXP != 0,
        )
        cleaned_bkg2 = np.divide(
            counts[i + 2].astype(float),
            clean_Frac_EXP**2,
            out=np.zeros_like(counts[i + 2].astype(float)),
            where=clean_Frac_EXP != 0,
        )

        plot_raw_lc(
            times[i],
            [counts[i], counts[i + 2]],
            [clean_Frac_EXP, clean_Frac_EXP],
            None,
            filename=f"user_lc_cleaned_{nustarchannel2energy(PI[0])}-{nustarchannel2energy(PI[1])}{suffix}",
        )

        if verbose:
            print(
                f"Min counts/bin in the src lc:",
                np.min(np.delete(cleaned_src, np.where(cleaned_src <= 0))),
            )

        src.append(cleaned_src)
        src2.append(cleaned_src2)
        bkg.append(cleaned_bkg)
        bkg2.append(cleaned_bkg2)
        clean_Frac_EXPs.append(clean_Frac_EXP)

    print()
    dt = timebin
    print(f"Number of bins before = {len(times)}")
    print(f"number of bins in src FPMA = {len(src[0])}")
    print(f"number of bins in src FPMB = {len(src[1])}")

    # remove the bins with zero counts
    zeros_index_FPMA = np.where(src[0] <= 0)[0]
    zeros_index_FPMB = np.where(src[1] <= 0)[0]
    mask = np.unique(np.concatenate([zeros_index_FPMA, zeros_index_FPMB]))

    src[0] = np.delete(src[0], mask)
    src[1] = np.delete(src[1], mask)
    bkg[0] = np.delete(bkg[0], mask)
    bkg[1] = np.delete(bkg[1], mask)
    src2[0] = np.delete(src2[0], mask)
    src2[1] = np.delete(src2[1], mask)
    bkg2[0] = np.delete(bkg2[0], mask)
    bkg2[1] = np.delete(bkg2[1], mask)
    # mask = np.unique(np.concatenate([zeros_index_FPMA.flatten(), zeros_index_FPMB.flatten()]))

    t = np.delete(times[0], mask)

    if verbose:
        print(f"Number of bins after = {len(t)}")

    # scale the background
    bkg[0] = bkg[0] * scales[0]
    bkg2[0] = bkg2[0] * scales[0] ** 2
    bkg[1] = bkg[1] * scales[1]
    bkg2[1] = bkg2[1] * scales[1] ** 2

    # get the net counts and error
    net = (src[0] + src[1] - bkg[0] - bkg[1]) / dt
    err = np.sqrt(src2[0] + src2[1] + bkg2[0] + bkg2[1]) / dt
    bg = (bkg[0] + bkg[1]) / dt
    bg_err = np.sqrt(bkg2[0] + bkg2[1]) / dt

    return t, net, err, bg, bg_err, t_bin, clean_Frac_EXPs, T0


def get_lightcurve(
    src_event_file,
    bkg_event_file,
    scale,
    user_defined_bti=None,
    verbose=False,
    min_Frac_EXP=0.3,
    CCDNR=4,
    CCDNR_bkg=4,
    input_timebin_size=50,
    bin_list=None,
    timebin=None,
    PATTERN=4,
    PI=[200, 10000],
    t_clip_start=10,
    t_clip_end=100,
    suffix="",
    is_nustar=False,
    instr="",
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
    CCDNR : int,str
        CCD number. Default is 4 for pn.
    CCDNR_bkg : int,str
        CCD number for the background. Default is 4 for pn.
    input_timebin_size : float
        Time bin size in seconds.
    bin_list: array
        Array of the time at the start of each time bin
    timebin : float
        Time bin size in seconds. Default is None as it is computed from the input_timebin_size.
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
    if isinstance(CCDNR, int):
        curr_CCDNR = f"{CCDNR:02d}"
    if is_nustar:
        assert CCDNR == "", "CCDNR should be empty for NuSTAR"

    hdu = fits.open(src_event_file)
    if not is_nustar:
        FRAME_TIME = hdu[f"STDGTI{curr_CCDNR}"].header["FRMTIME"]  # in ms
    # in practice we want a timebinsize which is a multiple of the frame time
    else:
        FRAME_TIME = 0.1  # in ms
    N_frames_per_bin = np.ceil(input_timebin_size * 1000 / FRAME_TIME).astype(
        int
    )  # integer number of frames per bin
    if timebin is None:
        timebin = N_frames_per_bin * FRAME_TIME / 1000  # s
    else:
        print("\t\t<! WARNING !> Using the user defined timebin")
    if verbose:
        print(f"\t\t\tN_frames_per_bin = {N_frames_per_bin}")
    if verbose:
        print(f"\t\t\ttimebin = {timebin} s")

    times = []
    t_bins = []
    counts = []
    btis = []
    frac_exposures = []
    clean_frac_exposures = []
    T0 = 0

    for iter, item in enumerate([src_event_file, bkg_event_file]):

        # set the CCDNR for the background or the source
        if iter == 0:
            curr_CCDNR = f"{CCDNR:02d}"
        else:
            curr_CCDNR = f"{CCDNR_bkg:02d}"

        hdu = fits.open(item)
        if verbose:
            print(f'\t\t\tNumber of events = {len(hdu["EVENTS"].data)}')
        event_table = hdu["EVENTS"].data

        # mask for the pattern and the energy
        if is_nustar:
            pattern_name = "GRADE"
        else:
            pattern_name = "PATTERN"
        mask_PATTERN = event_table[pattern_name] <= PATTERN
        mask_energy = (PI[0] < event_table["PI"]) & (event_table["PI"] < PI[1])
        mask = mask_PATTERN & mask_energy

        # apply the mask
        event_table = event_table[mask]
        if verbose:
            print(f"\t\t\tNumber of events after filtering = {len(event_table)}")
        # get the time column
        time = event_table["TIME"]
        # clip the start and the end of the light curve
        if iter == 0:  # for the source
            if bin_list is not None:
                print("\t\t<! WARNING !> Using the user defined bin_list")
                t_start = bin_list[0]
                t_end = bin_list[-1]
                bin_list = bin_list
            else:
                t_start = time[0] + t_clip_start
                t_end = time[-1] - t_clip_end
                bin_list = np.arange(t_start, t_end, timebin)

        # get the counts in each bin
        count, t_bin = np.histogram(
            time, bins=bin_list, range=(t_start, t_end)
        )  # counts and time at the start of each bin
        if verbose:
            print(f"\t\t\tNumber of bins = {len(t_bin)-1}")
            # print(f"t_bin[0]:{t_bin[0]}, time[0]:{time[0]},")
        t = np.copy(t_bin)[:-1]  # remove the last element of the array

        T0 = t[0]
        if verbose:
            print(f"T0 = {T0}")
        t_bin = t_bin - T0

        ## get the GTIs
        GTI = hdu[f"STDGTI{curr_CCDNR}"].data
        if verbose:
            print(f"\t\t\tNumber of GTIs = {len(GTI)}")
            # print(f"GTI start = {GTI['START']}")
            # print(f"GTI stop = {GTI['STOP']}")
            # print(f"GTI start - T0 = {GTI['START'] - T0}")
        if len(GTI) == 1:
            print("Only one good time interval")
            if GTI["START"][0] < T0:
                print("The start of the GTI is before the start of the observation")
                if GTI["STOP"][0] > t[-1]:
                    print("\t\t\tThe GTI is longer than the observation")
                    BTI = None
                else:
                    BTI = np.array([[GTI["STOP"][0] - T0, t[-1] - T0]])
            else:
                BTI = np.array(
                    [[T0, GTI["START"][0] - T0], [GTI["STOP"][0] - T0, t[-1] - T0]]
                )
        else:
            BTI = get_bad_time_intervals(GTI, T0)
        if BTI is None:
            Frac_EXP = np.ones(len(t))
        else:
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
    if btis[0] is None and btis[1] is None:
        btis = None
    plot_raw_lc(
        times[0],
        counts,
        frac_exposures,
        btis,
        filename=f"raw_lc_{instr}_{PI[0]/1000}-{PI[1]/1000}{suffix}",
    )
    # get the user defined bad time intervals
    clean_Frac_EXP = np.minimum(clean_frac_exposures[0], clean_frac_exposures[1])
    if user_defined_bti is not None:
        frac_exp = calculate_Frac_EXP(t_bin, user_defined_bti, user_btis=True)
        clean_Frac_EXP = np.minimum(clean_Frac_EXP, frac_exp)
        plot_raw_lc(
            times[0],
            counts,
            [frac_exp, frac_exp],
            [user_defined_bti, user_defined_bti],
            filename=f"user_lc_{instr}_{PI[0]/1000}-{PI[1]/1000}",
        )

        clean_Frac_EXP = np.where(clean_Frac_EXP < min_Frac_EXP, 0, clean_Frac_EXP)
        # clean_frac_exposures.append(clean_Frac_EXP)

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

    plot_raw_lc(
        times[0],
        counts,
        [clean_Frac_EXP, clean_Frac_EXP],
        None,
        filename=f"user_lc_cleaned_{instr}_{PI[0]/1000}-{PI[1]/1000}{suffix}",
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
    # zeros_index = np.where(src[0] <= 0)

    # src[0] = np.delete(src[0], zeros_index)
    # src[1] = np.delete(src[1], zeros_index)
    # bkg[0] = np.delete(bkg[0], zeros_index)
    # bkg[1] = np.delete(bkg[1], zeros_index)
    # t = np.delete(times[0], zeros_index)

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

    return times[0], net, err, bg, bg_err, t_bin, clean_Frac_EXP, T0, bin_list, timebin


def combine_lightcurves(times, count_rates, errors, bkg_counts, bkg_errors, T0):
    print("Combining light curves")
    n = len(times)
    assert n == len(
        count_rates
    ), "The number of time arrays and count rate arrays should be the same"
    assert n == len(T0)
    # first check that the time arrays are the same
    for i in range(1, n):
        if not np.allclose(times[0], times[i]):
            raise ValueError("The time arrays are not the same")
        if not np.allclose(T0[0], T0[i]):
            raise ValueError("The starting times are not the same")
    dt = times[0][1] - times[0][0]
    net = np.zeros_like(times[0])
    err = np.zeros_like(times[0])
    bkg = np.zeros_like(times[0])
    bkg_err = np.zeros_like(times[0])
    
    zeros_index = count_rates[0] <= 0

    for i in range(n):
        net += count_rates[i]
        err += (errors[i] * dt) ** 2
        bkg += bkg_counts[i]
        bkg_err += (bkg_errors[i] * dt) ** 2
        zeros_index = zeros_index|(count_rates[i] <= 0)

    net = net[~zeros_index]
    err = np.sqrt(err)[~zeros_index]/dt
    bkg = bkg[~zeros_index]
    bkg_err = np.sqrt(bkg_err)[~zeros_index]/dt
    t = times[0][~zeros_index]
    
    # err = np.sqrt(err) / dt
    # bkg_err = np.sqrt(bkg_err) / dt
    return t, net, err, bkg, bkg_err, T0[0]


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


def calculate_Frac_EXP(t_bin, btis, user_btis=False):
    """Calculate the fraction of the time bin which is not in the Bad Time Interval

    Parameters
    ----------
    t_bin : array
        Array of the time at the start of each time bin, including the end of the last time bin
    btis : array
        Array of the start and stop time of each Bad Time Interval
    user_btis : bool
        If True, the btis are given by the user. Default is False.

    Returns
    -------
    Frac_EXP : array
        Array of the fraction of the time bin which is not in the Bad Time Interval

    """
    timebin = t_bin[1] - t_bin[0]
    Frac_EXP = np.ones(len(t_bin)) * timebin
    nbtis = len(btis)
    for i in range(nbtis):
        if user_btis:
            lower = np.floor(btis[i][0] / timebin) * timebin
            higher = np.ceil(btis[i][1] / timebin) * timebin
        else:
            lower = btis[i][0]
            higher = btis[i][1]
        bti_start = np.searchsorted(t_bin, lower) - 1
        bti_end = np.searchsorted(t_bin, higher) - 1
        bti_start = max(0, bti_start)

        if bti_end >= 0:
            size = bti_end - bti_start

            if size == 0:
                Frac_EXP[bti_start] = Frac_EXP[bti_start] - (higher - lower)
            if size > 1:
                Frac_EXP[bti_start + 1 : bti_end] = 0
            if size > 0:
                Frac_EXP[bti_start] = Frac_EXP[bti_start] - (
                    t_bin[bti_start + 1] - lower
                )
                Frac_EXP[bti_end] = Frac_EXP[bti_end] - (higher - t_bin[bti_end])

    Frac_EXP = Frac_EXP[:-1]
    Frac_EXP = Frac_EXP / timebin
    return Frac_EXP


def plot_raw_lc(t, y, fr, btis, filename):
    """Plot the raw light curve and the fraction of exposure

    Parameters
    ----------
    t : array
        Array of the time at the start of each time bin
    y : array
        Array of the counts in each time bin
    fr : array
        Array of the fraction of exposure in each time bin
    btis : array
        Array of the start and stop time of each Bad Time Interval
    filename : str
        Name of the file to save the plot
    """
    cols = ["C1", "C4"]
    labels = ["source", "background"]
    for k in range(2):
        fig, ax = plt.subplots(2, 1, figsize=(15, 7.5))
        ax[0].step(t, y[k], color=cols[k], label=labels[k], where="post")
        ax[0].set_ylabel("Counts")
        ax[0].legend()
        ax[0].set_title("Light curve (Raw)")

        ax[1].step(t, fr[k], color=cols[k], label=labels[k], where="post")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Frac EXP")
        ax[1].legend()
        ax[1].set_title("Fraction of exposure")

        ax[1].sharex(ax[0])
        if btis is not None:
            for i in range(2):
                for bti in btis[k]:
                    ax[i].axvspan(bti[0], bti[1], alpha=0.15, color="C3")
        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig(f"{filename}_{labels[k]}.png", dpi=300)
        plt.close(fig)
