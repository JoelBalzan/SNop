import numpy as np
import os
from ..utils.globalpars import *


def generate_dynspec(t_ser, n):
	#	Creates a dynamic spectrum at the highest time resolution from the given voltage time series.
    #
    #	Inputs		input time series of voltages
    #				number of channels
    #
    #	Returns		dynamic spectrum of voltages
    
    dynspec = np.zeros((int(t_ser.shape[0] / n), n), dtype=np.complex64)
    for i in range(int(t_ser.shape[0] / n)):
        dynspec[i, :] = np.fft.fft(t_ser[i * n : (i + 1) * n])

    return(dynspec)

def voltage_ds(frbname, X, Y, dm, nchan, outdir):

	#	Construct voltage dynamic spectra for X and Y

	dslen		=	0
	
	if(os.path.exists(X) and os.path.exists(Y)):
		xvdat		=	np.load(X,mmap_mode='r')
		yvdat		=	np.load(Y,mmap_mode='r')
		xvdspec		=	generate_dynspec(xvdat, nchan)
		yvdspec		=	generate_dynspec(yvdat, nchan)
		#np.save("{}reduced/{}_x_vds_{}_{}.npy".format(outdir,frbname,dm,nchan),xvdspec.T)
		#np.save("{}reduced/{}_y_vds_{}_{}.npy".format(outdir,frbname,dm,nchan),yvdspec.T)
		dslen		=	xvdspec.shape[0]
		del(xvdat)
		del(yvdat)
	else:
		print("voltage_ds ---- PANIC - File(s) not found!")
		print("\n******* Have you polcaled ? ******\n")

	return xvdspec.T, yvdspec.T



def calculate_stokes_unnormalised(x, y):
    # 	lambda functions for each of the Stokes parameters
    #
    #	Inputs		X & Y voltage time series
    #
    #	Returns		Stokes time series (I, Q, U, V)
        
    stokes = {
        "i": lambda x, y: np.abs(x) ** 2 + np.abs(y) ** 2,
        "q": lambda x, y: np.abs(y) ** 2 - np.abs(x) ** 2,
        "u": lambda x, y: 2 * np.real(np.conj(x) * y),
        "v": lambda x, y: 2 * np.imag(np.conj(x) * y),
    }
    stks = []
    
    #print("*******************************")
    #print("\n Negating Q to conform to the world of pulsars !!! \n")
    #print("*******************************")
	
    for idx, stk in enumerate(["i", "q", "u", "v"]):
        par = stokes[stk](x, y)
        par_norm = par
        del par
        par = par_norm.transpose()
        del par_norm
        stks += [par]
    
    return(stks)


def gen_stokes_ds(frbname, x, y, dm, nchan, outdir):

	#	Construct dynamic spectra for I, Q, U, V

#	if(os.path.exists(x) and os.path.exists(y)):
	#xvds	=	np.load(x)
	#yvds	=	np.load(y)
	#pcals	=	np.loadtxt(polcalfile)
	stksds	=	np.zeros((4,x.shape[0],y.shape[1]), dtype=float)
	
	for cc in range(0,nchan):
		stkst	=	calculate_stokes_unnormalised(x[cc], y[cc])
		for i in range(0,4):
			stksds[i,cc]	=	stkst[i]	
	
	#np.save("{}/{}_stks_ds_{}_{}_avg_1_1.npy".format(outdir,frbname,dm,nchan),stksds)
	#print("Saved: {}/{}_stks_ds_{}_{}_avg_1_1.npy".format(outdir,frbname,dm,nchan))
	#print(stksds.shape)
	#del(stksds)
	del(x)
	del(y)
#	else:
#		print("gen_stokes_ds ---- PANIC - File(s) not found!")

	return stksds #(nchan*Raw_time_res_ms)



def _robust_zscore_1d(x):
    """
    Compute robust z-score using median and MAD.
    Returns z, median, sigma (MAD*1.4826). Handles NaNs.
    """
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    # Fallback if MAD==0: use std
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if sigma == 0 or not np.isfinite(sigma):
        sigma = 1.0
    z = (x - med) / sigma
    return z, med, sigma


def _compute_dm_shifts_samples(dm, freqs_MHz, tsamp_s, f_ref_MHz=None):
    """
    Compute integer sample shifts per channel for incoherent dedispersion.
    Uses cold-plasma dispersion delay:
      delay_ms = 4.148808e3 * DM * (f^-2 - f_ref^-2)

    Returns:
      shifts (int32 array of length nchan): number of samples to shift LEFT
        for each channel to align to f_ref_MHz (i.e., advance lower freqs).
    """
    freqs = np.asarray(freqs_MHz, dtype=float)
    if f_ref_MHz is None:
        f_ref_MHz = np.nanmax(freqs)  # reference = highest frequency
    k_ms = 4.148808e3  # ms
    delay_ms = k_ms * dm * (freqs**-2 - (f_ref_MHz**-2))
    shifts = np.rint(delay_ms / (tsamp_s * 1e3)).astype(np.int32)
    # Ensure non-negative shifts relative to reference (highest freq ~ 0)
    shifts = np.maximum(shifts, 0)
    return shifts


def _shift_left_no_wrap(arr_2d, shifts):
    """
    Shift each channel in a 2D [nchan, ntime] array LEFT by shifts[ch] samples.
    Zero-fills the trailing samples to avoid wrap-around artifacts.
    """
    nchan, ntime = arr_2d.shape
    out = np.zeros_like(arr_2d)
    for ch in range(nchan):
        s = int(shifts[ch])
        if s < ntime:
            out[ch, : ntime - s] = arr_2d[ch, s:]
        # else: entire row becomes zeros
    return out


def _shift_cube_left_no_wrap(stokes_cube, shifts):
    """
    Apply per-channel left shifts (no wrap) to a Stokes cube [4, nchan, ntime].
    """
    out = np.empty_like(stokes_cube)
    for si in range(stokes_cube.shape[0]):
        out[si] = _shift_left_no_wrap(stokes_cube[si], shifts)
    return out


def find_frb_and_zoom(
    stokes_ds,
    sigma_threshold=7.0,
    zoom_half_width=256,
    use_stokes="i",
    dm=None,
    freqs_MHz=None,
    tsamp_s=None,
    f_ref_MHz=None,
    dedisperse=True,
    save_path=None,
):
    """
    Find an FRB by thresholding the summed time series from Stokes I and return a zoomed-in cube.

    Inputs:
      stokes_ds      : ndarray with shape [4, nchan, ntime] (output of gen_stokes_ds)
      sigma_threshold: detection threshold in robust sigma for the time series
      zoom_half_width: number of samples before/after the peak to include in the zoom
      use_stokes     : which Stokes to detect on ("i","q","u","v"), detection uses "i" by default
      dm             : DM value (pc cm^-3). If provided with freqs_MHz and tsamp_s, apply incoherent dedispersion
      freqs_MHz      : 1D array of channel center frequencies in MHz (length nchan)
      tsamp_s        : sample time in seconds (time resolution of the dynamic spectrum)
      f_ref_MHz      : reference frequency for dedispersion; defaults to highest channel
      dedisperse     : whether to dedisperse when dm, freqs_MHz, tsamp_s are all provided
      save_path      : optional .npy path to save the zoomed cube

    Returns:
      zoom_cube      : ndarray [4, nchan, nzoom] cropped around the detected burst (dedispersed if applied)
      meta           : dict with keys:
                        - peak_index
                        - snr_peak
                        - start_index, end_index
                        - shifts (if dedispersed)
                        - used_stokes
                        - was_dedispersed
    """
    stokes_map = {"i": 0, "q": 1, "u": 2, "v": 3}
    if use_stokes.lower() not in stokes_map:
        raise ValueError("use_stokes must be one of: 'i','q','u','v'")
    sidx = stokes_map[use_stokes.lower()]

    stokes_ds = np.asarray(stokes_ds)
    if stokes_ds.ndim != 3 or stokes_ds.shape[0] != 4:
        raise ValueError("stokes_ds must have shape [4, nchan, ntime]")

    nchan, ntime = stokes_ds.shape[1], stokes_ds.shape[2]

    # Optionally dedisperse the entire cube
    shifts = None
    was_dedispersed = False
    cube = stokes_ds
    if (
        dedisperse
        and (dm is not None)
        and (freqs_MHz is not None)
        and (tsamp_s is not None)
    ):
        shifts = _compute_dm_shifts_samples(dm, freqs_MHz, tsamp_s, f_ref_MHz=f_ref_MHz)
        cube = _shift_cube_left_no_wrap(stokes_ds, shifts)
        was_dedispersed = True

    # Detect on chosen Stokes (default: I)
    S = cube[sidx]  # [nchan, ntime]
    time_series = np.nansum(S, axis=0)  # sum over frequency

    z, med, sigma = _robust_zscore_1d(time_series)
    peak_idx = int(np.nanargmax(z))
    snr_peak = float(z[peak_idx])

    if snr_peak < sigma_threshold:
        # No detection above threshold
        return None, {
            "peak_index": peak_idx,
            "snr_peak": snr_peak,
            "start_index": None,
            "end_index": None,
            "shifts": shifts,
            "used_stokes": use_stokes.lower(),
            "was_dedispersed": was_dedispersed,
        }

    start = max(0, peak_idx - zoom_half_width)
    end = min(ntime, peak_idx + zoom_half_width + 1)

    zoom_cube = cube[:, :, start:end].copy()

    if save_path is not None:
        np.save(save_path, zoom_cube)

    meta = {
        "peak_index": peak_idx,
        "snr_peak": snr_peak,
        "start_index": start,
        "end_index": end,
        "shifts": shifts,
        "used_stokes": use_stokes.lower(),
        "was_dedispersed": was_dedispersed,
    }
    return zoom_cube, meta