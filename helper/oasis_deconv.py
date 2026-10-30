"""Deconvolve df/f0 traces with OASIS (https://github.com/j-friedrich/OASIS, pip install oasis-deconv).

OASIS needs NaN-free traces, but rejected frames and image edges leave NaNs in df/f0.
Short gaps are bridged by linear interpolation; at longer gaps the trace is split and
every stretch is deconvolved on its own (with its own baseline and decay estimate).
Frames that were NaN in the input are NaN in the output.

    from helper.oasis_deconv import deconvolve_traces
    calcium, spikes, segments = deconvolve_traces(dff_values)   # frames x neurons, like dff_values
"""
import numpy as np
from oasis.functions import deconvolve


def stretches(valid, *, max_gap=10):
    """(start, stop) frame ranges of valid samples, bridging runs of up to max_gap invalid frames."""
    index = np.flatnonzero(valid)
    if not len(index):
        return []
    breaks = np.flatnonzero(np.diff(index) > max_gap + 1)
    starts = np.concatenate([[index[0]], index[breaks + 1]])
    stops = np.concatenate([index[breaks], [index[-1]]]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def deconvolve_trace(y, *, max_gap=10, min_length=100, penalty=1, **oasis_args):
    """Calcium (denoised) and spike traces of one 1-D trace, plus the fit of every stretch.

    Gaps of up to max_gap frames are interpolated, longer gaps split the trace.
    Stretches shorter than min_length frames are too short to estimate the decay and stay NaN.
    penalty and further keyword arguments go to oasis.functions.deconvolve (penalty=1: L1, 0: L0).
    """
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y)
    calcium = np.full(y.shape, np.nan)
    spikes = np.full(y.shape, np.nan)
    segments = []
    for start, stop in stretches(valid, max_gap=max_gap):
        if stop - start < min_length:
            continue
        frames = np.arange(start, stop)
        part = np.interp(frames, frames[valid[start:stop]], y[start:stop][valid[start:stop]])
        c, s, b, g, lam = deconvolve(part, penalty=penalty, **oasis_args)
        calcium[start:stop], spikes[start:stop] = c, s
        segments.append(dict(start=start, stop=stop, baseline=b, g=np.atleast_1d(g), lam=lam))
    calcium[~valid] = np.nan
    spikes[~valid] = np.nan
    return calcium, spikes, segments


def deconvolve_traces(dff, **kwargs):
    """deconvolve_trace for every column of a (frames x neurons) matrix.

    Returns calcium and spikes with the shape of dff and, per column, the list of fitted stretches.
    """
    dff = np.asarray(dff, dtype=np.float64)
    calcium = np.full(dff.shape, np.nan)
    spikes = np.full(dff.shape, np.nan)
    segments = []
    for column in range(dff.shape[1]):
        calcium[:, column], spikes[:, column], fits = deconvolve_trace(dff[:, column], **kwargs)
        segments.append(fits)
    return calcium, spikes, segments
