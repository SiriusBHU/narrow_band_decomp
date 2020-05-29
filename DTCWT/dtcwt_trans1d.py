from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy.common import Pyramid
from dtcwt.numpy.lowlevel import colfilter, coldfilt, colifilt
from dtcwt.utils import as_column_vector, asfarray


class Transform1d(object):
    """
    An implementation of the 1D DT-CWT in NumPy.

    :param biort: Level 1 wavelets to use. See :py:func:`DTCWT.coeffs.biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`DTCWT.coeffs.qshift`.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        self.biort = biort
        self.qshift = qshift

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
        the columns of a matrix *X*).

        :param X: 1D real array or 2D real array whose columns are to be transformed
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`DTCWT.Pyramid`-like object representing the transform result.

        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
        interpreted as tuples of vectors giving filter coefficients. In the *biort*
        case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
        be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift

        # Need this because colfilter and friends assumes input is 2d
        X = asfarray(X)
        if len(X.shape) == 1:
           X = np.atleast_2d(X).T

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        L = np.asanyarray(X.shape)

        # ensure that X is an even length, thus enabling it to be extended if needs be.
        if X.shape[0] % 2 != 0:
            raise ValueError('Size of input X must be a multiple of 2')

        if nlevels == 0:
            if include_scale:
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        if include_scale:
            # This is only required if the user specifies scales are to be outputted
            Yscale = [None,] * nlevels

        # Level 1.
        Hi = colfilter(X, h1o)
        Lo = colfilter(X, h0o)
        Yh[0] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
        if include_scale:
            Yscale[0] = Lo

        # Levels 2 and above.
        for level in xrange(1, nlevels):
            # Check to see if height of Lo is divisable by 4, if not extend.
            if Lo.shape[0] % 4 != 0:
                Lo = np.vstack((Lo[0,:], Lo, Lo[-1,:]))

            Hi = coldfilt(Lo,h1b,h1a)
            Lo = coldfilt(Lo,h0b,h0a)

            Yh[level] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
            if include_scale:
                Yscale[level] = Lo

        Yl = Lo

        if include_scale:
            return Pyramid(Yl, Yh, Yscale)
        else:
            return Pyramid(Yl, Yh)

    def inverse_hu(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
        reconstruction.

        :param pyramid: A :py:class:`DTCWT.Pyramid`-like object containing the transformed signal.
        :param gain_mask: Gain to be applied to each subband.

        :returns: Reconstructed real array.

        The *l*-th element of *gain_mask* is gain for wavelet subband at level l.
        If gain_mask[l] == 0, no computation is performed for band *l*. Default
        *gain_mask* is all ones. Note that *l* is 0-indexed.

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift

        Yl = pyramid.lowpass
        Yh = pyramid.highpasses

        a = len(Yh) # No of levels.

        if gain_mask is None:
            gain_mask = np.ones(a) # Default gain_mask.

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        level = a-1   # No of levels = no of rows in L.
        if level < 0:
            # if there are no levels in the input, just return the Yl value
            return Yl

        """ 此处可以将 hi Lo 拆开 逐层恢复，考虑到卷积是可拆分的 (x1 + x2) * k = x1 * k + x2 * k """
        low_band_signal = np.copy(Yl)
        high_band_signals = [c2q1d(np.copy(item)) for item in Yh[::-1]]

        for i in range(level):
            low_band_signal = colifilt(low_band_signal, g0b, g0a)
            high_band_signals[i] = colifilt(high_band_signals[i], g1b, g1a)

            if low_band_signal.shape[0] != high_band_signals[i + 1].shape[0]:
                low_band_signal = low_band_signal[1: -1, ...]

            if high_band_signals[i].shape[0] != high_band_signals[i + 1].shape[0]:
                high_band_signals[i] = high_band_signals[i][1: -1, ...]

            for j in range(i):
                high_band_signals[j] = colifilt(high_band_signals[j], g0b, g0a)
                if high_band_signals[j].shape[0] != high_band_signals[i + 1].shape[0]:
                    high_band_signals[j] = high_band_signals[j][1: -1, ...]

        low_band_signal = colfilter(low_band_signal,g0o)
        high_band_signals[-1] = colfilter(high_band_signals[-1],g1o)
        for i in range(level):
            high_band_signals[i] = colfilter(high_band_signals[i], g0o)

        return low_band_signal, high_band_signals


# ==========================================================================================
#                  **********      INTERNAL FUNCTION    **********
# ==========================================================================================

def c2q1d(x):
    """An internal function to convert a 1D Complex vector back to a real
    array,  which is twice the height of x.

    """
    a, b = x.shape
    z = np.zeros((a*2, b), dtype=x.real.dtype)
    z[::2, :] = np.real(x)
    z[1::2, :] = np.imag(x)

    return z

# vim:sw=4:sts=4:et

