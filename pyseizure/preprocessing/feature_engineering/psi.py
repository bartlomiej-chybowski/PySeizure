import numpy as np
from scipy.signal import get_window


class PSI:
    """
    Implementation copied from here FiNN library
    https://github.com/neurophysiological-analysis/FiNN/blob/stable/
    The full library requires R installed in the system, thus copy of only PSI

    """
    def run_psi(self, data_1, data_2, nperseg_outer, fs, nperseg_inner, nfft,
                window, pad_type, f_min, f_max, f_step_sz=1, normalize=True):
        """
        Calculates the phase slope index between two signals. Assumes data_1 and data_2 to be from time domain.

        :param data_1: First dataset from time domain; vector of samples.
        :param data_2: Second dataset from time domain; vector of samples.
        :param nperseg_outer: Outer window size. If normalize = False, this parameter is not used
        :param fs: Sampling frequency.
        :param nperseg_inner: Inner window size.
        :param nfft: fft window size.
        :param window: FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
        :param pad_type: Padding type, currently only "zero" padding is supported.
        :param f_min: Minimum frequence for the evaluated interval.
        :param f_max: Maximum frequence for the evaluated interval.
        :param f_step_sz: Frequency step size in the evaluated interval.
        :param normalize: Determines whether to normalize by dividing through the variance

        :return: Connectivity between data1 and data 2 measured using the phase slope index.
        """

        if normalize is True:
            data_coh = list()

            for idx_start in np.arange(0, len(data_1), nperseg_outer):
                (bins, cc) = self.run_cc(
                    data_1[idx_start:(idx_start + nperseg_outer)],
                    data_2[idx_start:(idx_start + nperseg_outer)],
                    nperseg_inner, pad_type, fs, nfft, window)

                data_coh.append(cc)
        else:
            (bins, tmp) = self.run_cc(data_1, data_2, nperseg_inner, "zero",
                                      fs, nfft, "hann")
            data_coh = [tmp]

        return self.calc_psi(data_coh, bins, f_min, f_max, f_step_sz)

    def run_cc(self, data_1, data_2, nperseg, pad_type, fs, nfft, window):
        """
        Calculate complex coherency from time domain data.

        :param data_1: data set X from time domain.
        :param data_2: data set Y from time domain.
        :param nperseg: number of samples in a fft segment.
        :param pad_type: padding type to be applied.
        :param fs: Sampling frequency.
        :param nfft: Size of fft window.
        :param window: FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.

        :return: Complex coherency.
        """

        seg_data_1 = self._segment_data(data_1, nperseg, pad_type)
        seg_data_2 = self._segment_data(data_2, nperseg, pad_type)

        seg_data_1 = seg_data_1[:seg_data_2.shape[0], :]
        seg_data_2 = seg_data_2[:seg_data_1.shape[0], :]

        (bins, f_data_1) = self._calc_FFT(seg_data_1, fs, nfft, window)
        (_, f_data_2) = self._calc_FFT(seg_data_2, fs, nfft, window)

        return bins, self._run_cc_fd(f_data_1, f_data_2)

    @staticmethod
    def _run_cc_fd(data_1, data_2):
        """
        Calculate complex coherency from frequency domain data.

        :param f_data_1: data set X from the complex frequency domain.
        :param f_data_2: data set Y from the compley frequency domain.

        :return: Complex coherency
        """

        s_xx = np.conjugate(data_1) * data_1 * 2
        s_yy = np.conjugate(data_2) * data_2 * 2
        s_xy = np.conjugate(data_1) * data_2 * 2

        s_xx = np.mean(s_xx, axis = 0)
        s_yy = np.mean(s_yy, axis = 0)
        s_xy = np.mean(s_xy, axis = 0)

        return s_xy/np.sqrt(s_xx*s_yy)

    @staticmethod
    def calc_psi(data, bins, f_min, f_max, f_step_sz=1):
        """

        Calculates the phase slope index (psi) from a list of complex coherency data.

        :param data: List of complex coherency data.
        :param bins: Frequency bins of the complex coherency data.
        :param f_min: Minimum frequency of interest.
        :param f_max: Maximum frequency of interest.
        :param f_step_size: Frequency step size.

        :return: Returns the sfc measured as psi computed from data.

        """

        f_min_idx = np.argmin(np.abs(bins - f_min))
        f_max_idx = np.argmin(np.abs(bins - f_max))

        psi = np.zeros((len(data)), dtype=np.complex64)
        for (psi_idx, comp_coh) in enumerate(data):
            for freq_idx in range(f_min_idx, f_max_idx, 1):
                psi[psi_idx] += np.conjugate(comp_coh[freq_idx]) * \
                                comp_coh[freq_idx + f_step_sz]
            psi[psi_idx] = np.imag(psi[psi_idx])
        psi = np.asarray(psi.real, dtype=np.float32)

        if len(data) > 1:
            var = 0
            for idx in range(len(data)):
                var += np.var(np.concatenate((psi[:idx], psi[(idx + 1):])))
            var /= len(data)

            return np.mean(psi) / (np.sqrt(var) * 2)
        else:
            return psi[0]

    @staticmethod
    def _segment_data(data, nperseg, pad_type="zero"):
        """
        Chop data into segments.

        @param data: Input data; single vector of samples.
        @param nperseg: Length of individual segments.
        @param pad_type: Type of applied padding.

        @return: Segmented data.
        """

        seg_cnt = int(len(data) / nperseg)
        pad_width = nperseg - (len(data) - (seg_cnt * nperseg))

        if pad_width != 0:
            if pad_type == "zero":
                s_data = np.pad(data, (0, pad_width), "constant",
                                constant_values=0)
            else:
                raise NotImplementedError("Error, only supports zero padding")
            seg_cnt += 1

        return np.reshape(s_data,
                          (seg_cnt, nperseg))[:int(len(data) / nperseg), :]

    @staticmethod
    def _calc_FFT(data, fs, nfft, window="hanning"):
        """
        Calculate fft from data

        @param data: Input data; single vector of samples.
        @param fs: Sampling frequency.
        @param nfft: FFT window size.
        @param window: Window type applied during fft.

        @return: (bins, f_data) - frequency bins and corresponding complex fft information.
        """
        m_data = data - np.mean(data)
        m_data = data - np.repeat(np.expand_dims(np.mean(data, axis=1), axis=1),
                                  data.shape[1], axis=1)

        if window == "hanning" or window == "hann":
            win = np.hanning(data.shape[1])
        else:
            win = np.concatenate((get_window(window, data.shape[1] - 1,
                                             fftbins=True), [0]))
        w_data = m_data * win

        if np.complex128 == data.dtype or np.complex256 == data.dtype \
                or np.complex64 == data.dtype:
            f_data = np.fft.fft(w_data, n=nfft, axis=1);
            f_data = f_data[:, :int(f_data.shape[1] / 2 + 1)]
        else:
            f_data = np.fft.rfft(w_data, n=nfft, axis=1)

        bins = np.arange(0, f_data.shape[1], 1) * fs / nfft

        return bins, f_data

