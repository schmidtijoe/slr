from slr import options
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pathlib as plb
import json


class SLR:
    def __init__(
            self,
            slr_config: options.SlrConfiguration = options.SlrConfiguration()):
        self.config: options.SlrConfiguration = slr_config
        # set effective ripple sizes
        self.sigma_1, self.sigma_2 = self._effective_ripples()

        # Set some vars
        self.pulse_duration_us: float = self.config.pulse.pulseDuration_in_us
        # bandwidth of pulse in Hz
        # sli_ds [m] * gz[T/m] * gamma [Hz/T] -> Hz
        self.bw_pulse = self.config.pulse.sliceThickness * self.config.globals.maxGrad * self.config.globals.gammaHz
        # time bandwidth prod
        self.tb = self.bw_pulse * self.config.pulse.pulseDuration
        # time steps of pulse
        self.dt_pulse: float = self.config.pulse.pulseDuration / self.config.pulse.pulseNumSamples

        self.a_n_z: np.ndarray = np.zeros(0, dtype=complex)
        self.b_n_z: np.ndarray = np.zeros(0, dtype=complex)
        self.id_b_n_z: np.ndarray = np.zeros(0, dtype=complex)
        self.pulse: np.ndarray = np.zeros(self.config.pulse.pulseNumSamples, dtype=complex)

    def _effective_ripples(self) -> (float, float):
        """
        Computes the effective ripples for different pulse types
        Parameters
        ----------
        type of pulse for which to compute
            - 'smalltip: small tip angle pulse
            - 'pi_2': pi half pulse
            - 'inversion': inversion pulse
            - 'spinecho': spin - echo refocussing pulse
            - 'saturation'

        Returns effective ripplesizes sigma 1 and 2
        -------

        """
        pulse_type = self.config.pulse.pulseType
        r_1 = self.config.pulse.ripple_1
        r_2 = self.config.pulse.ripple_2

        def case_0(s_1, s_2):
            return s_1, s_2

        def case_1(s_1, s_2):
            return np.sqrt(s_1 / 2), s_2 / np.sqrt(2)

        def case_2(s_1, s_2):
            return s_1 / 8, np.sqrt(s_2 / 2)

        def case_3(s_1, s_2):
            return s_1 / 4, np.sqrt(s_2)

        def case_4(s_1, s_2):
            return s_1 / 2, np.sqrt(s_2)

        table = {
            'smalltip': case_0,
            'pi_2': case_1,
            'excitation': case_1,
            '90': case_1,
            'inversion': case_2,
            'spinecho': case_3,
            'refocusing': case_3,
            '180': case_3,
            'saturation': case_4,
        }
        return table.get(pulse_type, 'invalid pulse type')(r_1, r_2)

    def _d_infinity(self, sigma_1: float = None, sigma_2: float = None) -> float:
        """
        empirical calculation for linear phase function d_inf for filter design
        Parameters
        ----------
        sigma_1 pass band ripples % in absolute factor number
        sigma_2 stop band ripples % in absolute factor number

        Returns see SLR paper
        -------

        """
        if sigma_1 is None:
            sigma_1 = self.sigma_1
        if sigma_2 is None:
            sigma_2 = self.sigma_2
        l_1 = np.log10(sigma_1)
        l_2 = np.log10(sigma_2)
        a_params = np.array([5.309 * 1e-3, 7.114 * 1e-2, -4.761 * 1e-1,
                             -2.66 * 1e-3, -5.941 * 1e-1, -4.278 * 1e-1])
        return (a_params[0] * l_1 ** 2 + a_params[1] * l_1 + a_params[2]) * l_2 + \
               (a_params[3] * l_1 ** 2 + a_params[4] * l_1 + a_params[5])

    def _d_minimum_phase(self) -> float:
        return 1 / 2 * self._d_infinity(2 * self.sigma_1, self.sigma_2 ** 2 / 2)

    @staticmethod
    def _pauly_ab2rf(a_k: np.ndarray, b_k: np.ndarray, samples: int) -> np.ndarray:
        b1 = np.zeros(samples, dtype=complex)
        for k_index in np.linspace(samples - 1, 1, samples - 1, dtype=int):
            c_i = np.sqrt(1 / (1 + np.abs(b_k[-1] / a_k[0]) ** 2))
            s_i = np.conj(c_i * np.divide(b_k[-1], a_k[0]))

            theta = np.arctan2(np.abs(s_i), c_i)
            psi = np.angle(s_i)

            b1[k_index] = 2 * (theta * np.cos(psi) + 1j * theta * np.sin(psi))

            a_k_m1 = c_i * a_k + s_i * b_k
            b_k_m1 = - np.conj(s_i) * a_k + c_i * b_k
            a_k = a_k_m1[:-1]
            b_k = b_k_m1[:-1]
        return b1[::-1]

    def plot(self, freqs, plot_save: bool = False):
        # plot
        fig = plt.figure(figsize=(10, 7), dpi=200)
        ax = fig.add_subplot(311)
        bool_idx = freqs ** 2 <= 0.002 ** 2
        freq_plot = freqs[bool_idx]
        ax.plot(freq_plot, np.abs(np.fft.fftshift(self.a_n_z))[bool_idx], label='abs A_n(z)')
        ax.plot(freq_plot, np.abs(np.fft.fftshift(self.b_n_z))[bool_idx], label='abs B_n(z)')
        ax.plot(freq_plot, np.abs(self.id_b_n_z)[bool_idx], label='abs ideal B_n(z)')
        ax.set_xlabel("slice position (z) [m]")
        ax.set_ylabel("amplitude [A.U.]")
        ax.legend(loc=4)

        ax = fig.add_subplot(312)
        ax.plot(freq_plot, np.fft.fftshift(np.real(self.a_n_z))[bool_idx], label='real A_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.imag(self.a_n_z))[bool_idx], label='imag A_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.real(self.b_n_z))[bool_idx], label='real B_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.imag(self.b_n_z))[bool_idx], label='imag B_n(z)')
        ax.set_xlabel("slice position (z) [m]")
        ax.set_ylabel("amplitude [A.U.]")
        ax.legend(loc=4)

        ax = fig.add_subplot(313)
        ax.set_title(f'pulse')
        ax.plot(np.arange(self.config.pulse.pulseNumSamples), np.real(self.pulse), label='real', color='#1b8185')
        ax.plot(np.arange(self.config.pulse.pulseNumSamples), np.imag(self.pulse), label='imag')
        ax.legend()
        plt.tight_layout()
        if plot_save:
            plt.savefig('slr_pi_half_' + str(self.config.pulse.phase) + '.png', bbox_inches='tight', dpi=200)
        else:
            plt.show()

    def build(self):
        """

        :param plot:
        :return:
        """
        # compute fractional width
        if self.config.pulse.phase == "minimum:":
            w = self._d_minimum_phase() / self.tb
        else:
            w = self._d_infinity(self.sigma_1, self.sigma_2) / self.tb

        # compute transition band in frequency
        bw = self.bw_pulse * w
        # pass - and stop band edges
        f_p = (self.bw_pulse - bw) / 2
        f_s = (self.bw_pulse + bw) / 2

        # number of samples of pulse, sampling frequency
        freq_sampling = 1 / self.dt_pulse

        # firls finite response filter function scipy
        # for minimum phase see SLR paper
        # polynomial evaluated at unit circle (aka dft)
        oversampling = 8
        id_b_amplitude = np.sin(self.config.pulse.angle)
        bands_freq = [0, f_p, f_s, 0.5 * freq_sampling]
        bands_gain = [id_b_amplitude, id_b_amplitude, 0, 0]
        bands_weight = [1 / self.sigma_1, 1 / self.sigma_2]

        # minimum phase version of bn
        if self.config.pulse.phase == 'minimum':
            b_n_co = signal.firls(2 * self.config.pulse.pulseNumSamples - 1, bands_freq, bands_gain,
                                  weight=bands_weight, fs=freq_sampling)
            b_n_co = signal.minimum_phase(b_n_co)
        else:
            b_n_co = signal.firls(self.config.pulse.pulseNumSamples, bands_freq, bands_gain,
                                  weight=bands_weight, fs=freq_sampling)

        b_n_z = np.fft.fft(b_n_co, n=oversampling * 2 * self.config.pulse.pulseNumSamples)

        a_n_abs = np.sqrt(1 - np.multiply(b_n_z, np.conjugate(b_n_z)))

        # polynomial evaluated at unit circle (aka dft)
        freqs = np.fft.fftshift(np.fft.fftfreq(oversampling * 2 * self.config.pulse.pulseNumSamples))
        freqs *= self.config.pulse.pulseNumSamples / (
                self.config.globals.gammaHz * self.config.globals.maxGrad * self.config.pulse.pulseDuration)

        # minimum phase version of an
        an_log = np.log(a_n_abs)
        an_logfft = np.fft.fft(an_log)
        an_logfft[1:oversampling * self.config.pulse.pulseNumSamples] *= 2
        an_logfft[oversampling * self.config.pulse.pulseNumSamples + 1:] *= 0
        an_log = np.fft.ifft(an_logfft)
        a_n_z = np.exp(an_log)

        # inverse dft to get coefficients, use positive part of which
        a_n_co = np.fft.ifft(a_n_z, n=oversampling * 2 * self.config.pulse.pulseNumSamples)[
                 :self.config.pulse.pulseNumSamples]

        # setup ideal Bn
        id_b_n_z = np.zeros_like(freqs, dtype=complex)
        id_b_n_z[freqs ** 2 < (self.config.pulse.sliceThickness / 2) ** 2] = 1j * id_b_amplitude

        # save an, bn
        self.a_n_z = a_n_z
        self.id_b_n_z = id_b_n_z
        self.b_n_z = b_n_z

        # Inverse SLR, recover b1
        self.pulse = self._pauly_ab2rf(a_n_co, b_n_co, self.config.pulse.pulseNumSamples)

        if self.config.f_config.visualize:
            self.plot(freqs=freqs)

    def save_rf(self, filename: str = None, phase: np.ndarray = None):
        # if filename not given explicitly, override
        if filename is None:
            filename = plb.Path(self.config.f_config.outputPulseFile).absolute()
        else:
            filename = plb.Path(filename).absolute()
        # pulse calculated along y axis, needs -pi/2 phase for x
        with open(filename, 'w') as txtfile:
            for k_index in range(self.pulse.shape[0]):
                txtfile.write('{:e}'.format(self.pulse[k_index]))
                txtfile.write('\t')
                if phase:
                    if phase.shape[0] != self.pulse.shape[0]:
                        print('phase and amplitude dont match')
                        exit(-1)
                    else:
                        txtfile.write('{:e}'.format(phase[k_index] - np.pi / 2))
                else:
                    txtfile.write('{:e}'.format(-np.pi / 2))
                txtfile.write('\n')

        filename = filename.with_name(f"{filename.stem}_specs").with_suffix(".json")
        j_dict = self._get_pulse_specs()
        with open(filename, "w") as j_file:
            j_file.write(json.dumps(j_dict, indent=2))

    def _get_pulse_specs(self) -> dict:
        d = {
            "bandwidth": self.bw_pulse,
            "time-bandwidth": self.tb,
            "@duration_us": self.pulse_duration_us
        }
        return d

