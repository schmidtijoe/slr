from slr import options
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def effective_ripples(sigma_1: float, sigma_2: float, pulse: str = 'pi_2') -> (float, float):
    """
    Computes the effective ripples for different pulse types
    Parameters
    ----------
    sigma_1: wanted in-slice ripple error unitless, for 1% = 0.01
    sigma_2: wanted out-of slice ripple error unitless, for 1% = 0.01
    pulse: type of pulse for which to compute
        - 'smalltip: small tip angle pulse
        - 'pi_2': pi half pulse
        - 'inversion': inversion pulse
        - 'spinecho': spin - echo refocussing pulse
        - 'saturation'

    Returns effective ripplesizes sigma 1 and 2
    -------

    """

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
    return table.get(pulse, 'invalid pulse type')(sigma_1, sigma_2)


def d_infinity(sigma_1: float, sigma_2: float) -> float:
    """
    empirical calculation for linear phase function d_inf for filter design
    Parameters
    ----------
    sigma_1 pass band ripples % in absolute factor number
    sigma_2 stop band ripples % in absolute factor number

    Returns see SLR paper
    -------

    """
    l_1 = np.log10(sigma_1)
    l_2 = np.log10(sigma_2)
    a_params = np.array([5.309 * 1e-3, 7.114 * 1e-2, -4.761 * 1e-1,
                         -2.66 * 1e-3, -5.941 * 1e-1, -4.278 * 1e-1])
    return (a_params[0] * l_1**2 + a_params[1] * l_1 + a_params[2]) * l_2 +\
           (a_params[3] * l_1**2 + a_params[4] * l_1 + a_params[5])


def d_minimum_phase(sigma_1: float, sigma_2: float) -> float:
    return 1 / 2 * d_infinity(2 * sigma_1, sigma_2**2 / 2)


def pauly_ab2rf(a_k: np.ndarray, b_k: np.ndarray, samples: int) -> np.ndarray:
    b1 = np.zeros(samples, dtype=complex)
    for k_index in np.linspace(samples-1, 1, samples-1, dtype=int):
        c_i = np.sqrt(1 / (1 + np.abs(b_k[-1] / a_k[0])**2))
        s_i = np.conj(c_i * np.divide(b_k[-1], a_k[0]))

        theta = np.arctan2(np.abs(s_i), c_i)
        psi = np.angle(s_i)

        b1[k_index] = 2 * (theta * np.cos(psi) + 1j * theta * np.sin(psi))

        a_k_m1 = c_i * a_k + s_i * b_k
        b_k_m1 = - np.conj(s_i) * a_k + c_i * b_k
        a_k = a_k_m1[:-1]
        b_k = b_k_m1[:-1]
    return b1[::-1]


def build(slr_conf: options.SlrConfiguration, plot: bool = True, plot_save: bool = False) -> np.ndarray:
    """

    :param slr_conf:
    :param plot_save:
    :param plot:
    :return:
    """
    # bandwidth of pulse in Hz
    # sli_ds [m] * gz[T/m] * gamma [Hz/T] -> Hz
    bw_pulse = slr_conf.pulse.sliceThickness * slr_conf.globals.maxGrad * slr_conf.globals.gammaHz
    # time bandwidth prod
    tb = bw_pulse * slr_conf.pulse.pulseDuration
    # effective ripples to ripples translation
    # want 1% effective ripples in profile
    sig_1, sig_2 = effective_ripples(slr_conf.pulse.ripple_1, slr_conf.pulse.ripple_2, slr_conf.pulse.pulseType)

    # compute fractional width
    if slr_conf.pulse.phase == "minimum:":
        w = d_minimum_phase(sig_1, sig_2) / tb
    else:
        w = d_infinity(sig_1, sig_2) / tb

    # compute transition band in frequency
    bw = bw_pulse * w
    # pass - and stop band edges
    f_p = (bw_pulse - bw) / 2
    f_s = (bw_pulse + bw) / 2

    # number of samples of pulse, sampling frequency
    dt_pulse = slr_conf.pulse.pulseDuration / slr_conf.pulse.pulseNumSamples
    freq_sampling = 1 / dt_pulse

    # firls finite response filter function scipy
    # for minimum phase see SLR paper
    # polynomial evaluated at unit circle (aka dft)
    oversampling = 8
    id_b_amplitude = np.sin(slr_conf.pulse.angle)
    bands_freq = [0, f_p, f_s, 0.5 * freq_sampling]
    bands_gain = [id_b_amplitude, id_b_amplitude, 0, 0]
    bands_weight = [1 / sig_1, 1 / sig_2]

    # minimum phase version of bn
    if slr_conf.pulse.phase == 'minimum':
        b_n_co = signal.firls(2 * slr_conf.pulse.pulseNumSamples - 1, bands_freq, bands_gain,
                              weight=bands_weight, fs=freq_sampling)
        b_n_co = signal.minimum_phase(b_n_co)
    else:
        b_n_co = signal.firls(slr_conf.pulse.pulseNumSamples, bands_freq, bands_gain,
                              weight=bands_weight, fs=freq_sampling)

    b_n_z = np.fft.fft(b_n_co, n=oversampling * 2 * slr_conf.pulse.pulseNumSamples)

    a_n_abs = np.sqrt(1 - np.multiply(b_n_z, np.conjugate(b_n_z)))

    # polynomial evaluated at unit circle (aka dft)
    freqs = np.fft.fftshift(np.fft.fftfreq(oversampling * 2 * slr_conf.pulse.pulseNumSamples))
    freqs *= slr_conf.pulse.pulseNumSamples / (
            slr_conf.globals.gammaHz * slr_conf.globals.maxGrad * slr_conf.pulse.pulseDuration)

    # minimum phase version of an
    an_log = np.log(a_n_abs)
    an_logfft = np.fft.fft(an_log)
    an_logfft[1:oversampling * slr_conf.pulse.pulseNumSamples] *= 2
    an_logfft[oversampling * slr_conf.pulse.pulseNumSamples + 1:] *= 0
    an_log = np.fft.ifft(an_logfft)
    a_n_z = np.exp(an_log)

    # inverse dft to get coefficients, use positive part of which
    a_n_co = np.fft.ifft(a_n_z, n=oversampling * 2 * slr_conf.pulse.pulseNumSamples)[:slr_conf.pulse.pulseNumSamples]

    # setup ideal Bn
    id_b_n_z = np.zeros_like(freqs, dtype=complex)
    id_b_n_z[freqs ** 2 < (slr_conf.pulse.sliceThickness / 2) ** 2] = 1j * id_b_amplitude

    # Inverse SLR, recover b1
    b1 = pauly_ab2rf(a_n_co, b_n_co, slr_conf.pulse.pulseNumSamples)

    if plot:
        # plot
        fig = plt.figure(figsize=(10, 7), dpi=200)
        ax = fig.add_subplot(311)
        bool_idx = freqs ** 2 <= 0.002 ** 2
        freq_plot = freqs[bool_idx]
        ax.plot(freq_plot, np.abs(np.fft.fftshift(a_n_z))[bool_idx], label='abs A_n(z)')
        ax.plot(freq_plot, np.abs(np.fft.fftshift(b_n_z))[bool_idx], label='abs B_n(z)')
        ax.plot(freq_plot, np.abs(id_b_n_z)[bool_idx], label='abs ideal B_n(z)')
        ax.set_xlabel("slice position (z) [m]")
        ax.set_ylabel("amplitude [A.U.]")
        ax.legend(loc=4)

        ax = fig.add_subplot(312)
        ax.plot(freq_plot, np.fft.fftshift(np.real(a_n_z))[bool_idx], label='real A_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.imag(a_n_z))[bool_idx], label='imag A_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.real(b_n_z))[bool_idx], label='real B_n(z)')
        ax.plot(freq_plot, np.fft.fftshift(np.imag(b_n_z))[bool_idx], label='imag B_n(z)')
        ax.set_xlabel("slice position (z) [m]")
        ax.set_ylabel("amplitude [A.U.]")
        ax.legend(loc=4)

        ax = fig.add_subplot(313)
        ax.set_title(f'pulse')
        ax.plot(np.arange(slr_conf.pulse.pulseNumSamples), np.real(b1), label='real', color='#1b8185')
        ax.plot(np.arange(slr_conf.pulse.pulseNumSamples), np.imag(b1), label='imag')
        ax.legend()
        plt.tight_layout()
        if plot_save:
            plt.savefig('slr_pi_half_' + str(slr_conf.pulse.phase) + '.png', bbox_inches='tight', dpi=200)
        else:
            plt.show()

    return b1


def save_rf(rf_b1, filename, phase=None):
    # pulse calculated along y axis, needs -pi/2 phase for x
    with open(filename, 'w') as txtfile:
        for k_index in range(len(rf_b1)):
            txtfile.write('{:e}'.format(rf_b1[k_index]))
            txtfile.write('\t')
            if phase:
                if len(phase) != len(rf_b1):
                    print('phase and amplitude dont match')
                    exit(-1)
                else:
                    txtfile.write('{:e}'.format(phase[k_index]-np.pi/2))
            else:
                txtfile.write('{:e}'.format(-np.pi/2))
            txtfile.write('\n')
