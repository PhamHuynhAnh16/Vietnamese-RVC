import math

import numba as nb
import numpy as np

from scipy import signal
from scipy.interpolate import interp1d

def dio(x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, target_fs=4000, frame_period=10, allowed_range=0.1):
    temporal_positions = np.arange(0, int(1000 * len(x) / fs / frame_period + 1)) * frame_period / 1000
    boundary_f0_list = f0_floor * (2.0 ** ((np.arange(math.ceil(np.log2(f0_ceil / f0_floor) * channels_in_octave)) + 1) / channels_in_octave))
    
    y = decimate(x, int(fs / target_fs))
    y_spectrum = get_spectrum(y, target_fs, f0_floor)
    raw_f0_candidate, raw_stability = get_candidate_and_stability(np.size(temporal_positions), boundary_f0_list, np.size(y), temporal_positions, target_fs, y_spectrum, f0_floor, f0_ceil)
    
    return np.array(fix_f0_contour(sort_candidates(raw_f0_candidate, raw_stability), frame_period, f0_floor, allowed_range), dtype=np.float32), np.array(temporal_positions, dtype=np.float32)

def get_downsampled_signal(x, fs, target_fs):
    decimation_ratio = int(fs / target_fs + 0.5)

    if fs < target_fs:
        y = np.empty_like(x)
        y[:] = x
        actual_fs = fs
    else: 
        y = decimate_matlab(x, decimation_ratio, n = 3)
        actual_fs = fs / decimation_ratio

    y -= np.mean(y)
    return y, actual_fs

def get_spectrum(x, fs, lowest_f0):
    fft_size = 2 ** math.ceil(math.log(np.size(x) + int(fs / lowest_f0 / 2 + 0.5) * 4,2))
    cutoff_in_sample = int(fs / 50 + 0.5)

    low_cut_filter = signal.windows.hann(2 * cutoff_in_sample + 3)[1:-1] 
    low_cut_filter = -low_cut_filter / np.sum(low_cut_filter)
    low_cut_filter[cutoff_in_sample] = low_cut_filter[cutoff_in_sample] + 1
    low_cut_filter = np.r_[low_cut_filter, np.zeros(fft_size - len(low_cut_filter))]
    low_cut_filter = np.r_[low_cut_filter[cutoff_in_sample:], low_cut_filter[:cutoff_in_sample]]
    
    return np.fft.fft(x, fft_size) * np.fft.fft(low_cut_filter, fft_size)

def get_candidate_and_stability(number_of_frames, boundary_f0_list, y_length, temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil):
    raw_f0_candidate = np.zeros((np.size(boundary_f0_list), number_of_frames))
    raw_f0_stability = np.zeros((np.size(boundary_f0_list), number_of_frames))

    for i in range(np.size(boundary_f0_list)):
        interpolated_f0, f0_deviations = get_raw_event(boundary_f0_list[i], actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil)
        raw_f0_stability[i, :] = np.exp(-(f0_deviations / np.maximum(interpolated_f0, 0.0000001)))
        raw_f0_candidate[i, :] = interpolated_f0

    return raw_f0_candidate, raw_f0_stability

def sort_candidates(f0_candidate_map, stability_map):
    number_of_candidates, number_of_frames = f0_candidate_map.shape
    sorted_index = np.argsort(-stability_map, axis=0, kind='quicksort')
    f0_candidates = np.zeros((number_of_candidates, number_of_frames))

    for i in range(number_of_frames):
        f0_candidates[:, i] = f0_candidate_map[sorted_index[:number_of_candidates,i], i]

    return f0_candidates

def get_raw_event(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil):
    low_pass_filter = nuttall(int(fs / boundary_f0 / 2 + 0.5) * 4)

    filtered_signal = np.real(np.fft.ifft(np.fft.fft(low_pass_filter, len(y_spectrum)) * y_spectrum))
    filtered_signal = filtered_signal[low_pass_filter.argmax() + np.arange(1, y_length + 1)] 
    
    neg_loc, neg_f0 = ZeroCrossingEngine(filtered_signal, fs)
    pos_loc, pos_f0 = ZeroCrossingEngine(-filtered_signal, fs)
    peak_loc, peak_f0 = ZeroCrossingEngine(np.diff(filtered_signal), fs)
    dip_loc, dip_f0 = ZeroCrossingEngine(-np.diff(filtered_signal), fs)
    
    f0_candidate, f0_deviations = get_f0_candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions)
    
    f0_candidate[f0_candidate > boundary_f0] = 0
    f0_candidate[f0_candidate < (boundary_f0 / 2)] = 0
    f0_candidate[f0_candidate > f0_ceil] = 0
    f0_candidate[f0_candidate < f0_floor] = 0
    f0_deviations[f0_candidate == 0] = 100000 
    
    return f0_candidate, f0_deviations

def get_f0_candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions):
    usable_channel = max(0, np.size(neg_loc) - 2) * max(0, np.size(pos_loc) - 2) * max(0, np.size(peak_loc) - 2) * max(0, np.size(dip_f0) - 2)
    interpolated_f0_list = np.zeros((4, np.size(temporal_positions)))
    
    if usable_channel > 0:
        interpolated_f0_list[0, :] = interp1d(neg_loc, neg_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1, :] = interp1d(pos_loc, pos_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[2, :] = interp1d(peak_loc, peak_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3, :] = interp1d(dip_loc, dip_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
        f0_deviations = np.std(interpolated_f0_list, axis=0, ddof=1)
    else:
        interpolated_f0 = temporal_positions * 0
        f0_deviations = temporal_positions * 0 + 1000

    return interpolated_f0, f0_deviations

@nb.jit((nb.float64[:], nb.float64), nopython=True, cache=True)
def ZeroCrossingEngine(x, fs):
    y = np.empty_like(x)
    y[:-1] = x[1:]
    y[-1] = x[-1]

    negative_going_points = np.arange(1, len(x) + 1) * ((y * x < 0) * (y < x))
    edge_list = negative_going_points[negative_going_points > 0]
    fine_edge_list = (edge_list) - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1])

    return (fine_edge_list[:len(fine_edge_list) - 1] + fine_edge_list[1:]) / 2 / fs, fs / np.diff(fine_edge_list)

def nuttall(N):
    return np.squeeze(np.asarray(np.array([0.355768, -0.487396, 0.144232, -0.012604]) @ np.cos(np.matrix([0,1,2,3]).T @ np.asmatrix(np.arange(N) * 2 * math.pi / (N-1)))))

def fix_f0_contour(f0_candidates, frame_period, f0_floor, allowed_range):
    voice_range_minimum =int(1 / (frame_period / 1000) / f0_floor + 0.5) * 2 + 1
    f0_step2 = fix_step2(fix_step1(f0_candidates, voice_range_minimum, allowed_range), voice_range_minimum)
    section_list = count_voiced_sections(f0_step2)
    f0_step4 = fix_step4(fix_step3(f0_step2, f0_candidates, section_list, allowed_range), f0_candidates, section_list, allowed_range)
    
    return np.copy(f0_step4)

def fix_step1(f0_candidates, voice_range_minimum, allowed_range):
    f0_base = f0_candidates[0]
    f0_base[ : voice_range_minimum] = 0
    f0_base[-voice_range_minimum : ] = 0
    
    f0_step1 = np.copy(f0_base)
    rounding_f0_base = np.array([float("{0:.6f}".format(elm)) for elm in f0_base])
    for i in np.arange(voice_range_minimum - 1, len(f0_base)):
        if abs((rounding_f0_base[i] - rounding_f0_base[i-1]) / (0.000001 + rounding_f0_base[i])) > allowed_range: f0_step1[i] = 0

    return f0_step1

def fix_step2(f0_step1, voice_range_minimum):
    f0_step2 = np.copy(f0_step1)
    for i in np.arange((voice_range_minimum - 1) / 2 , len(f0_step1) - (voice_range_minimum - 1) / 2).astype(int):
        for j in np.arange( -(voice_range_minimum - 1) / 2 , (voice_range_minimum - 1) / 2 + 1).astype(int):
            if f0_step1[i + j] == 0:
                f0_step2[i] = 0
                break

    return f0_step2

def fix_step3(f0_step2, f0_candidates, section_list, allowed_range):
    f0_step3 = np.empty_like(f0_step2)
    f0_step3[:] = f0_step2

    for i in np.arange(section_list.shape[0]):
        limit = len(f0_step3) - 1 if i == section_list.shape[0] - 1 else section_list[i + 1, 0] + 1

        for j in np.arange(section_list[i, 1], limit).astype(int):
            f0_step3[j + 1] = select_best_f0(f0_step3[j], f0_step3[j - 1], f0_candidates[:, j + 1], allowed_range)
            if f0_step3[j + 1] == 0: break

    return f0_step3

def fix_step4(f0_step3, f0_candidates, section_list, allowed_range):
    f0_step4 = np.copy(f0_step3)
    
    for i in range(section_list.shape[0] - 1, -1 , -1):
        limit = 1 if i == 0 else section_list[i - 1, 1]

        for j in np.arange(section_list[i, 0], limit - 1,  -1).astype(int):
            f0_step4[j - 1] = select_best_f0(f0_step4[j], f0_step4[j + 1], f0_candidates[:, j - 1], allowed_range)
            if f0_step4[j - 1] == 0: break

    return f0_step4

def select_best_f0(current_f0, past_f0, candidates, allowed_range):
    from sys import float_info

    reference_f0 = (current_f0 * 3 - past_f0) / 2
    minimum_error = abs(reference_f0 - candidates[0])
    best_f0 = candidates[0]
    
    for i in range(1, len(candidates)):
        current_error = abs(reference_f0 - candidates[i])
        if current_error < minimum_error:
            minimum_error = current_error
            best_f0 = candidates[i]

    if abs(1 - best_f0 / (reference_f0 + float_info.epsilon)) > allowed_range: best_f0 = 0
    return best_f0

def count_voiced_sections(f0):
    vuv = np.copy(f0)
    vuv[vuv != 0] = 1
    diff_vuv = np.diff(vuv)
    boundary_list = np.append(np.append([0], np.where(diff_vuv != 0)[0]), [len(vuv) - 2])
    
    first_section = np.ceil(-0.5 * diff_vuv[boundary_list[1]])
    number_of_voiced_sections = np.floor((len(boundary_list) - (1 - first_section)) / 2).astype(int)

    voiced_section_list = np.zeros((number_of_voiced_sections, 2))
    for i in range(number_of_voiced_sections):
        voiced_section_list[i, :] = np.array([1 + boundary_list[int((i - 1) * 2 + 1 + (1 - first_section)) + 1], boundary_list[int((i * 2) + (1 - first_section)) + 1]])

    return voiced_section_list

def decimate_matlab(x, q, n=None, axis=-1):
    if not isinstance(q, int): raise TypeError
    if n is not None and not isinstance(n, int): raise TypeError

    system = signal.dlti(*signal.cheby1(n, 0.05, 0.8 / q))
    y = signal.filtfilt(system.num, system.den, x, axis=axis, padlen=3 * (max(len(system.den), len(system.num)) - 1))

    nd = len(y)
    return y[int(q - (q * np.ceil(nd / q) - nd)) - 1::q]

def FilterForDecimate(x,r):
    a, b = np.zeros(3), np.zeros(2)

    if r==11:
        a[0] = 2.450743295230728
        a[1] = -2.06794904601978
        a[2] = 0.59574774438332101
        b[0] = 0.0026822508007163792
        b[1] = 0.0080467524021491377
    elif r==12:
        a[0] = 2.4981398605924205
        a[1] = -2.1368928194784025
        a[2] = 0.62187513816221485
        b[0] = 0.0021097275904709001
        b[1] = 0.0063291827714127002
    elif r==10:
        a[0] = 2.3936475118069387
        a[1] = -1.9873904075111861
        a[2] = 0.5658879979027055
        b[0] = 0.0034818622251927556
        b[1] = 0.010445586675578267
    elif r==9:
        a[0] = 2.3236003491759578
        a[1] = -1.8921545617463598
        a[2] = 0.53148928133729068
        b[0] = 0.0046331164041389372
        b[1] = 0.013899349212416812
    elif r==8:
        a[0] = 2.2357462340187593
        a[1] = -1.7780899984041358
        a[2] = 0.49152555365968692
        b[0] = 0.0063522763407111993
        b[1] = 0.019056829022133598
    elif r==7:
        a[0] = 2.1225239019534703
        a[1] = -1.6395144861046302
        a[2] = 0.44469707800587366
        b[0] = 0.0090366882681608418
        b[1] = 0.027110064804482525
    elif r==6:
        a[0] = 1.9715352749512141
        a[1] = -1.4686795689225347
        a[2] = 0.3893908434965701
        b[0] = 0.013469181309343825
        b[1] = 0.040407543928031475
    elif r==5:
        a[0] = 1.7610939654280557
        a[1] = -1.2554914843859768
        a[2] = 0.3237186507788215
        b[0] = 0.021334858522387423
        b[1] = 0.06400457556716227
    elif r==4:
        a[0] = 1.4499664446880227
        a[1] = -0.98943497080950582
        a[2] = 0.24578252340690215
        b[0] = 0.036710750339322612
        b[1] = 0.11013225101796784
    elif r==3:
        a[0] = 0.95039378983237421
        a[1] = -0.67429146741526791
        a[2] = 0.15412211621346475
        b[0] = 0.071221945171178636
        b[1] = 0.21366583551353591
    elif r==2:
        a[0] = 0.041156734567757189
        a[1] = -0.42599112459189636
        a[2] = 0.041037215479961225
        b[0] = 0.16797464681802227
        b[1] = 0.50392394045406674
    else: a[0] = a[1] = a[2] = b[0] = b[1] = 0.0

    w = np.zeros(3)
    y_prime = np.zeros_like(x)

    for i in range(len(x)):
        wt = x[i] + a[0] * w[0] + a[1] * w[1] + a[2] * w[2]
        y_prime[i] = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2]
        w[2] = w[1]
        w[1] = w[0]
        w[0] = wt

    return y_prime

def decimate(x,r):
    y = []
    kNFact = 9
    x_length = len(x)

    tmp1 = np.zeros(x_length + kNFact * 2)
    tmp2 = np.zeros(x_length + kNFact * 2)

    for i in range(kNFact):
        tmp1[i] = 2 * x[0] - x[kNFact - i]

    for i in range(kNFact, kNFact + x_length):
        tmp1[i] = x[i - kNFact]

    for i in range(kNFact + x_length, 2 * kNFact + x_length):
        tmp1[i] = 2 * x[-1] - x[x_length - 2 - (i - (kNFact + x_length))]

    tmp2 = FilterForDecimate(tmp1, r)
    for i in range(2 * kNFact + x_length):
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]

    tmp2 = FilterForDecimate(tmp1, r)
    for i in range(2 * kNFact + x_length):
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]

    nbeg = int(r - r * np.ceil(x_length / r + 1) + x_length)

    count = 0
    for i in range(nbeg, x_length + kNFact, r):
        y.append(tmp1[i + kNFact - 1])
        count += 1

    return np.array(y)