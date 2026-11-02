"""
Module: hmm_tts_backend.py
CSE457 • Xử lý Âm thanh và Tiếng nói | Lab 4
Sinh viên: Mai Thị Thu Hường - MSV: 2351260657

Mô phỏng bộ tổng hợp tiếng nói tham số dựa trên HMM (HMM-based Parametric Speech Synthesis).
Theo kiến trúc HTS / Huang et al. (Spoken Language Processing Ch. 16):
- Context-dependent phone / state sequence
- Parameter trajectory generation (F0, Formants/MFCC, Duration)
- Source-Filter resynthesis (Harmonic glottal excitation + vocal tract formant filter)
"""

import numpy as np
import scipy.signal as signal
from pathlib import Path

# Phoneme / Word acoustic parameters prototype for HMM states
# Mỗi từ vựng được mô hình hóa bằng chuỗi state với phân bố Gauss (mean F0, formants F1, F2, F3, duration)
HMM_ACOUSTIC_MODELS = {
    "mời": {
        "states": ["m", "ow", "i"],
        "f0_mean": 195.0,
        "f0_slope": -15.0,  # Dấu huyền: cao độ đi xuống
        "durations": [0.12, 0.32, 0.16],  # giây
        "formants": [
            [300, 1100, 2400],  # m (nasal murmur)
            [520, 950, 2500],   # ow (nguyên âm ơ)
            [320, 2200, 2900]   # i (lướt sang i)
        ]
    },
    "khách": {
        "states": ["kh", "ae", "ch"],
        "f0_mean": 220.0,
        "f0_slope": 25.0,   # Dấu sắc: cao độ vút lên
        "durations": [0.08, 0.18, 0.08],
        "formants": [
            [450, 1500, 2600],
            [750, 1750, 2700],  # a (nguyên âm mở trước)
            [350, 2100, 3100]
        ]
    },
    "hàng": {
        "states": ["h", "aa", "ng"],
        "f0_mean": 190.0,
        "f0_slope": -20.0,  # Dấu huyền
        "durations": [0.08, 0.30, 0.14],
        "formants": [
            [500, 1400, 2500],
            [780, 1250, 2550],  # a
            [320, 1200, 2400]   # ng (velar nasal)
        ]
    },
    "số": {
        "states": ["s", "oo"],
        "f0_mean": 225.0,
        "f0_slope": 30.0,   # Dấu sắc
        "durations": [0.14, 0.40],
        "formants": [
            [400, 1800, 3200],  # s (fricative)
            [500, 900, 2500]    # o (nguyên âm sau tròn môi)
        ]
    },
    "không": {
        "states": ["kh", "aw", "ng"],
        "f0_mean": 205.0,
        "f0_slope": 0.0,    # Thanh ngang: bằng phẳng
        "durations": [0.08, 0.22, 0.12],
        "formants": [
            [450, 1500, 2500],
            [480, 950, 2450],
            [320, 1150, 2400]
        ]
    },
    "một": {
        "states": ["m", "o_dot", "t"],
        "f0_mean": 185.0,
        "f0_slope": -35.0,  # Dấu nặng: gãy tụt xuống
        "durations": [0.10, 0.20, 0.08],
        "formants": [
            [300, 1100, 2400],
            [520, 1100, 2500],
            [350, 1600, 2700]
        ]
    },
    "hai": {
        "states": ["h", "aa", "i"],
        "f0_mean": 205.0,
        "f0_slope": 0.0,    # Thanh ngang
        "durations": [0.08, 0.26, 0.14],
        "formants": [
            [500, 1400, 2500],
            [750, 1300, 2550],
            [320, 2200, 2900]
        ]
    },
    "ba": {
        "states": ["b", "aa"],
        "f0_mean": 205.0,
        "f0_slope": 0.0,    # Thanh ngang
        "durations": [0.08, 0.40],
        "formants": [
            [250, 900, 2300],   # b (voiced stop)
            [780, 1250, 2500]   # a
        ]
    },
    "bốn": {
        "states": ["b", "oo", "n"],
        "f0_mean": 225.0,
        "f0_slope": 25.0,   # Dấu sắc
        "durations": [0.08, 0.32, 0.12],
        "formants": [
            [250, 900, 2300],
            [500, 920, 2500],
            [320, 1650, 2600]
        ]
    },
    "năm": {
        "states": ["n", "aa", "m"],
        "f0_mean": 205.0,
        "f0_slope": 0.0,    # Thanh ngang
        "durations": [0.10, 0.28, 0.12],
        "formants": [
            [320, 1650, 2600],
            [720, 1350, 2500],
            [300, 1100, 2400]
        ]
    },
    "đến": {
        "states": ["d", "ee", "n"],
        "f0_mean": 225.0,
        "f0_slope": 25.0,   # Dấu sắc
        "durations": [0.08, 0.32, 0.12],
        "formants": [
            [300, 1700, 2600],
            [480, 1950, 2750],  # e
            [320, 1650, 2600]
        ]
    },
    "quầy": {
        "states": ["w", "ay"],
        "f0_mean": 190.0,
        "f0_slope": -20.0,  # Dấu huyền
        "durations": [0.12, 0.40],
        "formants": [
            [350, 850, 2400],
            [650, 1500, 2650]
        ]
    }
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_text_hmm(text):
    text = text.lower().replace(".", "").replace(",", "")
    tokens = []
    for tok in text.split():
        if tok.isdigit():
            tokens.extend(DIGIT_MAP.get(c, c) for c in tok)
        else:
            tokens.append(tok)
    return tokens

def synthesize_hmm(text, sr=16000, frame_shift_ms=10, duration_scale=1.0, f0_scale=1.0):
    """
    HMM Parametric Speech Synthesis:
    1. Text -> HMM State Sequence
    2. MLPG Parameter Generation (F0 contour, Formants trajectory)
    3. Source-Filter Vocoder Resynthesis
    
    Returns:
        y (ndarray): waveform âm thanh Fs=16kHz
        sr (int): tần số lấy mẫu (16000)
        tracks (dict): quỹ đạo tham số âm học {'f0', 'f1', 'f2', 'f3', 'time', 'states'}
    """
    tokens = normalize_text_hmm(text)
    frame_step = int(sr * frame_shift_ms / 1000)
    
    all_states = []
    raw_f0 = []
    raw_f1 = []
    raw_f2 = []
    raw_f3 = []
    
    for tok in tokens:
        model = HMM_ACOUSTIC_MODELS.get(tok, HMM_ACOUSTIC_MODELS["hai"])
        states = model["states"]
        base_f0 = model["f0_mean"] * f0_scale
        f0_slope = model["f0_slope"] * f0_scale
        durations = [d * duration_scale for d in model["durations"]]
        formants = model["formants"]
        
        n_sub_states = len(states)
        total_tok_dur = sum(durations)
        total_frames = max(1, int(total_tok_dur * 1000 / frame_shift_ms))
        
        # Tạo F0 trajectory trơn có ngữ điệu (tone curve)
        f0_tok = np.linspace(base_f0 - f0_slope/2, base_f0 + f0_slope/2, total_frames)
        # Thêm một chút dao động vi mô tự nhiên (jitter)
        f0_tok += 1.5 * np.sin(2 * np.pi * 5 * np.linspace(0, total_tok_dur, total_frames))
        
        # State alignment
        frame_idx = 0
        for s_idx, (st, dur, fmt) in enumerate(zip(states, durations, formants)):
            st_frames = max(1, int(dur * 1000 / frame_shift_ms))
            for f in range(st_frames):
                if frame_idx < total_frames:
                    raw_f0.append(f0_tok[frame_idx])
                    raw_f1.append(fmt[0])
                    raw_f2.append(fmt[1])
                    raw_f3.append(fmt[2])
                    all_states.append(f"{tok}_{st}")
                    frame_idx += 1
                    
        # Chèn khoảng nghỉ ngắn giữa các từ (short pause 40ms)
        pause_frames = int(40 / frame_shift_ms)
        for _ in range(pause_frames):
            raw_f0.append(0.0)  # unvoiced / silence
            raw_f1.append(300.0)
            raw_f2.append(1000.0)
            raw_f3.append(2500.0)
            all_states.append("sil")
            
    # Áp dụng bộ lọc làm mịn (Smoothing mô phỏng MLPG với dynamic features Delta)
    f0_smooth = signal.medfilt(np.array(raw_f0), kernel_size=5)
    b_lpf, a_lpf = signal.butter(2, 0.15, 'low')
    f1_smooth = signal.filtfilt(b_lpf, a_lpf, np.array(raw_f1))
    f2_smooth = signal.filtfilt(b_lpf, a_lpf, np.array(raw_f2))
    f3_smooth = signal.filtfilt(b_lpf, a_lpf, np.array(raw_f3))
    
    total_frames = len(f0_smooth)
    total_samples = total_frames * frame_step
    time_axis = np.arange(total_frames) * (frame_shift_ms / 1000.0)
    
    # Source-Filter Resynthesis
    # 1. Source Excitation (Pulse train cho voiced, noise cho unvoiced/silence)
    excitation = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    for i in range(total_frames):
        start = i * frame_step
        end = min(start + frame_step, total_samples)
        f0_val = f0_smooth[i]
        
        if f0_val > 50.0:  # Voiced
            T0_samples = sr / f0_val
            for n in range(start, end):
                phase += 1.0
                if phase >= T0_samples:
                    phase -= T0_samples
                    excitation[n] = 1.0  # Glottal pulse
        else:  # Unvoiced / Silence
            excitation[start:end] = np.random.normal(0, 0.02, end - start)
            
    # Glottal pulse shaping (Rosenberg glottal model / derivative)
    b_glot = [1.0, -0.96]
    source = signal.lfilter(b_glot, [1.0], excitation)
    
    # 2. Vocal Tract Filter (3 Formant Resonators)
    # Lọc qua từng frame hoặc trung bình lọc dải cộng hưởng
    y_synth = np.zeros(total_samples, dtype=np.float32)
    for i in range(total_frames):
        start = i * frame_step
        end = min(start + frame_step, total_samples)
        seg_source = source[start:end]
        if len(seg_source) == 0:
            continue
            
        # 3 Formant bandpass / resonator filters
        F1, F2, F3 = f1_smooth[i], f2_smooth[i], f3_smooth[i]
        BW1, BW2, BW3 = 90.0, 110.0, 150.0
        
        # Resonator 1
        b1, a1 = signal.iirpeak(F1, F1 / BW1, fs=sr)
        # Resonator 2
        b2, a2 = signal.iirpeak(F2, F2 / BW2, fs=sr)
        # Resonator 3
        b3, a3 = signal.iirpeak(F3, F3 / BW3, fs=sr)
        
        out1 = signal.lfilter(b1, a1, seg_source)
        out2 = signal.lfilter(b2, a2, seg_source)
        out3 = signal.lfilter(b3, a3, seg_source)
        
        y_synth[start:end] = 0.5 * out1 + 0.3 * out2 + 0.2 * out3
        
    # Chuẩn hóa âm lượng đầu ra
    rms_val = np.sqrt(np.mean(y_synth**2) + 1e-12)
    if rms_val > 0:
        y_synth = y_synth * (0.15 / rms_val)
    y_synth = np.clip(y_synth, -0.95, 0.95)
    
    tracks = {
        "time": time_axis,
        "f0": f0_smooth,
        "f1": f1_smooth,
        "f2": f2_smooth,
        "f3": f3_smooth,
        "states": all_states,
        "frame_shift_ms": frame_shift_ms
    }
    
    return y_synth, sr, tracks

if __name__ == "__main__":
    y, sr, tr = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
    print(f"HMM Synthesis complete: {len(y)} samples ({len(y)/sr:.2f}s), {len(tr['f0'])} frames.")

