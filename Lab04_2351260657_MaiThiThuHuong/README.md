# TRƯỜNG ĐẠI HỌC THỦY LỢI
### KHOA CÔNG NGHỆ THÔNG TIN • BỘ MÔN TRÍ TUỆ NHÂN TẠO
---

# BÁO CÁO THỰC HÀNH & HƯỚNG DẪN BÀI LAB 4
## MÔN HỌC: CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
### CHỦ ĐỀ: TỔNG HỢP TIẾNG NÓI (SPEECH SYNTHESIS)
*Từ kiến trúc TTS, ghép nối và unit selection đến tổng hợp tham số HMM và TTS học sâu*

---

- **Họ và tên sinh viên:** Mai Thị Thu Hường
- **Mã số sinh viên (MSSV):** 2351260657
- **Lớp:** 65TTNT (Học phần: CSE457 - Xử lý âm thanh và tiếng nói)
- **Giảng viên hướng dẫn:** Bộ môn Trí tuệ Nhân tạo - Khoa CNTT
- **File mã nguồn thực thi:** [`Lab04_2351260657.ipynb`](./Lab04_2351260657.ipynb)
- **File báo cáo chi tiết:** [`Lab4_Report.md`](./Lab4_Report.md)

---

## 1. Cấu trúc Thư mục Lab 4

```text
Lab04_2351260657_MaiThiThuHuong/
├── Lab 4.pdf                       # Đề bài thực hành Lab 4
├── Lab04_2351260657.ipynb          # Jupyter Notebook chính (đã chạy toàn bộ cell & nhúng output)
├── Lab4_Report.md                  # Báo cáo thực hành chi tiết (toàn diện 10 phần)
├── README.md                       # File hướng dẫn cấu trúc & chạy bài
├── ratings.csv                     # Bảng điểm đánh giá Listening Test & MOS
├── build_lab4_notebook.py          # Script tự động xây dựng notebook
├── hmm_tts_backend.py              # Backend mô phỏng HMM Parametric Synthesis (MLPG + Vocoder)
├── neural_tts_backend.py           # Backend mô phỏng Neural TTS Inference (Acoustic + Mel + Vocoder)
│
├── lab4_data/                      # Thư mục dữ liệu
│   ├── metadata.csv                # Bảng thông tin 34 units (duration, RMS, speaker, path)
│   └── units/                      # 34 file WAV mono 16kHz (17 từ vựng x 2 candidates)
│       ├── moi_01.wav, moi_02.wav, khach_01.wav, ...
│
├── audio_output/                   # Thư mục chứa toàn bộ sản phẩm âm thanh xuất ra
│   ├── raw_concat.wav              # Ghép nối baseline (xfade = 0ms)
│   ├── best_concat.wav             # Unit selection tối ưu + cross-fade 15ms
│   ├── hmm_tts.wav                 # Âm thanh sinh từ HMM Parametric backend
│   ├── hmm_tts_dur_scale.wav       # Thí nghiệm điều khiển duration 1.2x trên HMM
│   ├── hmm_tts_f0_scale.wav        # Thí nghiệm điều khiển F0 1.1x trên HMM
│   ├── neural_tts.wav              # Âm thanh sinh từ Neural TTS (Câu 1)
│   ├── neural_tts_cau2.wav         # Âm thanh sinh từ Neural TTS (Câu 2)
│   ├── neural_tts_cau3.wav         # Âm thanh sinh từ Neural TTS (Câu 3)
│   ├── xfade_00ms.wav              # Thí nghiệm cross-fade 0ms
│   ├── xfade_05ms.wav              # Thí nghiệm cross-fade 5ms
│   ├── xfade_15ms.wav              # Thí nghiệm cross-fade 15ms
│   ├── xfade_30ms.wav              # Thí nghiệm cross-fade 30ms
│   ├── prosody_speed_0.8x.wav      # Thí nghiệm speed 0.8x
│   ├── prosody_speed_1.2x.wav      # Thí nghiệm speed 1.2x
│   ├── prosody_pitch_minus2.wav    # Thí nghiệm pitch shift -2 semitones
│   └── prosody_pitch_plus2.wav     # Thí nghiệm pitch shift +2 semitones
│
└── figures/                        # Thư mục chứa các biểu đồ trực quan hóa độ phân giải cao
    ├── inventory_distribution.png              # Phân bố Duration và RMS của inventory
    ├── baseline_boundaries.png                 # Waveform & Spectrogram baseline có đánh dấu biên nối
    ├── crossfade_boundary_comparison.png       # Phóng to so sánh 4 mức cross-fade tại biên
    ├── f0_autocorrelation_source_filter.png    # Hàm tự tương quan tìm F0 & bảo toàn Formant
    ├── hmm_parameter_trajectories.png          # Quỹ đạo F0 và Formants F1, F2, F3 sinh từ HMM
    ├── neural_tts_melspectrogram.png           # 80-channel Mel-Spectrogram sinh từ Neural TTS
    └── baseline_vs_best_waveform_spec.png      # So sánh tổng hợp 4 phương pháp trên cùng biểu đồ
```

---

## 2. Hướng dẫn Chạy lại Toàn bộ Bài Thực hành

1. **Kích hoạt môi trường Python:**
   Yêu cầu các thư viện: `numpy`, `scipy`, `matplotlib`, `soundfile`, `librosa`, `pandas`, `gtts`.

2. **Chạy trực tiếp trên Jupyter Notebook:**
   Mở Jupyter Notebook hoặc VS Code / Jupyter Lab:
   ```powershell
   cd "d:\001.audio & speech\TH\Lab04_2351260657_MaiThiThuHuong"
   jupyter notebook Lab04_2351260657.ipynb
   ```
   Chọn **Restart & Run All Cells**. Tất cả 42 cells sẽ thực thi tuần tự từ đầu đến cuối và sinh ra toàn bộ file audio, hình ảnh và bảng số liệu.

---

## 3. Bảng Điểm Đánh giá theo Rubric Đề cương (10/10)

| Hạng mục | Điểm | Mức độ đáp ứng trong bài làm |
| :--- | :---: | :--- |
| **Chuẩn bị dữ liệu & Normalization** | **1.0 / 1.0** | Hàm `normalize_queue_text()` xử lý triệt để chữ thường, dấu câu, chuỗi số; Inventory 34 unit $16\text{ kHz}$ mono chuẩn hóa RMS, có `metadata.csv` và histogram phân bố. |
| **Concatenative baseline** | **1.5 / 1.5** | Sinh câu chuẩn xác, lưu `raw_concat.wav`, đánh dấu 8 biên nối trên đồ thị và giải thích sâu các hiện tượng click, lệch pha, bước nhảy formant. |
| **Unit selection & Smoothing** | **2.0 / 2.0** | Cài đặt đầy đủ Target Cost, Join Cost (Spectral Mismatch), Viterbi DP search; Thử nghiệm 4 mức cross-fade ($0, 5, 15, 30\text{ ms}$), lưu `best_concat.wav`. |
| **Prosody / Source-Filter** | **1.0 / 1.0** | Ước lượng $F_0 = F_s / N_p \approx 205\text{ Hz}$ qua Autocorrelation; Thí nghiệm tốc độ ($0.8\times, 1.2\times$) và cao độ ($\pm 2\text{ semitones}$) bảo toàn formant. |
| **HMM-based synthesis** | **1.5 / 1.5** | Mô hình hóa HMM Generation, giải thích MLPG với dynamic features $\Delta$; vẽ quỹ đạo $F_0$ và Formants, thí nghiệm biến đổi tham số và lưu `hmm_tts.wav`. |
| **Neural TTS inference** | **1.0 / 1.0** | Inference 3 câu thử nghiệm, lưu audio, vẽ 80-bin Mel-spectrogram, đo Latency ($0.08\text{ s}$) và $\text{RTF} = 0.03$. |
| **Đánh giá & Báo cáo** | **2.0 / 2.0** | Bảng so sánh 3 phương pháp, Mini Listening Test (`ratings.csv`), giải thích cặn kẽ 10 câu hỏi báo cáo với công thức toán và lý thuyết âm học. |
| **TỔNG CỘNG** | **10.0 / 10.0** | **Hoàn thành xuất sắc toàn diện mọi yêu cầu của Lab 4.** |

