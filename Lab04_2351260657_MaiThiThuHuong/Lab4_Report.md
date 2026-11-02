# TRƯỜNG ĐẠI HỌC THỦY LỢI
### KHOA CÔNG NGHỆ THÔNG TIN • BỘ MÔN TRÍ TUỆ NHÂN TẠO
---

# BÁO CÁO THỰC HÀNH CHI TIẾT LAB 4
## MÔN HỌC: CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
### CHỦ ĐỀ: TỔNG HỢP TIẾNG NÓI (SPEECH SYNTHESIS)
*Từ kiến trúc TTS, ghép nối và unit selection đến tổng hợp tham số HMM và TTS học sâu*

---

- **Họ và tên sinh viên:** Mai Thị Thu Hường
- **Mã số sinh viên (MSSV):** 2351260657
- **Lớp:** 65TTNT (Học phần: CSE457 - Xử lý âm thanh và tiếng nói)
- **Giảng viên hướng dẫn:** Bộ môn Trí tuệ Nhân tạo - Khoa CNTT
- **File mã nguồn thực thi:** [`Lab04_2351260657.ipynb`](./Lab04_2351260657.ipynb)
- **Thư mục âm thanh kết quả:** [`audio_output/`](./audio_output/)
- **Thư mục biểu đồ:** [`figures/`](./figures/)
- **Bảng điểm đánh giá chủ quan:** [`ratings.csv`](./ratings.csv)

---

## 1. Mục tiêu và Chuẩn đầu ra của Bài Lab

1. **Nắm vững kiến trúc Text-to-Speech tổng quát:**
   - Phân tích và thực thi toàn bộ chuỗi xử lý: $\text{Text Analysis} \to \text{Phonetic Analysis} \to \text{Prosodic Analysis} \to \text{Synthesis Backend} \to \text{Waveform}$.
2. **Text Normalization & Xây dựng Inventory:**
   - Xây dựng quy tắc chuẩn hóa văn bản miền hẹp "Thông báo xếp hàng" (chuyển chữ thường, bỏ dấu câu, đọc số theo từng chữ số).
   - Thiết lập kho lưu trữ 34 units (17 từ vựng $\times$ 2 candidates) chuẩn hóa $F_s = 16\text{ kHz}$, mono, 16-bit PCM và cân bằng năng lượng RMS.
3. **Phân tích Ghép nối Baseline (Concatenative Synthesis):**
   - Sinh câu nói bằng cách ghép trực tiếp các mẫu sóng tự nhiên, xác định tọa độ biên nối và phân tích các hiện tượng lỗi click, lệch pha, lệch formant và nhảy cao độ.
4. **Tối ưu hóa Unit Selection & Boundary Smoothing:**
   - Tính toán Target Cost ($C_{target}$) và Join Cost ($C_{join}$) dựa trên sai lệch phổ log-magnitude ($D_{spec}$) và năng lượng RMS.
   - Ứng dụng quy hoạch động (Dynamic Programming / Viterbi Search) để tìm chuỗi candidate tối ưu.
   - Thử nghiệm 4 mức Cross-fade ($0, 5, 15, 30\text{ ms}$) và chứng minh mức $15\text{ ms}$ là tối ưu để loại bỏ tiếng nổ click mà không làm mờ phụ âm.
5. **Khảo sát Ngữ điệu (Prosody) & Mô hình Source-Filter:**
   - Đo tần số cơ bản $F_0 \approx F_s / N_p$ qua hàm tự tương quan trên đoạn nguyên âm hữu thanh.
   - Thí nghiệm biến đổi tốc độ ($0.8\times, 1.0\times, 1.2\times$) và cao độ ($\pm 2\text{ semitones}$) theo công thức $r = 2^{\Delta s / 12}$.
6. **Mô phỏng HMM-based Parametric Synthesis:**
   - Khảo sát nguyên lý phát sinh tham số (Generation Mode) của HMM; vai trò của thuật toán MLPG kết hợp đặc trưng động $\Delta$ trong việc làm trơn quỹ đạo $F_0$ và Formant.
   - Thí nghiệm điều khiển tham số giọng nói (scale duration $1.2\times$, scale $F_0$ $1.1\times$).
7. **Inference Neural TTS & Đo đạc Hiệu năng:**
   - Thực thi pipeline TTS học sâu hai giai đoạn (Acoustic Model sinh Mel-Spectrogram + Neural Vocoder tái tạo sóng).
   - Đo đạc Latency và Real-Time Factor (RTF) trên 3 câu thử nghiệm.
8. **Đánh giá toàn diện & Báo cáo:**
   - So sánh 3 phương pháp trên 5 tiêu chí: Intelligibility, Naturalness, Continuity, Latency và Tài nguyên.
   - Thiết kế Listening Test, xuất file `ratings.csv` và giải đáp 10 câu hỏi lý thuyết chuyên sâu.

---

## 2. Phần A - Chuẩn hóa Văn bản (Text Normalization) & Kho lưu trữ Inventory

### 2.1. Phương pháp & Công thức
Trong miền hẹp hệ thống gọi số xếp hàng:
$$\text{Input: } T = \text{"Mời khách hàng số 23 đến quầy 5."}$$
Quy tắc chuẩn hóa:
1. Chuyển toàn bộ ký tự sang chữ thường (lowercase).
2. Loại bỏ các dấu câu (".", ",").
3. Đọc từng chữ số riêng biệt (digit-by-digit mapping) thay vì đọc số đếm hay số thứ tự:
   $$\text{"23"} \to [\text{"hai"}, \text{"ba"}], \quad \text{"5"} \to [\text{"năm"}]$$
Kết quả thu được chuỗi token:
$$W = [\text{"mời"}, \text{"khách"}, \text{"hàng"}, \text{"số"}, \text{"hai"}, \text{"ba"}, \text{"đến"}, \text{"quầy"}, \text{"năm"}]$$

### 2.2. Chuẩn hóa Năng lượng RMS
Mỗi đơn vị âm thanh được chuẩn hóa theo mức RMS mục tiêu $\text{RMS}_{target} = 0.08$:
$$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}, \quad y[n] = x[n] \cdot \frac{\text{RMS}_{target}}{\text{RMS}(x)}$$
Dữ liệu lưu trữ gồm 34 files trong `lab4_data/units/` kèm file `metadata.csv`. Thời lượng trung bình của các unit là $0.49\text{ s}$, năng lượng RMS tập trung đều đặn tại mức $0.078 - 0.080$.

---

## 3. Phần B - Ghép nối Baseline & Phân tích Biên nối

### 3.1. Kết quả Ghép nối Trực tiếp
Khi ghép nối trực tiếp candidate cố định mà không có xử lý biên (`xfade_ms = 0`), file [`raw_concat.wav`](./audio_output/raw_concat.wav) có tổng thời lượng $4.70\text{ s}$ với 8 vị trí biên nối:
- $t_1 = 0.64\text{ s}$ (mời $\to$ khách)
- $t_2 = 0.96\text{ s}$ (khách $\to$ hàng)
- $t_3 = 1.47\text{ s}$ (hàng $\to$ số)
- $t_4 = 2.08\text{ s}$ (số $\to$ hai)
- $t_5 = 2.56\text{ s}$ (hai $\to$ ba)
- $t_6 = 3.07\text{ s}$ (ba $\to$ đến)
- $t_7 = 3.62\text{ s}$ (đến $\to$ quầy)
- $t_8 = 4.19\text{ s}$ (quầy $\to$ năm)

### 3.2. Phân tích Hiện tượng Âm học tại Biên nối
Quan sát trên biểu đồ Waveform và Spectrogram ([`figures/baseline_boundaries.png`](./figures/baseline_boundaries.png)):
1. **Tiếng nổ / Click âm học:** Tại các vị trí chuyển tiếp, sự chênh lệch giá trị biên độ tức thời giữa mẫu cuối của từ trước và mẫu đầu của từ sau tạo ra một hàm bước nhảy biên độ lớn (step discontinuity). Trên phổ tần số, bước nhảy này biểu hiện dưới dạng một vạch năng lượng trải rộng ở mọi dải tần số cao.
2. **Lệch pha (Phase Mismatch):** Hai sóng âm độc lập không đồng bộ chu kỳ làm triệt tiêu cục bộ hoặc gây gián đoạn chu kỳ dao động tự nhiên của dây thanh.
3. **Bước nhảy Formant:** Không có sự chuyển tiếp mượt mà của các dải cộng hưởng Formant $F_1, F_2$ (coarticulation), gây cảm giác câu nói bị cắt vụn cơ học.

---

## 4. Phần C - Unit Selection & Boundary Smoothing

### 4.1. Thiết lập Hàm Chi phí (Cost Function)
Hệ thống giải bài toán tối ưu hóa tổng chi phí:
$$\theta^* = \arg\min_{\theta} \sum_{j=1}^J C_{target}(\theta_j) + \lambda \sum_{j=1}^{J-1} C_{join}(\theta_j, \theta_{j+1})$$
- **Target Cost:** Sai lệch so với thời lượng mong muốn $\bar{d} = 0.45\text{ s}$ và $\text{RMS}_{target} = 0.08$.
- **Join Cost:** Độ không tương thích phổ log-magnitude tại $25\text{ ms}$ cửa sổ Hanning ở biên:
  $$D_{spec} = \frac{1}{K} \sum_{k=1}^K [L_{left}[k] - L_{right}[k]]^2$$

### 4.2. Tìm kiếm Chuỗi Candidate bằng Dynamic Programming
Lưới tìm kiếm gồm 9 vị trí, mỗi vị trí có 2 candidate. Thuật toán quy hoạch động Viterbi tìm ra chuỗi candidate có tổng chi phí nhỏ nhất:
- `mời`: `moi_02`
- `khách`: `khach_02`
- `hàng`: `hang_02`
- `số`: `so_02`
- `hai`: `hai_02`
- `ba`: `ba_02`
- `đến`: `den_02`
- `quầy`: `quay_02`
- `năm`: `nam_02`
- **Tổng chi phí tối ưu (Total Cost):** $184.22$.

### 4.3. Thí nghiệm Cross-fade Smoothing (TN1)
Công thức hòa trộn tuyến tính trên $N$ mẫu ($N = F_s \cdot \text{xfade\_ms} / 1000$):
$$y[n] = (1 - \alpha[n]) x_1[n] + \alpha[n] x_2[n], \quad \alpha[n] = \frac{n}{N-1}$$

| Mức Cross-fade | File âm thanh | Đặc điểm Waveform tại biên | Cảm nhận khi nghe |
| :---: | :---: | :--- | :--- |
| **0 ms** | [`xfade_00ms.wav`](./audio_output/xfade_00ms.wav) | Biên độ gãy khúc đứng đột ngột | Tiếng click/pop rất gắt và rõ |
| **5 ms** | [`xfade_05ms.wav`](./audio_output/xfade_05ms.wav) | Bước nhảy biên độ giảm bớt | Click giảm đáng kể nhưng còn hơi ráp |
| **15 ms** | [`xfade_15ms.wav`](./audio_output/xfade_15ms.wav) | Đường chuyển tiếp biên độ hình sin mượt mà | Hoàn toàn êm tai, phụ âm đầu giữ nguyên vẹn |
| **30 ms** | [`xfade_30ms.wav`](./audio_output/xfade_30ms.wav) | Vùng chồng lấn quá rộng ($480\text{ samples}$) | Phụ âm nổ / tắc bị mờ, méo pha nhẹ |

$\implies$ **Kết luận:** Mức **$15\text{ ms}$** là tối ưu nhất và được xuất ra làm file chuẩn [`best_concat.wav`](./audio_output/best_concat.wav).

---

## 5. Phần D - Ngữ điệu (Prosody) & Mô hình Source-Filter

### 5.1. Ước lượng Tần số Cơ bản F0
- Khảo sát đoạn nguyên âm hữu thanh "a" trong từ "ba":
- Tính hàm tự tương quan $R_{xx}[k]$ trên khung thời gian $62.5\text{ ms}$.
- Đỉnh cực đại thứ nhất xuất hiện tại trễ lag $N_p = 78\text{ mẫu}$.
- Tần số cơ bản đo được:
  $$F_0 = \frac{F_s}{N_p} = \frac{16000}{78} \approx 205.13\text{ Hz}$$
  Kết quả hoàn toàn trùng khớp với tần số giọng đọc nữ tự nhiên tiếng Việt ($\sim 200 - 210\text{ Hz}$).

### 5.2. Thí nghiệm Biến đổi Prosody (TN3 & TN4)
Quan hệ tần số cơ bản và số semitone biến thiên $\Delta s$:
$$r = 2^{\Delta s / 12}, \quad F_{0, new} = r \cdot F_{0, old}$$
1. **Biến đổi tốc độ (Speed / Duration):**
   - $0.8\times$ (chậm lại $25\%$): [`prosody_speed_0.8x.wav`](./audio_output/prosody_speed_0.8x.wav), giữ nguyên cao độ giọng, nhịp điệu rõ ràng, dễ nghe.
   - $1.2\times$ (nhanh hơn $20\%$): [`prosody_speed_1.2x.wav`](./audio_output/prosody_speed_1.2x.wav), giọng đọc nhanh nhẹn, không bị méo tiếng.
2. **Biến đổi cao độ (Pitch Shift):**
   - $-2\text{ semitones}$ ($r \approx 0.891$): [`prosody_pitch_minus2.wav`](./audio_output/prosody_pitch_minus2.wav), giọng trầm ấm hơn.
   - $+2\text{ semitones}$ ($r \approx 1.122$): [`prosody_pitch_plus2.wav`](./audio_output/prosody_pitch_plus2.wav), giọng cao và sáng hơn.
   - Phân tích đường bao phổ ([`figures/f0_autocorrelation_source_filter.png`](./figures/f0_autocorrelation_source_filter.png)) chứng minh các đỉnh cộng hưởng Formant ($F_1, F_2$) được bảo toàn nguyên vẹn, chỉ có khoảng cách giữa các vạch họa âm thay đổi.

---

## 6. Phần E - Tổng hợp Tham số dựa trên HMM (HMM Parametric Synthesis)

### 6.1. Nguyên lý Hoạt động & Thuật toán MLPG
- Trong chế độ Generation, HMM nhận chuỗi ngữ cảnh từ vựng và kích hoạt chuỗi trạng thái tương ứng.
- Mỗi state phát sinh vector tham số âm học đa chiều $O_t = [c_t^T, \Delta c_t^T]^T \sim \mathcal{N}(\mu_s, \Sigma_s)$.
- Thuật toán Maximum Likelihood Parameter Generation (MLPG) giải bài toán tối ưu với ràng buộc động $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$. Lực căng vi phân này làm triệt tiêu các bước nhảy bậc thang giữa các trạng thái, tạo ra quỹ đạo tham số $F_0(t)$ và Formants ($F_1, F_2, F_3$) trơn tru liên tục ([`figures/hmm_parameter_trajectories.png`](./figures/hmm_parameter_trajectories.png)).
- Waveform được tái tạo qua mô hình Source-Filter: Glottal pulse excitation kết hợp 3 bộ lọc cộng hưởng cộng dồn. Kết quả lưu tại [`hmm_tts.wav`](./audio_output/hmm_tts.wav).

### 6.2. Thí nghiệm Điều khiển Tham số Giọng nói
- **Scale Duration $1.2\times$:** [`hmm_tts_dur_scale.wav`](./audio_output/hmm_tts_dur_scale.wav) - Thời lượng tăng từ $4.86\text{ s}$ lên $5.66\text{ s}$, giọng nói chậm rãi tự nhiên mà không cần thuật toán co dãn mẫu sóng.
- **Scale $F_0$ $1.1\times$:** [`hmm_tts_f0_scale.wav`](./audio_output/hmm_tts_f0_scale.wav) - Tần số cơ bản nâng lên toàn dải, thể hiện tính mềm dẻo tuyệt đối của mô hình tham số.

---

## 7. Phần F - Tổng hợp Tiếng nói Học sâu (Neural TTS Inference)

### 7.1. Kiến trúc Hai Giai đoạn
Hệ thống học sâu hiện đại sử dụng pipeline:
$$\text{Text} \to \text{Text Frontend} \to \text{Acoustic Model (Mel-Spectrogram 80 bins)} \to \text{Neural Vocoder (HiFi-GAN)} \to \text{Waveform}$$

### 7.2. Kết quả Thử nghiệm trên 3 Câu
| ID | Văn bản đầu vào | Thời lượng Audio | Latency xử lý | Real-Time Factor (RTF) | File Audio đầu ra |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Câu 1** | *"Mời khách hàng số 23 đến quầy 5"* | $2.59\text{ s}$ | $0.081\text{ s}$ | $0.031$ | [`neural_tts.wav`](./audio_output/neural_tts.wav) |
| **Câu 2** | *"Mời khách hàng số 104 đến quầy 2"* | $3.07\text{ s}$ | $0.082\text{ s}$ | $0.027$ | [`neural_tts_cau2.wav`](./audio_output/neural_tts_cau2.wav) |
| **Câu 3** | *"Xin mời khách hàng số 79 đến quầy 1"* | $2.82\text{ s}$ | $0.081\text{ s}$ | $0.029$ | [`neural_tts_cau3.wav`](./audio_output/neural_tts_cau3.wav) |

$\implies$ Chỉ số **$\text{RTF} \ll 1.0$** chứng minh hệ thống có khả năng xử lý thời gian thực vượt trội (sinh 1 giây âm thanh chỉ mất khoảng $0.03\text{ giây}$). Biểu đồ Mel-Spectrogram ([`figures/neural_tts_melspectrogram.png`](./figures/neural_tts_melspectrogram.png)) thể hiện tính liên tục hoàn hảo, đường bao Formant sắc nét và không có bất kỳ khuyết tật cơ học nào.

---

## 8. Phần G - Đánh giá Toàn diện, So sánh 3 Phương pháp & Listening Test

### 8.1. Bảng Tổng hợp So sánh 3 Phương pháp
| Tiêu chí Đánh giá | Concatenative Synthesis (Unit Selection) | HMM Parametric Synthesis | Neural TTS (Deep Learning) |
| :--- | :--- | :--- | :--- |
| **Độ rõ âm (Intelligibility)** | **4.8 / 5.0** (Phát âm rất rõ ràng từ mẫu thu thật) | **4.9 / 5.0** (Chuẩn xác tuyệt đối theo từng state) | **4.9 / 5.0** (Cực kỳ rõ ràng, chuẩn ngữ âm) |
| **Độ tự nhiên (Naturalness)** | **3.4 / 5.0** (Ngữ điệu từng từ tốt nhưng toàn câu còn rời rạc) | **2.8 / 5.0** (Có âm hưởng robot, hơi mờ/rè đặc trưng vocoder) | **4.8 / 5.0** (Tương đương giọng người thật 95-98%) |
| **Tính liên tục tại biên (Continuity)**| **3.7 / 5.0** (Phụ thuộc cross-fade, vẫn có thể lệch pha) | **4.5 / 5.0** (MLPG tạo trajectory trơn tru) | **5.0 / 5.0** (Hoàn hảo tuyệt đối, không có biên nối) |
| **Độ trễ xử lý (Latency)** | **Rất thấp (< 20 ms)** | **Thấp (~ 50-80 ms)** | **Khá (~ 80-150 ms)** |
| **Khả năng biến đổi Prosody** | Rất khó, dễ bị méo tiếng / buzziness | Cực kỳ linh hoạt, biến đổi tham số tức thời | Rất mạnh mẽ qua conditioning vectors |
| **Dung lượng lưu trữ** | Lớn (Vài trăm MB đến GB cho kho mẫu sóng) | Rất nhỏ gọn (Vài MB ma trận Gauss) | Trung bình (Vài chục MB trọng số mạng) |

### 8.2. Kết quả Listening Test & MOS ([`ratings.csv`](./ratings.csv))
- **Neural TTS:** Điểm MOS trung bình **$4.89 / 5.0$** (được đánh giá cao nhất về độ tự nhiên và cảm xúc truyền cảm).
- **Concatenative Best:** Điểm MOS trung bình **$3.94 / 5.0$** (chất lượng rất tốt trong miền hẹp, ngữ âm rõ ràng).
- **HMM Parametric:** Điểm MOS trung bình **$3.41 / 5.0$** (độ rõ âm xuất sắc nhưng điểm tự nhiên bị giới hạn bởi chất giọng robotic).

---

## 9. Phần H - Lời giải Chi tiết 10 Câu hỏi Lý thuyết Báo cáo

### Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?
**Trả lời:**
1. **Lỗi đứt gãy biên nối (Boundary Discontinuity):** Sự sai lệch biên độ tức thời, lệch pha và bước nhảy cao độ ($F_0$ jump) giữa các đoạn ghi âm riêng rẽ gây ra xung nhiễu âm học.
2. **Thiếu đường nét ngữ điệu toàn câu (Lack of Sentence-level Prosody):** Mỗi từ đơn lẻ được thu âm với ngữ điệu độc lập. Trong lời nói tự nhiên, câu luôn có quy luật giảm cao độ về cuối câu ($F_0$ declination) và các điểm nhấn trọng âm câu (sentence focus). Khi ghép nối cơ học, câu nói bị rời rạc, thiếu nhịp điệu thống nhất.
3. **Mất đi hiện tượng đồng cấu âm (Coarticulation):** Cơ quan phát âm di chuyển liên tục giữa các âm tố. Việc ghép các từ độc lập làm mất các đoạn chuyển tiếp Formant tự nhiên, tạo cảm giác chắp vá thô ráp.

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
**Trả lời:**
- **Target Cost:** Đo mức độ phù hợp giữa candidate trong kho với yêu cầu mục tiêu của vị trí cần tổng hợp (ngữ cảnh âm vị, thời lượng mong muốn, cao độ mục tiêu). Đảm bảo câu tổng hợp đúng nội dung và đúng ngữ điệu.
- **Join Cost:** Đo độ êm, mượt mà tại điểm tiếp giáp giữa hai candidate kề nhau (sai lệch phổ log-magnitude, chênh lệch $F_0$ và RMS). Đảm bảo không bị click và gián đoạn.
- **Nếu chỉ tối ưu Join Cost:** Hệ thống sẽ ưu tiên chọn những unit "dễ ghép với nhau nhất" mà phớt lờ yêu cầu ngữ cảnh và ngữ điệu. Hậu quả là mối nối có thể rất êm nhưng câu nói bị sai lệch hoàn toàn về nhịp điệu (duration quá dài/ngắn), cao độ biến động bất thường hoặc phẳng lì, thậm chí chọn nhầm biến thể phát âm.

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
**Trả lời:**
Các phụ âm (đặc biệt là phụ âm tắc vô thanh $/p, t, k/$ và phụ âm xát $/s, kh/$) có đặc trưng âm học diễn ra cực kỳ ngắn (tiếng nổ plosive burst chỉ $10 - 20\text{ ms}$, dải VOT rất hẹp). Nếu cửa sổ cross-fade quá dài ($30 - 50\text{ ms}$), năng lượng nguyên âm của từ trước sẽ phủ lấp và làm phẳng các đỉnh năng lượng đột biến (transient cues) của phụ âm từ sau, kèm theo hiện tượng méo pha triệt tiêu, khiến phụ âm đầu từ bị triệt tiêu và người nghe không thể nhận diện được từ ngữ.

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
**Trả lời:**
Theo mô hình $X(z) = E(z) \cdot H(z)$:
- Nguồn kích thích $E(z)$ sinh ra do sự dao động của dây thanh âm, có tần số lặp lại là $F_0$. Phổ của nguồn là các vạch họa âm cách nhau $F_0$.
- Bộ lọc $H(z)$ đại diện cho khoang cộng hưởng thanh quản, các đỉnh cộng hưởng tự nhiên chính là các Formant ($F_1, F_2, F_3$), quy định âm sắc nguyên âm.
- Hai thành phần này độc lập về cơ chế vật lý: khi người nói thay đổi cao độ ($F_0$), các vạch họa âm dịch chuyển nhưng hình dạng khoang miệng không đổi nên các đỉnh Formant vẫn giữ nguyên vị trí tần số. Do đó, thuật toán TTS phải thay đổi khoảng cách giữa các pitch pulse mà không được làm biến dạng đường bao phổ Formant để tránh hiệu ứng "giọng sóc chuột".

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
**Trả lời:**
- **Cơ chế PSOLA:** Cắt các đoạn sóng ngắn $2 T_0$ quanh các mốc pitch marks bằng cửa sổ Hanning. Để đổi pitch thành $T_{new}$, các đoạn sóng được đặt lại ở khoảng cách $T_{new}$ và overlap-add lại. Dạng sóng cục bộ được bảo tồn nên formant không đổi, nhưng tần số lặp lại đã chuyển thành $1/T_{new}$.
- **Buzziness ở phụ âm xát hữu thanh:** Voiced fricative (/v, z/) vừa chứa dao động tuần hoàn dây thanh vừa chứa luồng khí xoáy hỗn loạn ngẫu nhiên. Khi kéo dài thời lượng, việc lặp đi lặp lại một đoạn nhiễu ngẫu nhiên ở chu kỳ cố định sẽ biến nhiễu thành tín hiệu tuần hoàn nhân tạo, gây ra tiếng kim loại rè rè khó chịu (buzziness).

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
**Trả lời:**
Nếu chỉ dùng đặc trưng tĩnh $c_t$, nghiệm ML tại mỗi frame bằng chính vector kỳ vọng của state: $c_t = \mu_s$, tạo ra bước nhảy bậc thang gãy khúc giữa các state. Khi thêm đặc trưng động $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$, bài toán MLPG trở thành tối ưu hóa có ràng buộc vi phân liên kết giữa các frame lân cận. Ràng buộc động này đóng vai trò như một hàm làm mịn (smoothing regularization), buộc quỹ đạo $c_t$ phải chuyển dịch liên tục và trơn tru giữa các phân bố Gauss.

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
**Trả lời:**
Concatenative TTS không tái tạo âm thanh mà ghép trực tiếp các mẫu sóng thật. Để một mối nối tự nhiên, candidate phải thỏa mãn đồng thời: cùng ngữ cảnh âm vị (coarticulation), cùng cao độ $F_0$, cùng tốc độ và vị trí trong câu. Nếu inventory nhỏ, hệ thống buộc phải chọn các unit lệch ngữ cảnh, gây join cost cao và méo ngữ điệu. Do đó, cần cơ sở dữ liệu khổng lồ (hàng chục đến hàng trăm giờ thu âm) để bao phủ đầy đủ mọi tổ hợp ngữ cảnh âm vị trong ngôn ngữ.

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.
**Trả lời:**
1. **Cơ chế sinh âm:** Concatenative TTS tìm kiếm và ghép nối cơ học các mẫu sóng tự nhiên có sẵn trong cơ sở dữ liệu; Neural TTS là mô hình sinh sâu nhân tạo hoàn toàn (Acoustic model sinh Mel-spectrogram và Neural vocoder sinh waveform mẫu theo mẫu).
2. **Tính liên tục & Biên nối:** Concatenative TTS luôn đối mặt với nguy cơ lỗi click, lệch pha tại biên nối; Neural TTS có tính liên tục toàn câu hoàn hảo tuyệt đối, không có khái niệm mối nối và xử lý hiện tượng đồng cấu âm xuất sắc.

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
**Trả lời:**
1. **Vi phạm nguyên tắc kiểm soát biến độc lập (Fairness):** Model tải trên mạng có thể dùng dữ liệu thu âm phòng thu chuyên nghiệp, tần số lấy mẫu $48\text{ kHz}$ thay vì $16\text{ kHz}$, tập dữ liệu hàng triệu câu. So sánh như vậy không phản ánh đúng bản chất kỹ thuật của thuật toán.
2. **Tính tái lập (Reproducibility):** Model trên Internet có thể thay đổi phiên bản, phụ thuộc môi trường GPU phức tạp, không đảm bảo tính độc lập và khả năng chạy lại bài thực hành offline của sinh viên.

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
**Trả lời:**
Hệ thống có MOS cao (giọng truyền cảm, mượt mà, tự nhiên) nhưng Intelligibility thấp (một số phụ âm bị nuốt hoặc mờ nhẹ):
1. **Lồng tiếng nhân vật nền trong Game / Phim hoạt hình (NPC Ambient Speech):** Giọng nói nhằm tạo bầu không khí sống động và cảm xúc bối cảnh, người chơi đã có phụ đề trên màn hình để nắm thông tin.
2. **Ứng dụng kể chuyện ru ngủ / Thiền định (Sleep & Meditation Apps):** Nơi ngữ điệu êm dịu, ấm áp và thư giãn được ưu tiên hàng đầu, nội dung quen thuộc và không đòi hỏi độ chính xác tuyệt đối của từng chi tiết kỹ thuật.
- *Không thể dùng cho:* Cảnh báo an ninh, y tế, thông báo sân bay/nhà ga (nơi intelligibility bắt buộc phải đạt 100%).

---

## 10. Kết luận

Bài thực hành Lab 4 đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra:
- Xây dựng hoàn chỉnh pipeline Text-to-Speech từ chuẩn hóa văn bản, ghép nối waveform, mô hình tham số HMM đến suy luận học sâu Neural TTS.
- Phân tích toán học và thực nghiệm chi tiết cho tất cả các hiện tượng âm học (Click, Phase mismatch, Formant shift, MLPG smoothing, Mel-spectrogram).
- Sản phẩm nộp đầy đủ gồm: Notebook [`Lab04_2351260657.ipynb`](./Lab04_2351260657.ipynb), thư mục [`audio_output/`](./audio_output/), thư mục [`figures/`](./figures/) và bảng điểm [`ratings.csv`](./ratings.csv).

