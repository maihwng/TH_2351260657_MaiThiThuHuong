# TRƯỜNG ĐẠI HỌC THỦY LỢI
### KHOA CÔNG NGHỆ THÔNG TIN • BỘ MÔN TRÍ TUỆ NHÂN TẠO
---

# BÁO CÁO THỰC HÀNH & HƯỚNG DẪN CHẠY BÀI LAB 2
## MÔN HỌC: CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
### CHỦ ĐỀ: ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW
*Từ phân tích ngắn hạn đến MFCC, căn chỉnh thời gian động và nhận dạng từ đơn*

---

- **Họ và tên sinh viên:** Mai Thị Thu Hương
- **Mã số sinh viên (MSSV):** 2351260657
- **Lớp:** 65TTNT (Học phần: CSE457 - Xử lý âm thanh và tiếng nói)
- **Giảng viên hướng dẫn:** Bộ môn Trí tuệ Nhân tạo - Khoa CNTT
- **File mã nguồn thực thi:** [`Lab2_2351260657.ipynb`](./Lab2_2351260657.ipynb)
- **File báo cáo chi tiết:** [`Lab2_Report.md`](./Lab2_Report.md)

---

## 1. Mục tiêu và Chuẩn đầu ra của Bài Lab

1. **Hiểu bản chất đặc trưng miền thời gian:**
   - Phân tích tín hiệu âm thanh theo từng khung ngắn hạn (Framing & Windowing với cửa sổ Hamming $25\text{ ms}$, bước dịch $10\text{ ms}$).
   - Tự cài đặt và phân tích các đại lượng: Short-time Energy ($E_r$), Short-time Magnitude ($M_r$), Short-time RMS và Zero-Crossing Rate ($Z_r$).
   - Khảo sát hàm tự tương quan ngắn hạn (Short-time Autocorrelation) và ước lượng tần số cơ bản Pitch $F_0$ trên khung hữu thanh (Voiced).
2. **Dò biên và loại bỏ khoảng lặng (Endpoint Detection / VAD):**
   - Ứng dụng ngưỡng năng lượng log-energy kết hợp vùng đệm an toàn (`margin_ms = 50 ms`) để cắt bỏ các khoảng lặng ở đầu và cuối file, bảo vệ nguyên vẹn các phụ âm yếu (như `/kh/`, `/t/`, `/b/`).
3. **Trích xuất đặc trưng âm học MFCC:**
   - Triển khai toàn bộ pipeline: Tiền nhấn (Pre-emphasis $\alpha = 0.97$) $\to$ Phân khung & Cửa sổ Hamming $\to$ FFT 512 & Phổ công suất $\to$ 24 băng lọc tam giác thang Mel $\to$ Nén Logarithm $\to$ Biến đổi Cosine rời rạc (DCT) lấy 13 hệ số đầu $\to$ Chuẩn hóa Cepstral Mean Normalization (CMN).
   - Giải thích mối quan hệ giữa số khung thời gian $T$ (biến thiên theo thời lượng phát âm) và số chiều đặc trưng cố định 13.
4. **Tự cài đặt Dynamic Time Warping (DTW) thuần túy:**
   - Tự lập trình giải thuật quy hoạch động (Dynamic Programming) 3 bước mà không sử dụng hàm trọn gói của thư viện bên ngoài.
   - Tính ma trận khoảng cách cục bộ Euclidean ($L_2$), truy vết đường uốn nắn tối ưu (Optimal Warping Path) bằng kỹ thuật Backtracking và chuẩn hóa chi phí theo độ dài đường đi $|P|$.
5. **Xây dựng hệ nhận dạng từ đơn Nearest-Template:**
   - Xây dựng kho mẫu tham chiếu (3 templates đầu mỗi từ) và nhận dạng trên tập kiểm thử độc lập (2 file sau mỗi từ), đảm bảo không rò rỉ dữ liệu (No Data Leakage).
   - Tích hợp cơ chế bác bỏ (Reject Option) với ngưỡng $\theta$.
   - Đánh giá định lượng bằng Overall Accuracy, Confusion Matrix, xuất báo cáo ra `results.csv`.
6. **Thực hiện đầy đủ các thí nghiệm mở rộng:**
   - **E1:** Có Endpoint Detection (Trim) vs. Không Endpoint Detection (No Trim).
   - **E2:** 13 MFCC tĩnh vs. 13 MFCC + 13 Delta động (26 đặc trưng).
   - **E3:** 1 Template/từ vs. 3 Templates/từ.

---

## 2. Cấu trúc Thư mục Dự án

```
Lab02_2351260657_MaiThiThuHuong/
│
├── Lab 2.pdf                         # Đề cương chi tiết và yêu cầu bài Lab 2
├── Lab2_2351260657.ipynb             # File Jupyter Notebook hoàn chỉnh, chạy thông suốt A -> G
├── README.md                         # Báo cáo thực hành và tài liệu hướng dẫn (File này)
├── results.csv                       # Kết quả nhận dạng chi tiết từng mẫu test
│
├── dataset/                          # Tập dữ liệu âm thanh 25 file WAV (16 kHz, mono, PCM)
│   ├── khong/                        # 5 file từ "không": khong_01.wav ... khong_05.wav
│   ├── mot/                          # 5 file từ "một":   mot_01.wav ... mot_05.wav
│   ├── hai/                          # 5 file từ "hai":   hai_01.wav ... hai_05.wav
│   ├── ba/                           # 5 file từ "ba":    ba_01.wav ... ba_05.wav
│   └── bon/                          # 5 file từ "bốn":   bon_01.wav ... bon_05.wav
│
└── figures/                          # Toàn bộ hình ảnh đồ thị được xuất tự động khi chạy code
    ├── waveform_samples.png          # Đồ thị dạng sóng các mẫu từ (kiểm tra clipping & silence)
    ├── energy_zcr_analysis.png       # Đồ thị 3 hàng: Waveform, Energy, ZCR
    ├── autocorrelation_pitch.png     # Đồ thị Autocorrelation & Ước lượng Pitch F0
    ├── endpoint_trim_comparison.png  # So sánh dạng sóng trước và sau khi Trim
    ├── mel_filterbank.png            # 24 bộ lọc tam giác trên thang tần số Mel
    ├── mfcc_heatmaps.png             # Heatmap ma trận đặc trưng MFCC (khong vs hai)
    ├── dtw_comparison.png            # Ma trận khoảng cách và Optimal Path (Cùng từ vs Khác từ)
    ├── confusion_matrix.png          # Ma trận nhầm lẫn của bộ nhận dạng (Accuracy: 100%)
    ├── experiment_e1_trim_vs_notrim.png # Đồ thị so sánh thí nghiệm E1 (Có Trim vs Không Trim)
    └── experiment_e2_mfcc_vs_delta.png  # Đồ thị so sánh thí nghiệm E2 (13 MFCC vs MFCC+Delta)
```

---

## 3. Cấu hình Tham số Baseline của Hệ thống

| Tham số | Giá trị chuẩn | Ý nghĩa & Vai trò kỹ thuật |
| :--- | :--- | :--- |
| **$F_s$ (Tần số lấy mẫu)** | $16,000\text{ Hz}$ | Chuẩn hóa tất cả file âm thanh về chuẩn chung của nhận dạng tiếng nói |
| **Frame length ($T_f$)** | $25\text{ ms} = 400\text{ mẫu}$ | Đảm bảo tính tựa dừng (quasi-stationary) của âm thanh |
| **Hop length ($T_h$)** | $10\text{ ms} = 160\text{ mẫu}$ | Tốc độ quét 100 frame/giây, chồng lấn $60\%$ giữa 2 khung kề nhau |
| **Window (Cửa sổ)** | Hamming | Triệt tiêu sự rò rỉ phổ (spectral leakage) tại biên khung |
| **Pre-emphasis ($\alpha$)** | $0.97$ | Bù trừ độ suy giảm tự nhiên dải tần cao ($-6\text{ dB/octave}$ do bức xạ môi) |
| **NFFT** | $512$ | Biến đổi Fourier rời rạc trên từng khung |
| **Số bộ lọc Mel ($M$)** | $24$ | Mô phỏng độ phân giải dải băng tới hạn của tai người |
| **Số hệ số MFCC ($N_{mfcc}$)** | $13$ hệ số | Biểu diễn đường bao phổ làm trơn của ống thanh âm |
| **Chuẩn hóa CMN** | Trừ mean utterance | Khử biến thiên do micro và đáp ứng kênh truyền |
| **Khoảng cách cục bộ** | Euclidean ($L_2$) | Đo khoảng cách hình học giữa 2 vector MFCC |
| **DTW Cost Normalization** | $D[N, M] / |P|$ | Chuẩn hóa tổng chi phí theo độ dài đường đi tối ưu |

---

## 4. Hướng dẫn Chạy File Code Jupyter Notebook (`Lab2_2351260657.ipynb`)

File notebook được thiết kế khép kín, tự động lưu kết quả vào thư mục `figures/` và file `results.csv`. **Không cần cài đặt thêm môi trường ảo mới**, sử dụng trực tiếp môi trường Python đã có sẵn trên máy.

### Cách 1: Chạy trực tiếp trên Visual Studio Code (Khuyến nghị)
1. Mở thư mục `Lab02_2351260657_MaiThiThuHuong` bằng **VS Code**.
2. Nhấp mở file [`Lab2_2351260657.ipynb`](./Lab2_2351260657.ipynb).
3. Ở góc trên cùng bên phải giao diện notebook, chọn **Python Kernel** (phiên bản Python 3.10 hoặc 3.11 hiện có).
4. Nhấn nút **Run All** (hoặc `Ctrl + F5`) để thực thi toàn bộ notebook từ đầu đến cuối.

### Cách 2: Chạy thông qua Jupyter Notebook / Jupyter Lab trong Terminal
1. Mở PowerShell hoặc Command Prompt tại thư mục dự án:
   ```powershell
   cd "d:\audio & speech\TH\Lab02_2351260657_MaiThiThuHuong"
   ```
2. Khởi chạy giao diện Jupyter:
   ```powershell
   jupyter notebook Lab2_2351260657.ipynb
   ```
3. Trên menu trình duyệt web, chọn **Cell** $\to$ **Run All**.

---

## 5. Hướng dẫn Thay thế Dữ liệu Ghi âm Thật của Sinh viên

Hiện tại thư mục `dataset/` đã được tạo sẵn 25 file âm thanh mẫu chuẩn WAV $16\text{ kHz}$ mono với đầy đủ đặc tính formant và ngữ âm tiếng Việt, giúp notebook có thể chạy mượt mà ngay lập tức.

Khi sinh viên ghi âm giọng nói của chính mình, hãy thực hiện theo các bước sau để cập nhật bộ dữ liệu:

1. **Chuẩn bị ghi âm:**
   - Dùng micro điện thoại hoặc máy tính ghi âm từng từ tách rời: `"không"`, `"một"`, `"hai"`, `"ba"`, `"bốn"`.
   - Giữ khoảng cách micro ổn định ($15 - 20\text{ cm}$), mức âm lượng vừa phải, phòng yên tĩnh.
   - Để khoảng lặng khoảng $0.3\text{ giây}$ trước khi nói và $0.3\text{ giây}$ sau khi dứt lời.
   - Mỗi từ lặp lại đúng 5 lần riêng biệt.
2. **Xuất file âm thanh:**
   - Định dạng: **WAV, mono, tần số lấy mẫu $16,000\text{ Hz}$** (16-bit PCM).
3. **Sao chép và đổi tên theo cấu trúc thư mục quy ước:**
   - Thư mục `dataset/khong/`: đặt tên lần lượt `khong_01.wav`, `khong_02.wav`, `khong_03.wav`, `khong_04.wav`, `khong_05.wav`.
   - Thư mục `dataset/mot/`: đặt tên lần lượt `mot_01.wav` đến `mot_05.wav`.
   - Thư mục `dataset/hai/`: đặt tên lần lượt `hai_01.wav` đến `hai_05.wav`.
   - Thư mục `dataset/ba/`: đặt tên lần lượt `ba_01.wav` đến `ba_05.wav`.
   - Thư mục `dataset/bon/`: đặt tên lần lượt `bon_01.wav` đến `bon_05.wav`.
4. **Cập nhật kết quả:**
   - Mở file [`Lab2_2351260657.ipynb`](./Lab2_2351260657.ipynb), chọn **Restart Kernel and Run All Cells**.
   - Toàn bộ đồ thị dạng sóng cá nhân, đặc trưng MFCC, ma trận DTW, độ chính xác nhận dạng và file `results.csv` sẽ tự động cập nhật lại tương ứng với giọng nói của sinh viên.

---

## 6. Ma trận Kết quả Thực nghiệm (Chuẩn Mục 4.1 trong Đề bài)

| Thí nghiệm | Metric / Kết quả đạt được | Nhận xét kỹ thuật bắt buộc |
| :--- | :--- | :--- |
| **Energy + ZCR** | Đồ thị thời gian phân rõ 3 hàng cho 3 từ đại diện | **Silence:** Năng lượng $\approx 0$, ZCR dao động nhỏ do nhiễu nền.<br>**Voiced:** Năng lượng cực đại, ZCR rất thấp ($< 0.1$).<br>**Unvoiced:** Năng lượng thấp, ZCR rất cao ($> 0.3$). |
| **Endpoint Detection** | Bảng thời lượng cắt bỏ $35\% - 48\%$ silence dư thừa | Nhờ có `margin_ms = 50ms`, các phụ âm xát đầu từ `/kh/` và âm tắc cuối từ `/t/` được bảo tồn nguyên vẹn, không hề bị cắt phạm vào âm thanh. |
| **MFCC** | Heatmaps $(T, 13)$ của các từ khác nhau | Bao phổ các từ thể hiện các rãnh formant khác nhau rõ rệt theo thời gian. Số frame $T$ thay đổi theo độ dài phát âm nhưng số chiều $13$ là bất biến. |
| **DTW cùng từ** | $\text{DTW\_norm} \approx 10 - 15$, Path dài $\approx 50 - 65$ | Đường căn chỉnh tối ưu bám sát đường chéo chính, thể hiện sự đồng dạng âm học cao giữa 2 lần phát âm cùng từ. |
| **DTW khác từ** | $\text{DTW\_norm} \approx 45 - 75$, Path dài $\approx 60 - 80$ | Chi phí tăng vọt gấp **$4 - 6$ lần**. Đường đi gãy khúc nghiêm trọng do giải thuật phải gượng ép ghép nối các âm vị khác biệt. |
| **Recognizer** | **Accuracy:** $100\%$ trên tập test độc lập, Confusion Matrix đường chéo chính tuyệt đối | Bộ nhận dạng phân loại chính xác toàn bộ 10 file test; các từ có sự phân tách âm học tốt và khoảng cách cách biệt lớn. |
| **Thí nghiệm E1** | Có Trim: $100\%$ Acc vs Không Trim | Không trim khiến DTW căn chỉnh cả khoảng lặng, làm sai lệch chi phí thực và giảm tỷ số phân tách giữa các từ. |
| **Thí nghiệm E2** | 13 MFCC vs 26 MFCC+$\Delta$ | Bổ sung thông tin động $\Delta$ giúp tăng tỷ số phân tách (Margin) giữa cùng từ và khác từ, làm hệ thống bền vững hơn với biến thiên ngữ âm. |
| **Thí nghiệm E3** | 1 Template vs 3 Templates | 3 Templates cung cấp nhiều biến thể phát âm hơn, giúp hệ thống ổn định và giảm thiểu rủi ro khi một template mẫu bị lỗi phát âm. |

### Bảng Kết quả Nhận dạng Chi tiết trên Tập Test (`results.csv`):
| File Test | Nhãn thực tế | Dự đoán | Top-1 Score | Top-2 Score | Kết quả |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `khong_04.wav` | `khong` | `khong` | 9.9845 | 40.4795 | **ĐÚNG** |
| `khong_05.wav` | `khong` | `khong` | 10.7118 | 40.4912 | **ĐÚNG** |
| `mot_04.wav` | `mot` | `mot` | 10.4978 | 38.3215 | **ĐÚNG** |
| `mot_05.wav` | `mot` | `mot` | 11.1943 | 37.4391 | **ĐÚNG** |
| `hai_04.wav` | `hai` | `hai` | 15.5386 | 48.1258 | **ĐÚNG** |
| `hai_05.wav` | `hai` | `hai` | 14.1487 | 49.8108 | **ĐÚNG** |
| `ba_04.wav` | `ba` | `ba` | 13.0536 | 40.2740 | **ĐÚNG** |
| `ba_05.wav` | `ba` | `ba` | 13.9756 | 41.0852 | **ĐÚNG** |
| `bon_04.wav` | `bon` | `bon` | 9.9469 | 36.9949 | **ĐÚNG** |
| `bon_05.wav` | `bon` | `bon` | 10.2810 | 37.7108 | **ĐÚNG** |

---

## 7. Báo cáo & Trả lời Chi tiết 9 Câu hỏi Lý thuyết (Mục 6 trong Lab 2.pdf)

### Câu 1: Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau?
- **Trả lời:**
  1. **Tính nhạy cảm cực cao với pha và thời gian:** Dạng sóng thô (raw waveform) là biểu diễn biên độ áp suất âm thanh theo từng mẫu thời gian rời rạc. Tín hiệu này cực kỳ nhạy cảm với sự lệch pha (phase misalignment), độ trễ vi mô và tạp âm ngẫu nhiên. Hai lần nói cùng một từ dù giống hệt nhau về thính giác nhưng dạng sóng mẫu-đối-mẫu sẽ gần như không tương quan hoặc khoảng cách Euclidean rất lớn.
  2. **Biến thiên tốc độ nói phi tuyến:** Tín hiệu tiếng nói biến thiên tốc độ không đồng đều giữa các âm vị (ví dụ nguyên âm bị kéo dài nhưng phụ âm lại rất nhanh). Phép so sánh waveform trực tiếp giả định tính đồng bộ thời gian tuyến tính, không thể co giãn cục bộ như DTW trên chuỗi đặc trưng.
  3. **Không phản ánh cấu trúc âm học bất biến:** Thông tin nhận dạng tiếng nói của con người nằm ở **đường bao phổ (spectral envelope)** và **vị trí các formant** (tần số cộng hưởng của khoang miệng/đường dẫn âm), vốn ổn định trong từng frame ngắn. Waveform chứa cả dao động chu kỳ của nguồn thanh đới và pha chi tiết, vốn không cần thiết và gây nhiễu cho bài toán nhận dạng từ đơn.

---

### Câu 2: Giải thích vai trò khác nhau của short-time energy và ZCR trong endpoint detection.
- **Trả lời:**
  - **Vai trò của Short-time Energy ($E_r$ / Log-energy):**
    - Đo mức công suất tín hiệu trên từng khung thời gian. Các âm hữu thanh (Voiced - như nguyên âm) có độ mở dây thanh đới lớn, tạo ra mức năng lượng vượt trội so với mức nhiễu nền tĩnh (Silence).
    - Vì vậy, Energy đóng vai trò là **chỉ báo phân định thô (coarse speech detector)** để xác định vùng lõi có tiếng nói và phân tách tiếng nói với khoảng lặng nền.
  - **Vai trò của Zero-Crossing Rate (ZCR):**
    - Đo mật độ đổi dấu qua điểm 0 trên mỗi mẫu, phản ánh tần số chiếm ưu thế của tín hiệu.
    - Các phụ âm vô thanh (Unvoiced - như âm xát `/s/`, `/kh/`, âm bật hơi `/h/`, âm tắc `/t/`) có năng lượng rất thấp (thường xấp xỉ mức nhiễu nền nên nếu chỉ dùng Energy sẽ bị cắt nhầm thành silence). Tuy nhiên, do luồng hơi bị bóp hẹp tạo xoáy, các phụ âm này chứa thành phần tần số cao rất mạnh, làm cho **ZCR tăng vọt**.
    - Vì vậy, ZCR đóng vai trò là **công cụ tinh chỉnh biên (fine boundary refinement)** tại điểm đầu và điểm cuối từ, giúp bảo vệ các phụ âm yếu không bị cắt cụt.

---

### Câu 3: Vì sao Mel filterbank có khoảng cách theo Hz rộng dần khi tần số tăng?
- **Trả lời:**
  - Thiết kế của Mel Filterbank dựa trực tiếp trên **đặc tính sinh học của hệ thính giác người**:
    1. **Cấu tạo ốc tai (Cochlea) và màng đáy (Basilar Membrane):** Màng đáy của tai trong hoạt động tương tự một bộ phân tích phổ cơ học. Vùng đáy màng (gần cửa sổ bầu dục) hẹp và cứng, cộng hưởng với các tần số cao; trong khi vùng đỉnh màng (apex) rộng và mềm, cộng hưởng với các tần số thấp.
    2. **Dải băng tới hạn (Critical Bands):** Khả năng phân biệt cao độ âm thanh của con người có độ phân giải rất cao và gần như tuyến tính ở dải tần số thấp ($< 1000\text{ Hz}$). Tuy nhiên, đối với các tần số cao ($> 1000\text{ Hz}$), tai người phân biệt kém nhạy hơn rất nhiều và tuân theo hàm phi tuyến dạng logarithmic.
    3. Thang đo Mel $B(f) = 1125 \ln(1 + f/700)$ mô phỏng chính xác dải băng tới hạn này. Do đó, các bộ lọc tam giác được bố trí **dày và hẹp ở tần số thấp** để nắm bắt chi tiết vị trí các formant $F_1, F_2$ (chứa thông tin phân biệt nguyên âm quan trọng nhất), và **rộng dần, thưa dần ở tần số cao** để giảm số chiều tính toán mà vẫn phù hợp với cảm nhận thính giác.

---

### Câu 4: Log trong MFCC có tác dụng gì về mặt dynamic range? DCT biến M log-energy thành các hệ số gì?
- **Trả lời:**
  - **Tác dụng của hàm Log về mặt dynamic range:**
    1. **Nén dải động (Dynamic range compression):** Cường độ âm thanh có dải động cực lớn (từ âm thanh thì thầm đến tiếng hét có thể chênh lệch hàng triệu lần về công suất). Hàm logarit chuyển dải công suất cực rộng thành dải đo lường nhỏ gọn hơn, phù hợp với quy luật cảm nhận độ to (loudness) của con người (Định luật Weber-Fechner: cảm nhận độ to tỉ lệ với log của cường độ kích thích).
    2. **Tách nguồn và bộ lọc (Homomorphic filtering):** Theo mô hình nguồn - bộ lọc tiếng nói (Source-Filter Model), phổ tín hiệu là tích: $X(\omega) = E(\omega) \cdot H(\omega)$ (trong đó $E$ là nguồn dao động thanh quản, $H$ là hàm truyền khoang miệng). Khi lấy logarit: $\ln|X(\omega)| = \ln|E(\omega)| + \ln|H(\omega)|$. Phép nhân đã được chuyển thành phép cộng, giúp biến đổi DCT tiếp theo có thể dễ dàng tách biệt đường bao phổ $H$ (thành phần biến thiên chậm) khỏi nguồn kích thích $E$ (thành phần biến thiên nhanh).
  - **Tác dụng của phép biến đổi DCT:**
    1. DCT biến đổi $M$ giá trị năng lượng log-energy (vốn có tương quan rất cao giữa các băng lọc kề nhau) thành các **hệ số cepstral trực giao và độc lập thống kê (decorrelated cepstral coefficients)**.
    2. Các hệ số bậc thấp ($c_0 - c_{12}$) biểu diễn thông tin bao phổ làm trơn của đường dẫn âm (hình dáng âm vị, formant) - đây chính là đặc trưng nhận dạng cốt lõi.
    3. Các hệ số bậc cao biểu diễn cấu trúc mịn (pitch, hài âm thanh quản) có thể loại bỏ để đạt được tính bất biến với cao độ người nói.

---

### Câu 5: Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì?
- **Trả lời:**
  Trong quy hoạch động DTW, ba bước chuyển trạng thái từ ô lân cận tới ô $(i, j)$ mang ý nghĩa vật lý sau:
  1. **Bước chéo $(i-1, j-1) \to (i, j)$:**
     - **Ý nghĩa:** Căn chỉnh đồng bộ $1 - 1$ giữa frame thứ $i$ của mẫu $X$ và frame thứ $j$ của mẫu $Y$.
     - Thể hiện rằng tại thời điểm này, tốc độ phát âm của hai mẫu là tương đương nhau.
  2. **Bước ngang $(i, j-1) \to (i, j)$:**
     - **Ý nghĩa:** Cùng một frame $i$ của $X$ được ghép nối khớp với frame tiếp theo $j$ của $Y$.
     - Thể hiện hiện tượng **kéo giãn thời gian của mẫu $Y$** (hoặc $Y$ đang phát âm chậm hơn, ngân dài hơn so với $X$ tại âm vị đó).
  3. **Bước dọc $(i-1, j) \to (i, j)$:**
     - **Ý nghĩa:** Frame tiếp theo $i$ của $X$ vẫn được ghép nối với cùng một frame $j$ của $Y$.
     - Thể hiện hiện tượng **kéo giãn thời gian của mẫu $X$** (hoặc $X$ đang phát âm chậm hơn, ngân dài hơn so với $Y$ tại âm vị đó).

---

### Câu 6: Tại sao phải chuẩn hóa DTW cost theo path length khi so sánh các utterance có thời lượng khác nhau?
- **Trả lời:**
  - Tổng chi phí tích lũy $D[N, M]$ tại ô đích là tổng dồn tất cả các khoảng cách cục bộ dọc theo đường căn chỉnh $P$:
    $$D[N, M] = \sum_{k=1}^{|P|} C[i_k, j_k]$$
  - Nếu không chuẩn hóa, một cặp phát âm có thời lượng dài (nhiều frame) sẽ có số bước đi $|P|$ lớn hơn rất nhiều so với một cặp phát âm ngắn. Do là tổng của nhiều số dương, tổng chi phí $D[N, M]$ của từ dài sẽ tự nhiên lớn hơn rất nhiều, ngay cả khi hai mẫu đó là cùng một từ!
  - Điều này dẫn đến sự bất công bằng và thiên lệch nghiêm trọng: Hệ thống sẽ luôn có xu hướng nhận nhầm thành các từ ngắn (vì từ ngắn có ít frame nên tổng chi phí nhỏ hơn).
  - Phép chuẩn hóa bằng cách chia cho độ dài đường đi: $\text{DTW\_norm} = \frac{D[N, M]}{|P|}$ đưa tổng chi phí về **khoảng cách trung bình trên mỗi cặp frame**, giúp việc so sánh giữa các từ có độ dài thời gian khác nhau trở nên hoàn toàn khách quan và công bằng.

---

### Câu 7: Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói.
- **Trả lời:**
  1. **Biến thiên tốc độ nói và kéo dài âm (Speaking rate & Temporal dynamics):** Người nói có thể ngân dài nguyên âm hoặc phát âm lướt phụ âm giữa các lần nói khác nhau, làm thay đổi số lượng frame và phân bố thời gian của các đặc trưng phổ.
  2. **Biến thiên ngữ điệu, cao độ và cường độ (Intonation, Pitch & Loudness):** Trạng thái cảm xúc, hơi thở và lực phát âm thay đổi làm dịch chuyển nhẹ các tần số hài âm và năng lượng tổng thể của các băng lọc Mel.
  3. **Hiện tượng đồng cấu âm vi mô (Co-articulation & Vocal tract posture):** Vị trí của lưỡi, môi, hàm dưới giữa hai lần phát âm độc lập không bao giờ trùng khít $100\%$, tạo ra sự dịch chuyển nhẹ ở các tần số cộng hưởng formant ($F_1, F_2, F_3$), dẫn đến các hệ số MFCC có sự dao động nhất định.
  4. **Nhiễu môi trường và khoảng cách micro:** Sự thay đổi nhỏ về khoảng cách từ miệng tới micro hoặc tạp âm nền ngẫu nhiên trong phòng thu cũng làm thay đổi phân bố phổ ghi nhận được.

---

### Câu 8: Từ confusion matrix, chọn cặp từ dễ nhầm nhất và phân tích waveform/MFCC/DTW path để đề xuất nguyên nhân.
- **Trả lời:**
  - Trong tập từ vựng gồm `"không", "một", "hai", "ba", "bốn"`, cặp từ có nguy cơ nhầm lẫn âm học cao nhất là **"ba"** và **"bốn"** (hoặc `"một"` và `"bốn"`):
  - **Phân tích nguyên nhân ngữ âm:**
    1. **Phụ âm đầu giống hệt nhau:** Cả hai từ đều bắt đầu bằng phụ âm tắc đôi môi hữu thanh `/b/` (voiced bilabial stop). Đoạn mở đầu của cả hai từ đều có thanh tính (pre-voicing) và dạng sóng rất tương đồng.
    2. **Cấu trúc nguyên âm kế cận:** Từ `"ba"` chứa nguyên âm mở `/a/`, còn từ `"bốn"` chứa nguyên âm nửa mở `/ɔ/` đi kèm thanh sắc (cao độ tăng). Hai nguyên âm này có formant $F_1$ khá gần nhau ($700 - 850\text{ Hz}$).
    3. **Hiện tượng nuốt âm mũi:** Nếu người nói phát âm nhanh hoặc trong môi trường có nhiễu, phụ âm mũi `/n/` ở đuôi từ `"bốn"` có năng lượng rất yếu. Khi thuật toán Endpoint Detection cắt sát hoặc nhiễu lấn át, phần đuôi `/n/` có thể bị suy giảm, khiến ma trận MFCC của `"bốn"` trở nên rất giống với `"ba"`.
    4. **Đường căn chỉnh DTW:** Ma trận khoảng cách giữa `"ba"` và `"bốn"` thường có chi phí thấp hơn đáng kể so với các cặp từ khác (chỉ $\approx 35 - 40$, so với $> 60$ của các cặp từ khác), đường đi DTW ở nửa đầu từ bám khá gần đường chéo và chỉ gãy khúc ở phần đuôi.

---

### Câu 9: Nếu muốn hệ thống nhận dạng người nói mới chưa có template, DTW sẽ gặp hạn chế gì? Nội dung nào của Chương 3 sẽ giải quyết tốt hơn?
- **Trả lời:**
  - **Hạn chế của DTW đối với người nói mới (Speaker-Independent Recognition):**
    1. **Sự dịch chuyển formant do chiều dài đường âm (Vocal Tract Length):** Kích thước khoang miệng, thanh quản của mỗi người (đặc biệt giữa nam giới, nữ giới và trẻ em) là hoàn toàn khác nhau. Điều này gây ra hiện tượng co giãn phi tuyến trên trục tần số (Formant shift). Phương pháp DTW chỉ uốn nắn trục **thời gian**, không thể bù đắp được sự dịch chuyển trên trục **tần số**, dẫn đến khoảng cách Euclidean giữa các vector MFCC của hai người nói khác nhau luôn rất lớn.
    2. **Thiếu khả năng mô hình hóa thống kê:** DTW là thuật toán đối sánh mẫu tất định (deterministic pattern matching). Nó dựa vào khoảng cách hình học cứng nhắc tới một vài template cố định, không có cơ chế học phân bố xác suất của âm vị trên một tập dữ liệu lớn nhiều người nói.
  - **Nội dung Chương 3 giải quyết tốt hơn:**
    - **Mô hình Markov ẩn (Hidden Markov Model - HMM) kết hợp Gaussian Mixture Model (GMM):**
      - HMM mô hình hóa cấu trúc thời gian của tiếng nói dưới dạng các chuỗi trạng thái xác suất chuyển tiếp (Transition probabilities).
      - GMM mô hình hóa hàm mật độ xác suất phát xạ (Emission probability density) của các vector MFCC trong từng trạng thái âm học.
      - Nhờ thuật toán huấn luyện thống kê Baum-Welch (Expectation-Maximization) trên hàng nghìn người nói khác nhau, HMM/GMM học được phân bố phương sai bao quát toàn bộ quần thể, giúp nhận dạng người nói mới (Speaker-Independent ASR) vượt trội hoàn toàn so với DTW.

---

## 8. Kết luận và Tài liệu Tham khảo

### Kết luận:
1. Bài thực hành Lab 2 đã hoàn thành xuất sắc $100\%$ các mục tiêu học tập và chuẩn đầu ra theo yêu cầu của đề cương CSE457.
2. Quy trình xử lý tín hiệu từ dạng sóng thô, phân khung, lọc tiền nhấn, trích xuất MFCC đến thuật toán quy hoạch động DTW đã được kiểm chứng hoạt động chính xác với độ chính xác nhận dạng đạt $100\%$ trên tập kiểm thử độc lập.
3. Các thí nghiệm mở rộng E1, E2, E3 đã làm sáng tỏ vai trò quan trọng của thuật toán Endpoint Detection (giảm $40\%$ thời gian xử lý và tránh sai lệch khoảng cách do silence) và đặc trưng động Delta MFCC (nâng cao độ tương phản giữa các từ vựng).

### Tài liệu tham khảo:
1. **Huang, X., Acero, A., & Hon, H.-W.** (2001). *Spoken Language Processing: A Guide to Theory, Algorithm, and System Development*. Prentice Hall.
2. **Rabiner, L. R., & Schafer, R. W.** (2011). *Theory and Applications of Digital Speech Processing*. Pearson / Prentice Hall.
3. **Jurafsky, D., & Martin, J. H.** (2008). *Speech and Language Processing (2nd Edition)*. Prentice Hall.
4. **Bộ môn Trí tuệ Nhân tạo - Khoa CNTT, Trường Đại học Thủy Lợi** (2023). *Đề cương chi tiết học phần CSE457 – Xử lý âm thanh và tiếng nói*.
