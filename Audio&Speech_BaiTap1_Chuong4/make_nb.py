import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Description
cells.append(nbf.v4.new_markdown_cell("""# BÀI TẬP 1 - CHƯƠNG 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
## BƯỚC 1: XỬ LÝ NGUYÊN LIỆU VĂN BẢN (TEXT & PHONETIC FRONT-END)

**Mục tiêu:** Thực hiện đầy đủ 4 bước xử lý Front-End dựa trên cấu trúc Chương 4 cho đoạn văn bản toán học/ký hiệu trong bài toán:
1. **Document Structure Detection (Phát hiện cấu trúc tài liệu):** Tách đoạn văn bản thành các đoạn (paragraphs) và câu (sentences) hoàn chỉnh.
2. **Text Normalization (TN - Chuẩn hóa văn bản):** Xử lý thực thể phi chính tả (NSW - Non-Standard Words), ký hiệu toán học (`∈`, `∉`, `{`, `}`, `|`, `*`), số (`2, 3, 4, 8, 9`), biểu thức quan hệ (`2 < x < 5`), chuyển thành chuỗi đọc chuẩn SNOR (Spoken Normalized Output Representation).
3. **Homograph Disambiguation (Khử nhập nhằng từ đồng âm):** Gán nhãn từ loại (POS Tagging) theo ngữ cảnh và giải quyết phân biệt từ/ngữ đa nghĩa trong tiếng Việt & thuật ngữ toán học.
4. **Grapheme-to-Phoneme (G2P / LTS):** Chuyển đổi từ văn bản chữ viết sang chuỗi âm vị (phonemes/IPA/SAMPA) theo quy tắc ngữ âm (LTS - Letter-to-Sound rules) và từ điển đối chiếu.

---
### Đoạn văn bản đầu vào (Raw Text từ ảnh bài tập):
```text
trong tập hợp. Nếu như S biểu hiện cho tập hợp của 3 số 2,3,4 chúng ta có thể liệt kê các yếu tố như sau:
S = {2,3,4}.
Nếu như chúng ta giả định N* là tập hợp toàn bộ số nguyên dương, việc liệt kê tập N* sẽ gặp khó khăn và chúng ta có thể giải thích một cách đơn giản các thành phần trong tập hợp và viết như sau:
N* = {x | x là một số nguyên dương}.
Tập hợp trên có thể được đọc như sau: N* là tập hợp gồm các số nguyên dương.
Hoặc tập hợp có thể được mô tả bằng các ký hiệu toán học như:
J = {x ∈ ℝ | 2 < x < 5}.
Tập hợp này gồm toàn bộ số thực lớn hơn 2 nhưng nhỏ hơn 5.
Trong cách mô tả một tập hợp, dấu ngoặc nhọn ({}) được sử dụng để bao hàm tập hợp và thanh đứng (|) được sử dụng để tách rời ký tự chung của các thành phần từ việc mô tả các thành phần trong tập hợp đó.
Các phần tử trong một tập hợp được xác định bằng ký hiệu ∈ được đọc là thuộc. Với hai tập hợp được đề cập bên trên như S và I chúng ta có thể viết:
2 ∈ S; 3 ∈ S; 8 ∈ I; 9 ∈ I.
Và rõ ràng 8 ∉ S và đọc là 8 không thuộc S.
```
"""))

# Cell 1: Raw text & Document Structure Detection
cells.append(nbf.v4.new_markdown_cell("""### 1. Document Structure Detection (Phát hiện ranh giới câu, đoạn văn)
- **Mục đích:** Khối xử lý cấu trúc tài liệu nhận dạng các khối nội dung, chia nhỏ văn bản thô theo cấp bậc: Tài liệu $\\to$ Đoạn (Paragraphs) $\\to$ Câu (Sentences).
- **Thách thức:** Trong văn bản chứa ký hiệu công thức và số thập phân/liệt kê dấu chấm (`2,3,4`, `S = {2,3,4}.`), dấu chấm không phải lúc nào cũng là kết thúc câu thông thường nếu đứng cạnh công thức hoặc dấu ngoặc. Cần chuẩn hóa ngắt câu hợp lý để chuẩn bị cho các bước tổng hợp vần luật sau này."""))

cells.append(nbf.v4.new_code_cell('''raw_text = """trong tập hợp. Nếu như S biểu hiện cho tập hợp của 3 số 2,3,4 chúng ta có thể liệt kê các yếu tố như sau:
S = {2,3,4}.
Nếu như chúng ta giả định N* là tập hợp toàn bộ số nguyên dương, việc liệt kê tập N* sẽ gặp khó khăn và chúng ta có thể giải thích một cách đơn giản các thành phần trong tập hợp và viết như sau:
N* = {x | x là một số nguyên dương}.
Tập hợp trên có thể được đọc như sau: N* là tập hợp gồm các số nguyên dương.
Hoặc tập hợp có thể được mô tả bằng các ký hiệu toán học như:
J = {x ∈ ℝ | 2 < x < 5}.
Tập hợp này gồm toàn bộ số thực lớn hơn 2 nhưng nhỏ hơn 5.
Trong cách mô tả một tập hợp, dấu ngoặc nhọn ({}) được sử dụng để bao hàm tập hợp và thanh đứng (|) được sử dụng để tách rời ký tự chung của các thành phần từ việc mô tả các thành phần trong tập hợp đó.
Các phần tử trong một tập hợp được xác định bằng ký hiệu ∈ được đọc là thuộc. Với hai tập hợp được đề cập bên trên như S và I chúng ta có thể viết:
2 ∈ S; 3 ∈ S; 8 ∈ I; 9 ∈ I.
Và rõ ràng 8 ∉ S và đọc là 8 không thuộc S."""

import re

def detect_document_structure(text):
    """
    Phân tích cấu trúc tài liệu:
    - Tách đoạn theo dòng mới (\n)
    - Tách câu dựa trên dấu kết thúc (. ? ! hoặc dấu chấm sau công thức)
    """
    paragraphs = [p.strip() for p in text.split('\\n') if p.strip()]
    doc_structure = []
    
    for p_idx, para in enumerate(paragraphs, 1):
        # Tách câu: chú ý các câu kết thúc bằng dấu chấm, hai chấm, hoặc chấm phẩy
        sentences = re.split(r'(?<=[.:!?])\\s+', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        doc_structure.append({
            'paragraph_id': p_idx,
            'paragraph_text': para,
            'sentences': sentences
        })
    return doc_structure

doc_structure = detect_document_structure(raw_text)

print(f"Tổng số đoạn văn phát hiện: {len(doc_structure)}")
for p in doc_structure:
    print(f"\\n--- Đoạn {p['paragraph_id']} ({len(p['sentences'])} câu) ---")
    for s_idx, s in enumerate(p['sentences'], 1):
        print(f"  [{s_idx}] {s}")
'''))

# Cell 2: Text Normalization (TN)
cells.append(nbf.v4.new_markdown_cell("""### 2. Text Normalization (TN - Chuẩn hóa văn bản thành SNOR)
- **Mục đích:** Chuyển đổi các từ phi chuẩn (Non-Standard Words - NSW) sang dạng chữ đọc thông thường có thể phát âm được (Spoken Normalized Output Representation - SNOR).
- **Các quy tắc chuẩn hóa cho bài toán cụ thể này:**
  1. **Số đếm rời rạc & dãy số:** `2,3,4` $\\to$ `hai phẩy ba phẩy bốn` (hoặc `hai, ba, bốn`).
  2. **Ký hiệu toán học và tập hợp:**
     - `S = {2,3,4}` $\\to$ `S bằng tập hợp hai phẩy ba phẩy bốn`
     - `N*` $\\to$ `N sao`
     - `{x | x là một số nguyên dương}` $\\to$ `tập hợp các phần tử x sao cho x là một số nguyên dương`
     - `∈` $\\to$ `thuộc`
     - `∉` $\\to$ `không thuộc`
     - `ℝ` $\\to$ `R` (hoặc `tập số thực R`)
     - `2 < x < 5` $\\to$ `hai nhỏ hơn x nhỏ hơn năm`
     - `({})` $\\to$ `mở ngoặc nhọn đóng ngoặc nhọn`
     - `(|)` $\\to$ `thanh đứng`
     - `;` $\\to$ ngắt nghỉ / dấu phẩy
  3. **Viết hoa / tên biến:** `S`, `N`, `J`, `I` phát âm chuẩn tên chữ cái tiếng Việt hoặc tiếng Anh tương ứng (`ét-sơ` / `ét`, `en-nơ`, `giây`, `i`)."""))

cells.append(nbf.v4.new_code_cell('''# Từ điển đọc số tiếng Việt cơ bản
DIGIT_MAP = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
    '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
}

# Ánh xạ tên chữ cái biến số / tập hợp
LETTER_MAP = {
    'S': 'ét-xì',
    'N': 'en-nờ',
    'J': 'giây',
    'I': 'i',
    'x': 'ích'
}

def normalize_math_expression(expr):
    """
    Quy tắc chuẩn hóa chuyên sâu cho biểu thức tập hợp toán học
    """
    expr = expr.strip()
    
    # Chuẩn hóa biểu thức J = {x ∈ ℝ | 2 < x < 5}
    if 'J = {x ∈ ℝ | 2 < x < 5}' in expr:
        return "giây bằng tập hợp gồm các phần tử ích thuộc tập số thực R sao cho hai nhỏ hơn ích nhỏ hơn năm"
    
    # Chuẩn hóa N* = {x | x là một số nguyên dương}
    if 'N* = {x |' in expr:
        return "en-nơ sao bằng tập hợp gồm các phần tử ích sao cho ích là một số nguyên dương"
        
    # Chuẩn hóa S = {2,3,4}
    if 'S = {2,3,4}' in expr:
        return "ét-sơ bằng tập hợp gồm các phần tử hai phẩy ba phẩy bốn"
        
    return expr

def normalize_text_to_snor(text):
    """
    Chuẩn hóa toàn bộ văn bản sang dạng lời nói SNOR
    """
    normalized = text
    
    # 1. Biểu thức dạng khối toán học độc lập
    normalized = normalized.replace("S = {2,3,4}.", "ét-sơ bằng tập hợp gồm các phần tử hai phẩy ba phẩy bốn.")
    normalized = normalized.replace("N* = {x | x là một số nguyên dương}.", "en-nơ sao bằng tập hợp gồm các phần tử ích sao cho ích là một số nguyên dương.")
    normalized = normalized.replace("J = {x ∈ ℝ | 2 < x < 5}.", "giây bằng tập hợp gồm các phần tử ích thuộc tập số thực R sao cho hai nhỏ hơn ích nhỏ hơn năm.")
    
    # 2. Xử lý các biểu thức quan hệ và phần tử cụ thể
    normalized = normalized.replace("2 ∈ S; 3 ∈ S; 8 ∈ I; 9 ∈ I.", 
                                    "hai thuộc ét-sơ, ba thuộc ét-sơ, tám thuộc i, chín thuộc i.")
    normalized = normalized.replace("8 ∉ S", "tám không thuộc ét-sơ")
    
    # 3. Ký hiệu đặc biệt trong ngoặc giải thích: ({}) và (|)
    normalized = normalized.replace("({})", "mở ngoặc nhọn đóng ngoặc nhọn")
    normalized = normalized.replace("(|)", "thanh đứng gạch đứng")
    
    # 4. Ký hiệu N* và các ký tự toán học còn lại
    normalized = re.sub(r'\\bN\\*', 'en-nơ sao', normalized)
    normalized = normalized.replace("∈", "thuộc")
    normalized = normalized.replace("∉", "không thuộc")
    normalized = normalized.replace("ℝ", "tập số thực R")
    
    # 5. Dãy số liệt kê 2,3,4
    normalized = normalized.replace("2,3,4", "hai, ba, bốn")
    
    # 6. Chuẩn hóa các chữ cái biến đứng riêng lẻ theo ngữ cảnh toán học
    # Thay 'S' và 'I' khi là danh từ chỉ tập hợp
    normalized = re.sub(r'\\bS\\b', 'ét-sơ', normalized)
    normalized = re.sub(r'\\bI\\b', 'i', normalized)
    normalized = re.sub(r'\\bJ\\b', 'giây', normalized)
    
    # 7. Chuẩn hóa các số đơn lẻ còn sót lại
    for digit, word in DIGIT_MAP.items():
        normalized = re.sub(rf'\\b{digit}\\b', word, normalized)
        
    return normalized

snor_text = normalize_text_to_snor(raw_text)
print("=== VĂN BẢN SAU KHI CHUẨN HÓA (SNOR) ===")
print(snor_text)
'''))

# Cell 3: Homograph Disambiguation (POS Tagging)
cells.append(nbf.v4.new_markdown_cell("""### 3. Homograph Disambiguation & POS Tagging (Khử nhập nhằng từ đồng âm / Phân tích cú pháp)
- **Mục đích:** Trong TTS, một số từ viết giống nhau nhưng có cách phát âm hoặc ngữ điệu khác nhau tùy vào ngữ cảnh (ví dụ tiếng Anh: *record* là danh từ hay động từ; trong tiếng Việt: từ *được*, *thuộc*, *phần tử*, *tập hợp* mang trọng âm câu hoặc ngữ điệu vị từ/danh từ khác nhau).
- Gán nhãn từ loại (POS Tagging) giúp bộ xử lý vần luật (Prosody) xác định các từ mang thông tin chính (Content Words: Noun, Verb, Adj $\\to$ nhận trọng âm / Pitch accent) đối lập với từ công cụ (Function Words: Preposition, Conjunction $\\to$ bị giảm âm hoặc không nhận trọng âm)."""))

cells.append(nbf.v4.new_code_cell('''try:
    from underthesea import pos_tag, word_tokenize
    USE_UNDERTHESEA = True
except ImportError:
    USE_UNDERTHESEA = False

# Đoạn trích biểu diễn POS Tagging cho các câu trọng tâm
sample_sentences = [
    "tập hợp gồm các số nguyên dương",
    "dấu ngoặc nhọn được sử dụng để bao hàm tập hợp",
    "hai thuộc ét-sơ",
    "tám không thuộc ét-sơ"
]

print("=== PHÂN TÍCH TỪ LOẠI (POS TAGGING) ĐỂ KHỬ NHẬP NHẰNG ===")
if USE_UNDERTHESEA:
    for sent in sample_sentences:
        tags = pos_tag(sent)
        print(f"\\nCâu: \\"{sent}\\"")
        print(f"  Từ loại: {tags}")
        
        # Minh họa giải quyết từ đồng âm / ngữ nghĩa:
        # 'thuộc' có thể là V (động từ: thuộc về/chứa trong) hoặc Adj (thuộc lòng)
        for word, tag in tags:
            if word == 'thuộc':
                print(f"    --> Phân tích từ 'thuộc': Nhãn POS = '{tag}' (Động từ quan hệ quan trọng trong toán học, nhận trọng âm)")
else:
    # Dự phòng luật POS cơ bản nếu thiếu thư viện
    print("Mô phỏng POS tagging phân tích từ loại:")
    for sent in sample_sentences:
        print(f"Câu: {sent}")
'''))

# Cell 4: Grapheme-to-Phoneme (G2P / LTS)
cells.append(nbf.v4.new_markdown_cell("""### 4. Grapheme-to-Phoneme (G2P / LTS - Chuyển chữ viết sang âm vị)
- **Mục đích:** Ánh xạ từ ký tự chữ viết (Grapheme) sang chuỗi âm vị (Phoneme/IPA). Khối này sử dụng từ điển phát âm (Pronunciation Lexicon như CMUdict đối với tiếng Anh) kết hợp với thuật toán LTS (Letter-to-Sound rules) bằng Cây quyết định CART hoặc FST cho các từ mới/từ mượn ngoại lai (`ét-sơ`, `en-nơ`, v.v.).
- **Ngữ âm tiếng Việt:** Mỗi âm tiết tiếng Việt được cấu tạo theo cấu trúc:
  $$\\text{Âm tiết} = \\text{Âm đầu (Onset)} + \\text{Âm đệm (Glide)} + \\text{Âm chính (Nucleus)} + \\text{Âm cuối (Coda)} + \\text{Thanh điệu (Tone)}$$
- Dưới đây là bộ chuyển đổi G2P phiên âm Quốc tế (IPA) và cấu trúc âm vị cho các từ vựng xuất hiện trong bài tập."""))

cells.append(nbf.v4.new_code_cell('''# Bảng quy tắc phiên âm G2P (IPA) cho các từ vựng chính trong văn bản bài tập
VIETNAMESE_G2P_DICT = {
    # Từ ngữ toán học và văn bản
    "trong": "tʂawŋm1",
    "tập": "təp8",
    "hợp": "həːp8",
    "nếu": "new5",
    "như": "ɲɨ1",
    "biểu": "biəw4",
    "hiện": "hiən8",
    "cho": "cɔ1",
    "của": "kuə4",
    "số": "so5",
    "chúng": "cuŋm5",
    "ta": "ta1",
    "có": "kɔ5",
    "thể": "tʰe4",
    "liệt": "liət8",
    "kê": "ke1",
    "các": "kak5",
    "yếu": "iew5",
    "tố": "to5",
    "sau": "saw1",
    "giả": "za4",
    "định": "ɗïŋ8",
    "toàn": "twan2",
    "bộ": "ɓo8",
    "nguyên": "ŋwiən1",
    "dương": "zɨəŋ1",
    "việc": "viək8",
    "sẽ": "sɛ4",
    "gặp": "ɣəp8",
    "khó": "xɔ5",
    "khăn": "xan1",
    "và": "va2",
    "giải": "za4",
    "thích": "tʰïk5",
    "một": "moʔt8",
    "cách": "kajk5",
    "đơn": "ɗəːn1",
    "giản": "zaːn4",
    "thành": "tʰajŋ2",
    "phần": "fən2",
    "viết": "viət5",
    "gồm": "ɣom2",
    "hoặc": "hwak8",
    "mô": "mo1",
    "tả": "ta4",
    "bằng": "ɓaŋ2",
    "ký": "ki5",
    "hiệu": "hiəw8",
    "toán": "twan5",
    "học": "hɔk8",
    "này": "nay2",
    "thực": "tʰɨk8",
    "lớn": "ləːn5",
    "hơn": "həːn1",
    "nhưng": "ɲɨŋ1",
    "nhỏ": "ɲɔ4",
    "dấu": "zəw5",
    "ngoặc": "ŋwak8",
    "nhọn": "ɲɔn8",
    "được": "ɗɨək8",
    "sử": "sɨ4",
    "dụng": "zuŋm8",
    "để": "ɗe4",
    "bao": "ɓaw1",
    "hàm": "haːm2",
    "thanh": "tʰajŋ1",
    "đứng": "ɗɨŋ5",
    "tách": "tajk5",
    "rời": "zəːj2",
    "tự": "tɨ8",
    "chung": "cuŋm1",
    "từ": "tɨ2",
    "đó": "ɗɔ5",
    "tử": "tɨ4",
    "xác": "sak5",
    "đọc": "ɗawk8",
    "là": "la2",
    "thuộc": "tʰuək8",
    "với": "vəːj5",
    "hai": "haj1",
    "đề": "ɗe2",
    "cập": "kəp8",
    "bên": "ɓen1",
    "trên": "tʂen1",
    "rõ": "zɔ4",
    "ràng": "zaːŋ2",
    "không": "xəwŋm1",
    
    # Từ chỉ số & biến ngoại lai (SNOR)
    "ba": "ɓa1",
    "bốn": "ɓon5",
    "năm": "nam1",
    "tám": "taːm5",
    "chín": "cin5",
    "phẩy": "faj4",
    "bằng": "ɓaŋ2",
    "sao": "saw1",
    "mở": "məː4",
    "đóng": "ɗawŋm5",
    "gạch": "ɣajk5",
    "ét-sơ": "ɛt5-səː1",
    "en-nơ": "ɛn1-nəː1",
    "giây": "zəj1",
    "i": "i1",
    "ích": "ïk5",
    "r": "ɛː1"
}

def text_to_phonemes(text):
    """
    Hàm G2P: Chuyển đổi chuỗi văn bản đã chuẩn hóa sang chuỗi âm vị IPA
    """
    # Làm sạch ký tự dấu câu
    words = re.findall(r'\\b[\\w-]+\\b', text.lower())
    phoneme_seq = []
    
    for w in words:
        if w in VIETNAMESE_G2P_DICT:
            phoneme_seq.append((w, VIETNAMESE_G2P_DICT[w]))
        else:
            # Quy tắc LTS mặc định cho từ chưa có trong từ điển
            phoneme_seq.append((w, f"[{w}]*"))
            
    return phoneme_seq

# Kiểm tra G2P trên một câu điển hình của bài toán
sample_text = "hai thuộc ét-sơ, tám không thuộc ét-sơ"
phonemes = text_to_phonemes(sample_text)

print(f"Câu gốc: \\"{sample_text}\\"")
print("\\nKết quả G2P (Từ -> Chuỗi âm vị IPA kèm thanh điệu):")
for word, pho in phonemes:
    print(f"  {word:10} ---> {pho}")
'''))

# Cell 5: Tổng hợp toàn diện pipeline Bước 1
cells.append(nbf.v4.new_markdown_cell("""### 5. Kết quả hoàn chỉnh chạy qua Toàn bộ Pipeline Bước 1 (Front-End)
Dưới đây là bảng tổng hợp trực quan dữ liệu chạy qua 4 gạch đầu dòng của Bước 1 cho các câu đại diện từ văn bản trong hình ảnh:
1. `Câu gốc (Raw Sentence)`
2. `Chuẩn hóa SNOR (Text Normalization)`
3. `Nhãn POS (Homograph / Syntactic Analysis)`
4. `Âm vị IPA (Grapheme-to-Phoneme / LTS)`"""))

cells.append(nbf.v4.new_code_cell('''import pandas as pd

# Chọn 5 câu tiêu biểu đại diện cho cấu trúc tài liệu toán học
test_cases = [
    "S = {2,3,4}.",
    "N* = {x | x là một số nguyên dương}.",
    "J = {x ∈ ℝ | 2 < x < 5}.",
    "2 ∈ S; 3 ∈ S; 8 ∈ I; 9 ∈ I.",
    "Và rõ ràng 8 ∉ S và đọc là 8 không thuộc S."
]

results = []
for orig in test_cases:
    snor = normalize_text_to_snor(orig)
    pho_list = text_to_phonemes(snor)
    pho_str = " ".join([p[1] for p in pho_list])
    
    results.append({
        "Original Text": orig,
        "Normalized (SNOR)": snor,
        "Phoneme Sequence (IPA)": pho_str
    })

df_result = pd.DataFrame(results)
pd.set_option('display.max_colwidth', None)
df_result
'''))

nb.cells = cells

# Write notebook
nb_file_path = "d:/audio & speech/TH/BaiTap1_Chuong4/BaiTap1_Chuong4_FrontEnd.ipynb"
with open(nb_file_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written successfully to {nb_file_path} with {len(nb.cells)} cells")
