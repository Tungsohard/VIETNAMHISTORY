# Sắc Việt: tô màu ảnh tư liệu lịch sử Việt Nam bằng học sâu

Ảnh và phim tư liệu về lịch sử Việt Nam phần lớn là đen trắng. Đồ án này xây dựng một mạng tô màu, **VietColorNet**, huấn luyện trên ảnh màu Việt Nam và khung hình phim tư liệu, rồi đánh giá nó so với **DDColor** (Kang và cộng sự, ICCV 2023), một mô hình công bố sẵn dùng làm đối chứng. Kèm theo là một trang web nhỏ để thử tô màu ảnh.

Ba câu hỏi mà đồ án cố trả lời:

1. Huấn luyện thêm trên dữ liệu Việt Nam có giúp tô màu ảnh tư liệu Việt Nam tốt hơn một mô hình tổng quát không?
2. Bước tách vật thể bằng Segment Anything (SAM) có làm màu bớt lem qua ranh giới vật thể không?
3. Các kết luận có ổn định khi lặp lại huấn luyện và có đứng vững dưới kiểm định thống kê không?

Kết quả tóm tắt (chi tiết ở mục 5): VietColorNet nhỏ hơn và nhanh hơn DDColor khoảng 2,5 lần, và ít sai màu hơn ở vùng biên vật thể trên phim tư liệu. Tuy vậy các độ đo sai số không cho thấy SAM có tác dụng. Màu cây cỏ và bầu trời vẫn là điểm yếu của cả hai mô hình.

## 1. Chạy thử

Cần Python 3.11 và GPU NVIDIA có CUDA (đã thử trên RTX 3050 Laptop 6 GB). Tạo môi trường, cài PyTorch bản CUDA từ pytorch.org, rồi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Một số test cần thư mục `data/` (không có trên Git) và tự bỏ qua khi thiếu nó, nên với bản clone mới kết quả bình thường là "OK (skipped=9)".

Dữ liệu và checkpoint không nằm trên Git vì quá nặng (dữ liệu hàng chục GB, mỗi checkpoint VietColorNet khoảng 400 MB). Hai trọng số công khai tải được ngay:

```powershell
python scripts/prepare/download_ddcolor.py        # DDColor-T, mô hình đối chứng
python scripts/prepare/download_sam.py            # Segment Anything ViT-B, cho bước chống lem
```

Hai file này có SHA-256 cố định; `outputs/reports/baseline_selection.json` ghi checksum của file DDColor, và `verify_product.py` đối chiếu nó. Chỉ với DDColor, trang web đã chạy được: menu khi đó chỉ có "DDColor gốc", và thêm `--no-sam` nếu chưa tải SAM.

```powershell
python verify_product.py          # kiểm tra trọng số, báo mô hình nào đang thiếu
python server.py --open-browser   # trang web tại http://127.0.0.1:8000
```

Hai mô hình VietColorNet cần các file `checkpoints/models/vietcolornet_historical.pth` và `vietcolornet_vivid.pth`. Hai file này chưa được công bố cùng mã nguồn; có thể xin từ tác giả hoặc tự huấn luyện theo mục 7. Khi chúng có mặt, chúng tự xuất hiện trong menu, và "Lịch sử" trở thành mô hình mặc định.

Có thể bấm đúp `start_demo.bat` thay cho lệnh thứ hai. `--no-sam` tắt bước chống lem (mỗi ảnh SAM thêm khoảng 4–6 giây trên RTX 3050), `--host 0.0.0.0` cho máy khác trong mạng LAN truy cập. Giao diện Gradio cũ vẫn chạy được bằng `python demo.py --open-browser`.

Ba mô hình trong menu là **VietColorNet · Lịch sử** (mặc định khi có), **VietColorNet · Tươi** và **DDColor gốc**, cùng ba chế độ xử lý: Hiện đại (mặc định), Rực rỡ (màu đậm hơn) và Nguyên bản (chỉ chống lem). Ảnh kết quả giữ độ phân giải ảnh gốc, tối đa 40 megapixel.

## 2. Phương pháp

### 2.1 Bài toán

Mạng làm việc trong không gian màu CIE Lab. Đầu vào là kênh độ sáng L của ảnh xám, đầu ra là hai kênh màu a và b. Ảnh màu được dựng lại từ L gốc và ab dự đoán, nên độ phân giải và chi tiết sáng của ảnh đầu vào không đổi. Một ảnh xám có nhiều cách tô màu hợp lý, nên mạng chỉ cần cho ra một màu hợp lý, và việc chấm điểm so với ảnh màu gốc chỉ là một thước đo gần đúng (xem mục 6).

### 2.2 Kiến trúc VietColorNet

Mã nguồn ở `colorization/model.py`. Mạng gồm:

- bộ mã hóa ResNet-34 (trọng số ImageNet của torchvision) cho đặc trưng ngữ nghĩa nhiều mức;
- nhánh biên dùng toán tử Scharr khả vi, và bộ giải mã có cổng biên, trong đó thông tin biên điều tiết đặc trưng khi giải mã, nhằm giữ ranh giới vật thể;
- strip pooling để gom ngữ cảnh theo chiều ngang và dọc;
- khối chú ý bảng màu (CPCA), trong đó các truy vấn màu học được đưa ngữ cảnh màu vào bộ giải mã;
- đầu ra tách hướng màu và độ đậm (chroma), có hiệu chỉnh chroma theo vùng và theo ảnh;
- cầu nối ngữ nghĩa DINOv2-S/14, đóng băng, thêm đặc trưng ngữ nghĩa cho bộ giải mã.

ResNet-34, DINOv2 và SAM là các mô hình công khai có sẵn, không phải mô hình tô màu. Phần dự đoán màu (nhánh biên, CPCA, bộ giải mã, đầu ra) do nhóm thiết kế.

### 2.3 Hai mô hình VietColorNet

Hai mô hình dùng chung kiến trúc và khác nhau ở dữ liệu huấn luyện. "Tươi" (`vietcolornet_vivid.pth`) được huấn luyện trên ảnh màu hiện đại Việt Nam, ảnh quân đội và một tập con ImageNet, qua nhiều vòng thực nghiệm trước đó (các config V6 refined, V8, V9, V10, V12 vẫn ở `configs/`). "Lịch sử" (`vietcolornet_historical.pth`) khởi tạo từ "Tươi" rồi tinh chỉnh thêm với khung hình phim tư liệu.

Tinh chỉnh "Lịch sử" chạy ở 384 px, BF16, batch hiệu dụng 16, tốc độ học 1,5×10⁻⁵, tối đa 6 epoch, 6.000 mẫu mỗi epoch, với trọng số lấy mẫu: Viet-Cultural-VQA 0,36, ImageNet chọn lọc 0,18, ảnh quân đội 0,16, KTVIC 0,15, ViCoW 0,10, Wikimedia 0,05. Huấn luyện dừng sớm sau epoch 3 vì điểm validation (cân bằng theo nguồn, không tính tập ViCoW validation) không cải thiện trong 3 epoch liên tiếp, và epoch 3 là bản được phát hành: nó có loss thấp nhất trên phim validation, dù điểm tổng hợp không vượt checkpoint xuất phát.

### 2.4 Pipeline suy luận

Ngoài đầu ra thô của mạng, hệ thống có các bước tùy chọn (`colorization/enhance.py`, `colorization/segmentation.py`): phục hồi kênh L (khử nhiễu, kéo dải sáng, làm nét), làm mịn màu theo vùng siêu điểm ảnh (mỗi vùng lấy màu từ lõi của nó và chỉ mượn từ vùng lân cận cùng độ sáng), dùng SAM tách vật thể để màu không được mượn qua ranh giới vật thể, rồi khử ám màu ở vùng gần trung tính và tăng nhẹ độ tươi. Khi đánh giá, các bước này chỉ bật ở những thí nghiệm nêu rõ; mặc định đánh giá đầu ra thô để chênh lệch giữa các mô hình đến từ mạng.

## 3. Dữ liệu

| Nguồn | Vai trò |
|---|---|
| Viet-Cultural-VQA, KTVIC | ảnh màu hiện đại Việt Nam (train, validation, test) |
| Wikimedia Commons | ảnh Việt Nam có giấy phép; tác giả và giấy phép ghi trong `attribution.csv` |
| Ảnh quân đội Việt Nam (Wikimedia) | train; 667 ảnh sau khi bỏ cờ, huy hiệu, bản đồ, giấy tờ và ảnh đen trắng |
| ImageNet chọn lọc, 64×64 | train; 36.797 ảnh của 32 lớp quân sự và cảnh Việt Nam |
| ViCoW (4 phim tư liệu) | train, validation và test lịch sử |

Bản huấn luyện của "Lịch sử" dùng 62.172 ảnh train, 1.025 ảnh validation và 1.193 ảnh test (757 ảnh hiện đại và 436 khung phim). Ảnh trùng và gần trùng (dHash) đã được loại. Cách chuẩn bị từng nguồn ghi ở `data/sources/README.md`.

**Cách chia ViCoW.** Cách chia gốc của bộ này (1.327/187/382) chia ngẫu nhiên theo khung hình và cả bốn phim đều có mặt ở mọi tập. Chúng tôi kiểm tra và thấy 288 trong 382 khung test (75%) có một khung train cách nó không quá 2 khung hình trong cùng phim, tức là test gần như bị lộ. Vì vậy dự án chia theo phim: train trên *Những người viết huyền thoại* và *Vào Nam ra Bắc*, validation trên *Mùi cỏ cháy*, test trên *Hà Nội 12 ngày đêm*. Khung train được làm nét nhẹ và căn lại màu theo biên (`scripts/prepare/correct_vicow_colors.py`) vì chúng được phóng từ video độ phân giải thấp; validation và test giữ nguyên bản gốc.

Màu "đáp án" của phim ViCoW là màu của phim cũ, nhạt hơn nhiều so với ảnh hiện đại (độ đậm khoảng 9–10 so với 17–19). Điểm trên phim vì thế đo mức giống màu phim cũ, chứ không phải mức đẹp hay mức thật.

## 4. Thiết lập đánh giá

**Quy trình.** Mỗi ảnh test được đổi thành ảnh chỉ còn kênh sáng, tô màu, rồi so với ảnh màu gốc. Ảnh được thu nhỏ để cạnh dài không quá 640 px.

**Độ đo** (đều thấp hơn là tốt):

- ΔE2000: sai khác màu CIEDE2000 trung bình theo ảnh;
- AB-MAE: sai số tuyệt đối trung bình của hai kênh a, b;
- sai số tại biên: AB-MAE tính trong vùng giãn 5×5 quanh biên Canny của kênh L. Đây là đại lượng thay thế cho hiện tượng lem màu, nhưng nó cũng chịu ảnh hưởng của việc màu nhìn chung giống đáp án đến đâu, nên không đo riêng được việc lem.

**Thống kê.** Khoảng tin cậy 95% lấy bằng bootstrap (5.000 lần). Với phim, các khung liền nhau tương quan mạnh nên lấy mẫu theo khối 12 khung. Chênh lệch giữa hai mô hình tính theo cặp trên cùng ảnh, kiểm định bằng hoán vị theo khối (20.000 lần), có hiệu chỉnh Holm khi so nhiều cặp. Kiểm định Wilcoxon cũng được tính nhưng không dùng để kết luận, vì nó coi các khung liền nhau là độc lập và cho p nhỏ quá mức. Để biết kết quả có ổn định không, bước tinh chỉnh cuối của "Lịch sử" được lặp lại với ba seed (2026, 1 và 2).

**Đối chứng.** DDColor-T (ConvNeXt-T, ảnh 256 px) với trọng số công bố, không tinh chỉnh.

## 5. Kết quả

### 5.1 Sai số màu và độ ổn định theo seed

Trung bình ± độ lệch chuẩn của ba seed, đầu ra thô. Seed phát hành (2026) là seed tốt nhất trong ba (ΔE 9,56 trên phim), nên con số đại diện là trung bình.

| Tập | Độ đo | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|---|
| Phim chưa thấy (436 khung) | ΔE2000 | 10,21 ± 0,57 | 11,03 | 11,43 |
| | sai số tại biên | 8,31 ± 0,58 | 10,22 | 12,08 |
| Ảnh hiện đại (757 ảnh) | ΔE2000 | 12,38 ± 0,29 | 12,61 | 11,77 |
| | sai số tại biên | 12,50 ± 0,15 | 12,75 | 12,91 |

Trên phim, sai số tại biên của "Lịch sử" thấp hơn DDColor ở cả ba seed (thấp hơn 3,2 đến 4,4, p đã hiệu chỉnh khoảng 0,0005), và thấp hơn "Tươi" ở cả ba seed. Về ΔE trên phim, lợi thế so với DDColor còn đó nhưng ở mức biên với hai seed (p = 0,03), và so với "Tươi" thì hai trong ba seed không đạt ý nghĩa. Trên ảnh hiện đại, DDColor có ΔE thấp hơn ở cả ba seed. Độ lệch giữa các seed chỉ phản ánh bước tinh chỉnh cuối, vì "Tươi" dùng chung nên chưa được lặp lại.

### 5.2 Tác dụng của hậu xử lý và SAM

Cùng ba mô hình, cùng tập test, ΔE2000 của một seed (2026):

| Tập | Cấu hình | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|---|
| Phim | thô | 9,56 | 11,03 | 11,43 |
| | + hậu xử lý | 9,44 | 10,93 | 11,07 |
| | + hậu xử lý + SAM | 9,51 | 11,03 | 11,18 |
| Ảnh hiện đại | thô | 12,19 | 12,61 | 11,77 |
| | + hậu xử lý | 12,16 | 12,57 | 11,60 |
| | + hậu xử lý + SAM | 12,15 | 12,58 | 11,48 |

Hậu xử lý cải thiện nhẹ cả ba mô hình và không đổi thứ hạng. SAM không làm giảm ΔE hay sai số tại biên (sai số biên của "Lịch sử" trên phim nhích từ 7,72 lên 7,83). Hai độ đo này không phát hiện được tác dụng của SAM, nhưng cũng chưa đủ để kết luận nó vô ích, vì chúng không đo riêng việc lem màu.

### 5.3 Ám màu

Đo trên 757 ảnh hiện đại, đầu ra thô, trên từng lớp màu của ảnh gốc: vùng xám (độ đậm thật dưới 5), cây cỏ, bầu trời, và da. Số dương nghĩa là lệch về phía vàng, hoặc mất màu xanh.

| Chỉ số | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|
| Vùng xám, b* (đúng bằng 0) | +6,2 | +7,8 | +4,3 |
| Cây cỏ, lệch a* | +20,2 | +21,1 | +18,3 |
| Bầu trời, lệch b* | +26,4 | +28,9 | +21,1 |
| Da, độ đậm so với thật (đúng bằng 1) | 0,77 | 0,90 | 1,04 |

Cả ba mô hình đều làm cây cỏ và bầu trời mất màu xanh, và da do VietColorNet tô nhạt hơn thật. Mặt nạ da ước lượng theo màu nên lẫn cả vật không phải da, vì vậy con số về da chỉ gần đúng.

### 5.4 Độ bền với ảnh xuống cấp và hiệu năng

Ảnh test được làm xuống cấp theo ba mức (nhẹ, vừa, nặng) bằng cách làm mờ, thêm nhiễu hạt, giảm tương phản, thêm vết xước dọc và nén JPEG mạnh; sai số được chấm chỉ trên màu (độ sáng dự đoán được thay bằng độ sáng thật). Tiêu chí được đặt trước khi chạy: "Lịch sử" bền hơn DDColor nếu ở mức vừa và nặng, trên mỗi tập, sai số tăng ít hơn và khoảng tin cậy của chênh lệch loại được 0 (4 phép kiểm). Chỉ một phép đạt: ảnh hiện đại, mức vừa (−0,63 ΔE, p = 0,003). Ảnh hiện đại mức nặng không khác biệt, và trên phim "Lịch sử" kém bền hơn (sai số tăng thêm +2,35 ở mức vừa và +3,06 ở mức nặng so với DDColor). Trên phim, sai số của DDColor còn giảm khi ảnh phai (11,37 → 9,92), có thể vì đầu ra của nó nhạt đi và gần màu phim cũ hơn; đây là suy đoán từ số đo, chưa kiểm chứng.

Về hiệu năng (RTX 3050 6 GB, ảnh 640 px, đầu ra thô, chưa tính SAM): VietColorNet cần 54 ms mỗi ảnh với 48,6 triệu tham số và 373 MB VRAM đỉnh; DDColor cần 133 ms với 55,0 triệu tham số và 430 MB.

## 6. Thảo luận và giới hạn

Câu hỏi 1: huấn luyện thêm trên dữ liệu Việt Nam giúp VietColorNet gần màu phim tư liệu hơn và giảm sai số ở biên trên phim, nhưng không giải quyết được ám màu (mục 5.3) và trên ảnh hiện đại không giảm được ΔE so với DDColor. Chúng tôi từng thử giảm ám màu bằng cân bằng màu hiếm và đầu ra phân loại màu, nhưng không cải thiện được. Câu hỏi 2: các độ đo hiện có không chứng minh được tác dụng của SAM, nên câu hỏi này vẫn mở. Câu hỏi 3: kết luận về sai số tại biên ổn định giữa các seed, còn kết luận về ΔE giữa hai mô hình VietColorNet thì không.

Những điều cần cân nhắc khi đọc các con số trên:

- Ảnh đen trắng không lưu thông tin màu, nên kết quả là màu hợp lý, không phải màu lịch sử xác thực. Quân phục, cờ và trang phục đặc biệt dễ sai vì đỏ, xanh đậm và nâu có cùng độ sáng trên ảnh xám.
- Ảnh lịch sử không có đáp án màu độc lập. Điểm trên phim ViCoW so với màu phim cũ, không so với màu thật.
- Màu cây cỏ và bầu trời vẫn mất xanh ở cả hai mô hình VietColorNet, và da bị tô nhạt.
- DDColor không học dữ liệu Việt Nam còn VietColorNet có, nên đây không phải so sánh cùng ngân sách huấn luyện; ImageNet trong dữ liệu train chỉ có độ phân giải 64×64 rồi phóng lên 384 px.
- Phim *Hà Nội 12 ngày đêm* từng là development test ở các phiên bản đầu; từ phiên bản V9 trở đi nó không được dùng để huấn luyện hay chọn mô hình, nhưng kết quả trên nó vẫn mang tính development.
- Số seed (ba) nhỏ.
- Ảnh hư hỏng nặng, mất mảng hoặc xước lớn cần thêm bước phục hồi riêng; chưa đo tính nhất quán theo thời gian khi tô video.

Điểm có thể dùng lại của đồ án nằm ở phương pháp: phát hiện việc cách chia ngẫu nhiên theo khung hình làm lộ test, cách chia theo phim, và quy trình đánh giá có khoảng tin cậy, kiểm định theo khối và độ lệch theo seed.

## 7. Tái lập

Các lệnh dưới đây chạy từ thư mục gốc, với dữ liệu và checkpoint đã chuẩn bị theo `data/sources/README.md`.

Huấn luyện baseline ImageNet và chạy thử nhanh (kết quả của lần chạy thử không dùng để chọn mô hình):

```powershell
python scripts/prepare/prepare_data.py
python train.py --config configs/resnet34.json --output checkpoints/runs/resnet34
python train.py --config configs/ddcolor_smoke.json --output checkpoints/runs/smoke --train-limit 16 --val-limit 8
```

Tinh chỉnh "Lịch sử" từ "Tươi":

```powershell
python scripts/prepare/correct_vicow_colors.py
python scripts/prepare/prepare_v15_vicow_protocol.py
python fine_tune_vietnam.py `
  --config configs/vietcolornet_v15_vicow.json `
  --protocol data/protocols/vietnam_modern_vicow_v15/manifest.json `
  --imagenet-manifest data/processed/imagenet_targeted_v10.json `
  --source-checkpoint checkpoints/models/vietcolornet_vivid.pth `
  --output checkpoints/runs/vietcolornet_v15_vicow
python scripts/evaluate/seed_runs.py 1 2        # hai seed còn lại; dùng epoch_03.pth
```

Mỗi epoch mất khoảng 12 phút trên GPU 6 GB; nên đóng các ứng dụng khác dùng GPU.

Đánh giá:

```powershell
python scripts/evaluate/compare_checkpoints.py --set modern-test --set vicow-test `
  vivid=checkpoints/models/vietcolornet_vivid.pth historical=checkpoints/models/vietcolornet_historical.pth
python scripts/evaluate/statistical_comparison.py `
  historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection
python scripts/evaluate/colour_cast.py historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection
python scripts/evaluate/degradation_robustness.py historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection
```

`statistical_comparison.py` nhận thêm `--mode natural --sam` để bật hậu xử lý và SAM giống nhau cho mọi mô hình. Các báo cáo số liệu nằm trong `outputs/reports/`.

`evaluate.py` và `train.py` thuộc giai đoạn baseline ImageNet ban đầu (`evaluate.py` đọc một file `model_selection.json` của giai đoạn đó) và chưa được kiểm tra lại với cấu trúc hiện tại.

## 8. Cấu trúc thư mục

```text
colorization/       mô hình, dữ liệu, suy luận, chống lem, SAM, độ đo
configs/            cấu hình huấn luyện (baseline ImageNet và các vòng tới VietColorNet)
scripts/prepare/    thu thập dữ liệu, dựng protocol, tải trọng số
scripts/evaluate/   so sánh mô hình, thống kê, ám màu, độ bền
tests/              unit test
third_party/        mã nguồn DDColor (giữ nguyên giấy phép)
web/  server.py     trang web và máy chủ
demo.py             giao diện Gradio
train.py            huấn luyện baseline ImageNet
fine_tune_vietnam.py  huấn luyện VietColorNet trên dữ liệu Việt Nam
verify_product.py   kiểm tra checkpoint và checksum
examples/           ảnh mẫu cho trang web
docs/               báo cáo khóa luận
data/  checkpoints/  outputs/    dữ liệu, mô hình, kết quả (không lưu trên Git)
```

`.venv/`, `data/`, `checkpoints/`, `outputs/` và file `*.pth` bị Git bỏ qua vì quá nặng; chỉ một số báo cáo được giữ lại có chủ đích.

## 9. Phần mềm và dữ liệu bên thứ ba

- **DDColor** (`third_party/DDColor/`, Apache-2.0, commit `2adb63f`): X. Kang, T. Yang, W. Ouyang, P. Ren, L. Li, X. Xie, "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders", ICCV 2023. <https://github.com/piddnad/DDColor>
- **Segment Anything** ViT-B (Apache-2.0): A. Kirillov và cộng sự, ICCV 2023. Chỉ dùng khi suy luận để tách ranh giới vật thể, không tham gia dự đoán màu.
- **DINOv2** ViT-S/14 (Meta, Apache-2.0): cầu nối ngữ nghĩa cố định.
- **ResNet-34** của torchvision, trọng số ImageNet.

Khi dùng ảnh Wikimedia Commons, giữ nguyên thông tin tác giả và giấy phép trong `attribution.csv`.
