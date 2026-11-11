# Sắc Việt: tô màu ảnh tư liệu lịch sử Việt Nam

Đây là đồ án khóa luận về tô màu ảnh đen trắng tư liệu lịch sử Việt Nam bằng học sâu. Chúng tôi tự thiết kế mạng **VietColorNet** và so sánh nó với **DDColor** (mô hình công bố sẵn của Kang và cộng sự, ICCV 2023) làm đối chứng. Kèm theo là một trang web nhỏ để thử tô màu ảnh.

Mạng làm việc trong không gian màu CIE Lab: nhận kênh sáng L của ảnh, dự đoán hai kênh màu a, b, rồi ghép lại với L gốc, nên độ phân giải và chi tiết sáng của ảnh được giữ nguyên.

Kết luận ngắn gọn (số liệu ở mục Kết quả): VietColorNet nhẹ hơn và nhanh hơn DDColor khoảng 2,5 lần, ít sai màu hơn ở biên vật thể trên phim tư liệu, nhưng người xem thấy màu của DDColor gốc tự nhiên hơn. Màu cây cỏ và bầu trời vẫn là điểm yếu của cả hai.

## Các mô hình

Trang web có ba mô hình để chọn:

- **VietColorNet · Lịch sử** (mặc định): `checkpoints/models/vietcolornet_historical.pth`, 384 px. Huấn luyện thêm trên ảnh màu hiện đại Việt Nam, ảnh quân đội và khung hình phim tư liệu ViCoW.
- **VietColorNet · Tươi**: `checkpoints/models/vietcolornet_vivid.pth`. Bản không dùng phim tư liệu, màu đậm hơn nhưng ám vàng nâu nhiều hơn. Giữ làm đối chứng.
- **DDColor gốc**: trọng số công bố, không tinh chỉnh, khai báo trong `outputs/reports/baseline_selection.json` (có kiểm tra SHA-256).

Ngoài ra có pipeline suy luận (`colorization/enhance.py`, `colorization/segmentation.py`): phục hồi độ sáng, làm mịn màu theo vùng, dùng Segment Anything (SAM) tách vật thể để màu không tràn qua ranh giới, rồi khử ám màu. Pipeline này áp dụng cho VietColorNet.

## Kiến trúc VietColorNet

Mã nằm trong `colorization/model.py`. Các phần chính:

1. Bộ mã hóa ResNet-34 (trọng số ImageNet của torchvision).
2. Nhánh biên dùng toán tử Scharr khả vi, giúp giữ ranh giới vật thể.
3. Strip pooling để lấy ngữ cảnh theo chiều ngang và dọc.
4. Bộ giải mã có cổng biên: thông tin biên điều tiết đặc trưng khi giải mã.
5. Chú ý bảng màu (CPCA): các truy vấn màu học được đưa ngữ cảnh màu vào bộ giải mã.
6. Đầu ra tách hướng màu và độ đậm (chroma), có hiệu chỉnh chroma theo vùng và theo ảnh.
7. Cầu nối ngữ nghĩa DINOv2-S/14 (đóng băng).

Các thành phần bên ngoài: ResNet-34, DINOv2 và SAM ViT-B đều là mô hình công khai (xem cuối file). Chúng không phải mô hình tô màu.

## Dữ liệu

| Nguồn | Dùng để |
|---|---|
| Viet-Cultural-VQA, KTVIC | ảnh màu hiện đại Việt Nam (train, validation, test) |
| Wikimedia Commons | ảnh Việt Nam có giấy phép, giữ thông tin tác giả trong `attribution.csv` |
| Ảnh quân đội Việt Nam (Wikimedia) | train, 667 ảnh sau khi lọc cờ, huy hiệu, bản đồ, ảnh đen trắng |
| ImageNet chọn lọc (64×64) | train, 36.797 ảnh của 32 lớp quân sự và cảnh Việt Nam |
| ViCoW, 4 phim tư liệu | train, validation và test lịch sử |

Bản huấn luyện cuối dùng 62.172 ảnh train, 1.025 ảnh validation và 1.193 ảnh test (757 ảnh hiện đại và 436 khung phim).

**Cách chia ViCoW.** Cách chia gốc (1.327/187/382) chia ngẫu nhiên theo khung hình, và cả bốn phim đều có mặt ở mọi tập. Chúng tôi thấy 75% khung test (288/382) có một khung train cách không quá 2 khung hình trong cùng phim, nên điểm số bị thổi phồng. Vì vậy dự án chia theo phim: train trên *Những người viết huyền thoại* và *Vào Nam ra Bắc*, validation trên *Mùi cỏ cháy*, test trên *Hà Nội 12 ngày đêm*. Khung train được làm nét nhẹ và căn lại màu theo biên (`scripts/prepare/correct_vicow_colors.py`); validation và test giữ nguyên.

Màu "thật" của phim ViCoW là màu của phim cũ (nhạt hơn nhiều so với ảnh hiện đại), nên điểm trên phim đo độ giống phim cũ, không đo độ đẹp hay độ thật.

Dữ liệu không nằm trên Git (hàng chục GB). Cách chuẩn bị từng nguồn ghi trong `data/sources/README.md` và các script ở `scripts/prepare/`.

## Cài đặt

Cần Python 3.11 và GPU NVIDIA có CUDA (đã thử trên RTX 3050 Laptop 6 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài PyTorch và TorchVision bản có CUDA từ trang pytorch.org trước, sau đó:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Checkpoint không nằm trên Git. Trọng số DDColor và SAM tải bằng:

```powershell
python scripts/prepare/download_ddcolor.py --model ddcolor_paper
python scripts/prepare/download_sam.py
```

Hai checkpoint VietColorNet phải được đặt vào `checkpoints/models/` (xem phần Các mô hình). `verify_product.py` kiểm tra checkpoint và checksum còn thiếu trước khi chạy ứng dụng.

## Chạy ứng dụng

```powershell
python verify_product.py
python server.py --open-browser
```

Trang web mở ở `http://127.0.0.1:8000`; cũng có thể bấm đúp `start_demo.bat`. Tùy chọn: `--host 0.0.0.0` để máy khác trong LAN truy cập, `--no-sam` để tắt chống lem (mỗi ảnh SAM mất khoảng 4–6 giây trên RTX 3050), `--checkpoint` để dùng checkpoint VietColorNet khác.

Giao diện Gradio cũ vẫn còn: `python demo.py --open-browser` (cổng 7860).

Ba chế độ xử lý: Hiện đại (mặc định: phục hồi, tô màu, chống lem, khử ám màu và tăng độ tươi), Rực rỡ (màu đậm hơn) và Nguyên bản (chỉ chống lem). Ảnh kết quả giữ độ phân giải gốc, tối đa 40 megapixel.

## Huấn luyện và đánh giá

Chuẩn bị ImageNet và huấn luyện các baseline:

```powershell
python scripts/prepare/prepare_data.py
python train.py --config configs/resnet34.json --output checkpoints/runs/resnet34
```

Huấn luyện thử nhanh để kiểm tra pipeline (kết quả không dùng để chọn mô hình): thêm `--train-limit 16 --val-limit 8` vào lệnh trên với `configs/ddcolor_smoke.json`.

Huấn luyện "Lịch sử" từ checkpoint "Tươi":

```powershell
python scripts/prepare/correct_vicow_colors.py
python scripts/prepare/prepare_v15_vicow_protocol.py
python fine_tune_vietnam.py `
  --config configs/vietcolornet_v15_vicow.json `
  --protocol data/protocols/vietnam_modern_vicow_v15/manifest.json `
  --imagenet-manifest data/processed/imagenet_targeted_v10.json `
  --source-checkpoint checkpoints/models/vietcolornet_vivid.pth `
  --output checkpoints/runs/vietcolornet_v15_vicow
```

Mỗi epoch mất khoảng 12 phút trên GPU 6 GB; nên đóng các ứng dụng khác dùng GPU. "Tươi" là kết quả của các vòng huấn luyện trước đó; config của các vòng này (V6 refined, V8, V9, V10, V12) vẫn ở `configs/`.

Đánh giá (đều chạy trên ảnh chỉ còn kênh sáng, so với màu gốc):

```powershell
# so sánh nhanh nhiều checkpoint
python scripts/evaluate/compare_checkpoints.py --set modern-test --set vicow-test `
  vivid=checkpoints/models/vietcolornet_vivid.pth historical=checkpoints/models/vietcolornet_historical.pth

# khoảng tin cậy bootstrap, kiểm định cặp, so với DDColor gốc
python scripts/evaluate/statistical_comparison.py `
  historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection

# độ lệch theo seed huấn luyện
python scripts/evaluate/seed_runs.py 1 2

# ám màu (da, vùng xám, cây, trời) và độ bền với ảnh xuống cấp
python scripts/evaluate/colour_cast.py historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection
python scripts/evaluate/degradation_robustness.py historical=checkpoints/models/vietcolornet_historical.pth ddcolor=@selection
```

Khảo sát người dùng: `scripts/evaluate/user_study/build_study.py` dựng trang so sánh mù (thêm `--form-url` và `--form-entry` để câu trả lời tự gửi về một Google Form), `analyze_study.py` phân tích file JSON hoặc file CSV xuất từ form.

## Kết quả

Tất cả số liệu dưới đây đo trên tập test, ảnh tối đa 640 px, đầu vào chỉ có kênh sáng. Báo cáo đầy đủ nằm trong `outputs/reports/`.

### So sánh theo sai số màu (3 seed huấn luyện)

Khoảng tin cậy 95% lấy bootstrap theo khối (12 khung liền nhau của một phim tính là một khối), kiểm định hoán vị theo khối có hiệu chỉnh Holm. "Lịch sử" được huấn luyện lại với ba seed (2026, 1, 2); bảng ghi trung bình ± độ lệch chuẩn. Seed phát hành (2026) là seed tốt nhất trong ba (ΔE 9,56 trên phim), nên nên dùng số trung bình.

| Tập | Độ đo (thấp hơn là tốt) | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|---|
| Phim chưa thấy (436) | ΔE2000 | 10,21 ± 0,57 | 11,03 | 11,43 |
| | sai số tại biên | 8,31 ± 0,58 | 10,22 | 12,08 |
| Ảnh hiện đại (757) | ΔE2000 | 12,39 ± 0,29 | 12,61 | 11,77 |
| | sai số tại biên | 12,50 ± 0,15 | 12,75 | 12,91 |

Trên phim, "Lịch sử" thấp hơn DDColor gốc về sai số tại biên ở cả ba seed (thấp hơn 3,2 đến 4,4), đó là kết quả ổn định nhất. Về ΔE trên phim, lợi thế so với DDColor chỉ ở mức biên với hai seed (p = 0,03), và so với "Tươi" thì hai trong ba seed không đạt ý nghĩa. Trên ảnh hiện đại, DDColor gốc có ΔE thấp hơn ở cả ba seed. Độ lệch giữa seed chỉ phản ánh bước tinh chỉnh cuối; checkpoint xuất phát dùng chung nên chưa được lặp lại.

Hậu xử lý giúp cả ba mô hình một chút và không đổi thứ hạng. SAM không làm giảm ΔE hay sai số tại biên, nên các độ đo này không chứng minh được lợi ích của SAM.

### Khảo sát người dùng

11 người xem 20 ảnh (5 ảnh tư liệu, 8 khung phim, 7 ảnh hiện đại), mỗi ảnh có ba bản tô màu xáo thứ tự ẩn danh và chọn bản có màu tự nhiên và thật nhất. Cả ba cùng pipeline.

| Mô hình | Được chọn | Khoảng tin cậy 95% |
|---|---|---|
| DDColor gốc | 73,2% | 63,2% – 81,4% |
| VietColorNet · Tươi | 15,9% | 11,4% – 21,4% |
| VietColorNet · Lịch sử | 10,9% | 5,5% – 17,3% |

DDColor gốc được chọn nhiều nhất ở cả ba nhóm ảnh, và 10 trên 11 người chọn nó ở 60–90% số ảnh. Hạn chế: 11 người quen nhau, vị trí A/B/C chưa cân bằng hoàn toàn, và khảo sát dùng chế độ Nguyên bản chứ không phải chế độ Hiện đại mặc định của trang web. Khảo sát không đo riêng việc lem màu.

### Ám màu và da (757 ảnh hiện đại, đầu ra thô)

Số dương nghĩa là lệch về phía vàng, hoặc mất màu xanh.

| Chỉ số | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|
| Vùng xám, b* (đúng bằng 0) | +6,2 | +7,8 | +4,4 |
| Cây cỏ, lệch a* | +20,2 | +21,1 | +18,4 |
| Bầu trời, lệch b* | +26,4 | +28,9 | +21,1 |
| Da, độ đậm so với thật (đúng bằng 1) | 0,77 | 0,90 | 1,04 |

Cả ba mô hình đều làm cây cỏ và bầu trời mất màu xanh; da của VietColorNet nhạt hơn thật. Mặt nạ da ước lượng theo màu nên lẫn cả vật không phải da.

### Độ bền với ảnh xuống cấp và hiệu năng

Khi đầu vào bị làm mờ, thêm nhiễu hạt, phai tương phản, xước và nén JPEG mạnh, "Lịch sử" bền hơn DDColor ở ảnh hiện đại mức vừa nhưng không bền hơn ở mức nặng, và kém bền hơn trên phim (tăng sai số +2,35 ở mức vừa, +3,06 ở mức nặng so với DDColor). Trên phim, sai số của DDColor còn giảm khi ảnh phai, nhiều khả năng vì đầu ra nhạt hơn nên gần màu phim cũ hơn; đây là suy luận từ số đo, chưa kiểm chứng.

Về tốc độ (RTX 3050 6 GB, ảnh 640 px, đầu ra thô, chưa tính SAM): VietColorNet 54 ms/ảnh với 48,6 triệu tham số; DDColor gốc 133 ms/ảnh với 55,0 triệu tham số.

## Giới hạn

- Tô màu có nhiều đáp án hợp lý. Ảnh đen trắng không lưu thông tin màu, nên kết quả là màu hợp lý, không phải màu lịch sử xác thực. Màu quân phục, cờ và trang phục đặc biệt dễ sai vì đỏ, xanh đậm và nâu có cùng độ sáng trên ảnh xám.
- Màu cây cỏ và bầu trời vẫn bị mất xanh (xem bảng ám màu). Chúng tôi từng thử giảm ám màu bằng cân bằng màu hiếm và đầu ra phân loại màu, nhưng chưa cải thiện được.
- Ảnh lịch sử không có đáp án màu độc lập. Điểm trên phim đo độ giống màu phim cũ.
- ImageNet trong dữ liệu train chỉ có 64×64 rồi phóng lên 384 px, nên là nguồn mờ nhất.
- DDColor gốc không học dữ liệu Việt Nam còn VietColorNet có, nên đây không phải so sánh cùng ngân sách huấn luyện.
- Ảnh hư hỏng nặng, mất mảng hoặc trầy xước lớn cần thêm bước phục hồi riêng.
- Tập `Hà Nội 12 ngày đêm` từng là development test ở các phiên bản đầu; từ phiên bản V9 không dùng để huấn luyện hay chọn mô hình. Các tập dùng để chọn kiến trúc chỉ nên coi là development set.

## Cấu trúc thư mục

```text
colorization/     mô hình, dữ liệu, suy luận, chống lem, SAM, metric
configs/          cấu hình huấn luyện (baseline và chuỗi tới VietColorNet "Tươi", "Lịch sử")
scripts/
  prepare/        thu thập dữ liệu, dựng protocol, tải trọng số
  evaluate/       so sánh mô hình, thống kê, ám màu, độ bền, khảo sát người dùng
tests/            unit test
third_party/      mã nguồn DDColor (giữ nguyên giấy phép)
web/              trang web (HTML, CSS, JS)
examples/         ảnh mẫu
docs/             báo cáo khóa luận
data/             dữ liệu (không lưu trên Git)
checkpoints/      mô hình (không lưu trên Git)
outputs/          kết quả sinh ra; chỉ vài báo cáo được giữ trên Git
server.py         máy chủ web        demo.py   giao diện Gradio
train.py          huấn luyện baseline ImageNet
fine_tune_vietnam.py   huấn luyện VietColorNet trên dữ liệu Việt Nam
evaluate.py       đánh giá trên ImageNet / Việt Nam
verify_product.py kiểm tra checkpoint và checksum
```

`.venv/`, `data/`, `checkpoints/`, `outputs/` và file `*.pth` bị Git bỏ qua vì quá nặng; chỉ vài báo cáo được thêm có chủ đích.

## Phần mềm bên thứ ba

- **DDColor** (`third_party/DDColor/`, Apache-2.0, commit `2adb63f`): X. Kang, T. Yang, W. Ouyang, P. Ren, L. Li, X. Xie, "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders", ICCV 2023. Repo: <https://github.com/piddnad/DDColor>.
- **Segment Anything** ViT-B (Apache-2.0): A. Kirillov et al., ICCV 2023. Chỉ dùng khi suy luận để tách ranh giới vật thể, không tham gia dự đoán màu.
- **DINOv2** ViT-S/14 (Meta, Apache-2.0): làm cầu nối ngữ nghĩa cố định.
- **ResNet-34** (torchvision) với trọng số ImageNet.

Khi dùng ảnh Wikimedia Commons, giữ nguyên thông tin tác giả và giấy phép trong `attribution.csv`.
