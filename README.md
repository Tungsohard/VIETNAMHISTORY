# Sắc Việt — Phục chế màu ảnh tư liệu lịch sử Việt Nam

Sắc Việt là hệ thống nghiên cứu và triển khai tô màu ảnh, video tư liệu lịch sử Việt Nam bằng học sâu. Dự án phát triển **VietColorNet** như kiến trúc đề xuất của khóa luận, đồng thời dùng **DDColor** làm baseline/đối chứng và teacher trong các thí nghiệm chưng cất tri thức.

> Trạng thái hiện tại: ứng dụng web có ba mô hình trong menu: **VietColorNet · Lịch sử** (mặc định, `checkpoints/models/vietcolornet_historical.pth`, trước đây là V15 epoch 3, 384 px), **VietColorNet · Tươi** (`vietcolornet_vivid.pth`, trước đây là V12 epoch 6) và **DDColor gốc** (trọng số công bố chính thức, chưa tinh chỉnh) làm baseline thuần. DDColor V6 tinh chỉnh đã bị vô hiệu hóa. Pipeline phục hồi ảnh và chống lem màu bằng SAM chỉ áp dụng cho VietColorNet. Trên phim tư liệu chưa thấy, "Lịch sử" giảm sai số tại biên khoảng 31% so với DDColor gốc (trung bình 3 seed); trên ảnh hiện đại, DDColor gốc có ΔE thấp hơn một chút (xem mục Đánh giá thống kê). Mô hình chưa đạt cổng phân bố chroma; màu cây cỏ vẫn lệch vàng so với màu thật.

## Mục lục

- [Mục tiêu và phạm vi](#mục-tiêu-và-phạm-vi)
- [Vai trò các mô hình](#vai-trò-các-mô-hình)
- [Kiến trúc VietColorNet](#kiến-trúc-vietcolornet)
- [Dữ liệu và protocol](#dữ-liệu-và-protocol)
- [Cài đặt](#cài-đặt)
- [Chạy ứng dụng Sắc Việt](#chạy-ứng-dụng-sắc-việt)
- [Huấn luyện và đánh giá](#huấn-luyện-và-đánh-giá)
- [Kết quả nghiên cứu đã kiểm chứng](#kết-quả-nghiên-cứu-đã-kiểm-chứng)
- [Thay checkpoint vận hành](#thay-checkpoint-vận-hành)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Giới hạn](#giới-hạn)
- [Phần mềm bên thứ ba](#phần-mềm-bên-thứ-ba)

## Mục tiêu và phạm vi

Dự án tập trung vào các mục tiêu sau:

- Tô màu ảnh xám và video tư liệu lịch sử Việt Nam.
- Giữ biên vật thể, hạn chế lem màu và giảm hiện tượng bạc màu.
- So sánh VietColorNet với DDColor trên cùng protocol và cùng bộ metric.
- Thử nghiệm domain adaptation trên dữ liệu Việt Nam, dữ liệu Wikimedia Commons và ảnh/video lịch sử.
- Cung cấp ứng dụng Gradio “Sắc Việt” để chạy suy luận cục bộ và đối chiếu A/B.
- Chạy dưới dạng ứng dụng web (Gradio) trên trình duyệt.

Hệ thống dùng không gian màu CIE Lab: mô hình nhận kênh độ sáng `L` và dự đoán hai kênh màu `a/b`. Ảnh kết quả được ghép lại và chuyển về RGB, đồng thời giữ kích thước đầu vào khi suy luận.

## Vai trò các mô hình

### VietColorNet

- Là **mô hình đề xuất do nhóm tự thiết kế và triển khai** trong `colorization/model.py`.
- Không phải là một biến thể của DDColor.
- Có thể học từ DDColor bằng knowledge distillation; điều đó không thay đổi nguồn gốc kiến trúc.
- Các nhánh V5, V6, V7 và V7.1 là các vòng nghiên cứu/ablation kế tiếp của VietColorNet.

### DDColor

- Là kiến trúc bên thứ ba của Xia và cộng sự, dùng làm **baseline/đối chứng** và teacher.
- Mã nguồn upstream được tích hợp tại commit `2adb63f2656ac41cbdf7b894cddd94121a3faf13` theo Apache License 2.0.
- `colorization/ddcolor_adapter.py` chỉ chuyển đổi giao diện tensor CIE Lab để DDColor dùng chung data loader, runtime và metric của dự án.
- Việc fine-tune DDColor trên dữ liệu Việt Nam không biến DDColor thành kiến trúc do nhóm tự xây dựng.

### Trạng thái lựa chọn model

| Phạm vi | Model/checkpoint | Trạng thái |
|---|---|---|
| Baseline thuần | DDColor gốc (`checkpoints/pretrained/ddcolor_paper_tiny.pth`) | Được ghi trong `outputs/reports/baseline_selection.json`, kiểm tra SHA-256 khi nạp; không tinh chỉnh |
| VietColorNet mặc định trong demo | **VietColorNet · Lịch sử** (tên cũ V15 epoch 3, 384 px) | Tốt nhất trên phim tư liệu chưa thấy; chưa đạt cổng phát hành |
| VietColorNet lựa chọn khác | **VietColorNet · Tươi** (tên cũ V12 epoch 6) | Màu tươi hơn nhưng ám vàng nâu nhiều hơn; giữ làm đối chứng |
| DDColor V6 tinh chỉnh | `checkpoints/runs/ddcolor_historical_still_replay_v6` | Vô hiệu hóa, không còn được nạp; chỉ giữ file để tái lập |
| Mốc rollback VietColorNet | VietColorNet Master V4 | Giữ để tái lập và so sánh |
| V7/V7.1 | DINOv2, palette loss và MoE ablations | Đã đánh giá, không phát hành |
| V11, V13, V14 | Thêm 220 ảnh quân đội; 512 px + cân bằng màu hiếm; đầu ra phân loại màu | Đã đánh giá, không thay V12/V15 (xem mục V10–V15) |

Quyết định checkpoint vận hành là kết quả thực nghiệm, không thay đổi vai trò học thuật: VietColorNet vẫn là mô hình đề xuất và DDColor vẫn là baseline.

## Kiến trúc VietColorNet

VietColorNet kết hợp các thành phần chính sau:

1. **Semantic encoder:** backbone ResNet-34 trích xuất đặc trưng ngữ nghĩa đa mức.
2. **Edge stream:** toán tử Scharr khả vi trích xuất biên để hỗ trợ bảo toàn ranh giới vật thể.
3. **Strip pooling:** gom ngữ cảnh theo phương ngang và dọc.
4. **Edge-gated decoder:** điều tiết đặc trưng giải mã bằng thông tin biên.
5. **CPCA palette attention:** dùng các truy vấn bảng màu học được để đưa ngữ cảnh màu vào decoder.
6. **Decoupled hue–chroma head:** tách hướng màu và độ lớn chroma trước khi dựng dự đoán `a/b`.
7. **Chroma calibration:** V6 bổ sung residual chroma cục bộ, neutral confidence và hệ số chroma theo ảnh.

V7/V7.1 thử thêm DINOv2-S/14 làm semantic bridge, soft palette contrastive loss và palette mixture-of-experts. Các thí nghiệm này có tác động đến phân bố chroma nhưng chưa cải thiện đồng thời độ chính xác màu và chất lượng cảm nhận so với V6 refined.

V8–V13 dùng cùng kiến trúc V7 (`vietcolornet_v8` ánh xạ tới `VietColorNetV7`); khác biệt nằm ở dữ liệu, độ phân giải và hàm loss. `VietColorNetV14` thay đầu ra hồi quy `a/b` bằng đầu ra phân loại 259 ô màu (giải mã annealed mean, T = 0,38) như thí nghiệm, không phát hành.

**Thành phần pretrained (bên thứ ba, không phải mô hình tô màu):** ResNet-34 (torchvision, ImageNet) làm encoder, DINOv2-S/14 (Meta, Apache-2.0) làm semantic bridge cố định, và Segment Anything ViT-B (Meta, Apache-2.0) chỉ dùng ở suy luận để tìm ranh giới vật thể chống lem màu. Toàn bộ phần dự đoán màu (edge stream, CPCA palette attention, decoder, đầu ra màu) là do nhóm thiết kế; không dùng mô hình tô màu bên ngoài làm teacher từ V9 trở đi.

## Dữ liệu và protocol

### ImageNet

- Nguồn ban đầu là Downsampled ImageNet 64×64, không phải ImageNet độ phân giải gốc.
- Train: 1.281.167 ảnh.
- Validation chọn mô hình: 5.000 ảnh, lấy cân bằng 5 ảnh mỗi lớp từ official validation.
- Test cuối: 45.000 ảnh official validation còn lại.
- Validation và test không giao nhau, được phân tầng theo 1.000 lớp.
- Manifest và SHA-256 được dùng để cố định split; test cuối không được dùng để chỉnh mô hình.

### Dữ liệu Việt Nam ban đầu

Protocol V1 chia theo video để tránh frame gần nhau xuất hiện ở nhiều tập:

- Train: 1.235 ảnh từ `NhungNguoiVietLenHuyenThoai` và `VaoNamRaBac`.
- Validation: 225 ảnh từ `MuiCoChay`.
- Development test: 436 ảnh từ `HaNoi12NgayDem`.

Sau khi kết quả của `HaNoi12NgayDem` được dùng để quyết định cải tiến, tập này được xem là development test, không còn là test cuối độc lập.

### Dữ liệu từ V9 trở đi (màu hiện đại + phim tư liệu)

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| Viet-Cultural-VQA, KTVIC, Wikimedia Commons | Train/validation/test màu hiện đại | Split val/test chính thức của VQA/KTVIC; loại trùng chính xác và gần trùng (dHash) |
| Ảnh quân đội Việt Nam (Wikimedia Commons) | Train | 944 ảnh có giấy phép, giữ 667 ảnh chụp thật sau khi bỏ cờ, huy hiệu, bản đồ, giấy tờ, ảnh đen trắng |
| ImageNet chọn lọc (replay) | Train | 36.797 ảnh màu của 32 lớp quân sự và cảnh Việt Nam (xe tăng, pháo, quân phục, trâu, nhà tranh…) |
| ViCoW (4 phim) | Train/validation/test lịch sử | Chia theo phim; ảnh train được hiệu chỉnh nhẹ |

- **V15:** train 62.172 ảnh (VQA 19.247, KTVIC 3.749, Wikimedia 477, quân đội 667, ViCoW 1.235, ImageNet chọn lọc 36.797), lấy mẫu có trọng số; validation 1.025 ảnh; test 1.193 ảnh.
- **ViCoW chia theo phim, không dùng split gốc.** Split gốc của ViCoW (1.327/187/382) là ngẫu nhiên theo khung hình và cả 4 phim đều có ở mọi tập: 75% ảnh test (288/382) có một ảnh train cách không quá 2 khung hình trong cùng phim, nên điểm số bị thổi phồng. Protocol V15 train trên *Những người viết huyền thoại* và *Vào Nam ra Bắc*, validation *Mùi cỏ cháy*, test *Hà Nội 12 ngày đêm*.
- **Hiệu chỉnh ViCoW train** (`scripts/prepare/correct_vicow_colors.py`): khung hình 1280×720 được phóng từ video độ phân giải thấp nên mềm. Kênh L được làm nét (độ nét tăng gấp đôi), chroma được căn lại theo biên bằng guided filter, khử ám màu nhẹ trên vùng gần trung tính và tăng nhẹ độ đậm cho phim bị phai. Phần lớn màu xanh của *Vào Nam ra Bắc* là rừng thật nên chỉnh có giới hạn. Phim validation/test giữ nguyên bản gốc vì là đáp án để đo.
- Màu "chuẩn" của ViCoW là màu phim cũ (độ đậm khoảng 9–10 so với 17–19 của ảnh hiện đại), nên trên phim này chroma ratio thấp hơn không có nghĩa là kém hơn; điểm chọn epoch loại tập validation ViCoW (`selection_exclude_sources`) và chỉ theo dõi nó.

### Wikimedia Commons và dữ liệu lịch sử

- Collector dùng API Wikimedia Commons, lưu metadata, tác giả, URL nguồn, giấy phép và credit trong `attribution.csv`.
- 477/500 ảnh của vòng thu thập đã vượt kiểm tra tự động và duyệt thủ công cho protocol mixed-source V3.
- Protocol nghiên cứu VietColorNet V5–V7 sử dụng 3.919 mẫu train, 825 mẫu validation và 436 mẫu test/development theo artefact đã chốt.
- Các protocol lịch sử mới phải tách theo nguồn/video và kiểm tra trùng hoặc gần trùng trước khi dùng để kết luận.

Không xóa attribution hoặc thông tin giấy phép khi dùng dữ liệu Wikimedia cho nghiên cứu hay công bố.

## Cài đặt

Yêu cầu khuyến nghị:

- Python 3.11.
- GPU NVIDIA có CUDA; cấu hình dự án đã được tối ưu và thử nghiệm trên RTX 3050 Laptop 6 GB.
- Windows PowerShell cho các lệnh cài đặt và chạy trong README.

Từ thư mục dự án:

```powershell
cd KHOA_LUAN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài bản PyTorch/TorchVision hỗ trợ CUDA phù hợp từ trang chính thức của PyTorch trước, sau đó:

```powershell
python -m pip install -r requirements.txt
python doctor.py
```

Các dependency trực tiếp gồm NumPy, OpenCV, Pillow, scikit-image, LPIPS, Hugging Face Hub, PyYAML, SciPy, Requests, Gradio, imageio-ffmpeg và segment-anything. Phiên bản cụ thể được giới hạn trong `requirements.txt`.

### Dữ liệu và trọng số không lưu trên Git

Repository chứa mã nguồn hệ thống nhưng không chứa dataset, môi trường ảo, checkpoint huấn luyện hoặc binary đã build vì các artefact này có dung lượng từ hàng trăm MB đến hàng chục GB. Sau khi clone, cần chuẩn bị dữ liệu bằng các script trong `scripts/prepare/` (mô tả từng nguồn dữ liệu trong `data/sources/README.md`) và đặt checkpoint vào đúng đường dẫn được khai báo trong selection/config tương ứng.

Checkpoint DDColor chính thức có thể tải bằng:

```powershell
python scripts/prepare/download_ddcolor.py --model ddcolor_paper
```

Checkpoint VietColorNet đã fine-tune và trọng số DDColor là artefact của quá trình huấn luyện, không nằm trong Git. `verify_product.py` sẽ báo rõ checkpoint hoặc checksum còn thiếu trước khi chạy giao diện.

Kiểm tra môi trường, DDColor và unit test:

```powershell
python doctor.py --check-ddcolor
python -m unittest discover -s tests -v
```

## Chạy ứng dụng Sắc Việt

Kiểm tra checkpoint sản phẩm và checksum:

```powershell
.\.venv\Scripts\python.exe verify_product.py
```

Nạp thử model để kiểm tra toàn bộ runtime:

```powershell
.\.venv\Scripts\python.exe verify_product.py --load-model
```

Khởi động giao diện:

```powershell
.\.venv\Scripts\python.exe demo.py --open-browser
```

Hoặc bấm đúp `start_demo.bat`. Địa chỉ mặc định là `http://127.0.0.1:7860`.

Các chế độ khác:

```powershell
# Cho phép thiết bị trong cùng LAN truy cập
.\.venv\Scripts\python.exe demo.py --host 0.0.0.0

# Tạo link Gradio tạm thời
.\.venv\Scripts\python.exe demo.py --share

# Bật tính năng video (mặc định tắt)
.\.venv\Scripts\python.exe demo.py --enable-video
```

Ứng dụng:

- Tự nạp DDColor gốc từ `outputs/reports/baseline_selection.json` nếu checkpoint hợp lệ.
- Nạp `checkpoints/models/vietcolornet_historical.pth` (mặc định) và `vietcolornet_vivid.pth`; DDColor gốc được nạp theo `outputs/reports/baseline_selection.json`. Cả ba có trong menu chọn mô hình.
- Chống lem màu bằng SAM (`colorization/segmentation.py`): mỗi ảnh mất khoảng 4–6 giây trên RTX 3050 6 GB; tắt bằng `--no-sam`.
- Chọn VietColorNet làm model mặc định khi checkpoint này nạp thành công; nếu không sẽ dùng model khả dụng đầu tiên.
- Hỗ trợ tô màu ảnh, so sánh A/B DDColor–VietColorNet, lưu PNG vào `outputs/demo_outputs/` và giữ độ phân giải ảnh gốc.
- Có giới hạn 40 megapixel cho ảnh đầu vào.
- Ba chế độ xử lý (`colorization/enhance.py`): **Hiện đại** (mặc định: phục hồi ảnh mờ/nhiễu → tô màu → chống lem → khử ám màu và tăng độ tươi), **Rực rỡ** (như Hiện đại, màu đậm hơn), **Nguyên bản** (chỉ chống lem). Không truyền `mode` thì `ColorizationEngine` trả về màu thô của mô hình, dùng cho đánh giá.
- Khi bật video, dùng optical flow để làm dịu biến động chroma và reset khi phát hiện chuyển cảnh.

Có thể chỉ định checkpoint VietColorNet khác:

```powershell
.\.venv\Scripts\python.exe demo.py --checkpoint checkpoints\models\vietcolornet_historical.pth
```

## Huấn luyện và đánh giá

### Chuẩn bị ImageNet

```powershell
python scripts/prepare/prepare_data.py
```

Script từ chối ghi đè manifest đã tồn tại. Khi cần checkpoint DDColor chính thức:

```powershell
python scripts/prepare/download_ddcolor.py --model ddcolor_paper
python scripts/prepare/register_pretrained.py --output checkpoints/runs/ddcolor_official
```

Không dùng `ddcolor_artistic` cho baseline ImageNet-only vì model đó được huấn luyện thêm trên dữ liệu riêng.

### Smoke test và huấn luyện

```powershell
python train.py `
  --config configs/ddcolor_smoke.json `
  --output checkpoints/runs/smoke_ddcolor `
  --train-limit 16 `
  --val-limit 8
```

Run có `--train-limit` hoặc `--val-limit` chỉ dùng chẩn đoán và không được tham gia chọn mô hình.

Ví dụ huấn luyện DDColor và các baseline:

```powershell
python train.py --config configs/ddcolor_tiny_budgeted.json --output checkpoints/runs/ddcolor_tiny_budgeted
python train.py --config configs/resnet18.json --output checkpoints/runs/resnet18
python train.py --config configs/resnet34.json --output checkpoints/runs/resnet34
```

Tiếp tục run bị gián đoạn bằng đúng config và thư mục cũ:

```powershell
python train.py `
  --config configs/ddcolor_tiny.json `
  --output checkpoints/runs/ddcolor_tiny `
  --resume checkpoints/runs/ddcolor_tiny/latest.pth
```

### Domain adaptation Việt Nam

```powershell
python fine_tune_vietnam.py `
  --config configs/ddcolor_vietnam_adaptation_v2.json `
  --output checkpoints/runs/ddcolor_vietnam_adaptation_v2
```

Pipeline hỗ trợ protocol riêng, checkpoint nguồn, teacher checkpoint, distillation, dữ liệu Wikimedia, dữ liệu lịch sử và resume. Mỗi run phải giữ nguyên config, manifest và seed để tái lập.

### Huấn luyện V15 (ảnh hiện đại + phim tư liệu)

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

So sánh checkpoint trên ảnh hiện đại và phim test lịch sử (ΔE, sai số tại biên, độ lệch màu theo vùng):

```powershell
python scripts/evaluate/compare_checkpoints.py --set modern-test --set vicow-test `
  vivid=checkpoints/models/vietcolornet_vivid.pth `
  historical=checkpoints/models/vietcolornet_historical.pth
```

Khi train GPU 6 GB ở 384 px, đóng demo và các ứng dụng dùng GPU khác. Mô hình “Lịch sử” là epoch 3 của run này, lưu tại `checkpoints/models/vietcolornet_historical.pth`; mô hình “Tươi” (bản khởi tạo, không dùng phim tư liệu) là `checkpoints/models/vietcolornet_vivid.pth`.

### Chọn và đánh giá mô hình

Chỉ đưa các run hoàn chỉnh vào bước chọn:

```powershell
python select_model.py checkpoints/runs/ddcolor_tiny checkpoints/runs/resnet18 checkpoints/runs/resnet34
```

Đánh giá checkpoint đã chọn:

```powershell
python evaluate.py `
  --selection outputs/reports/model_selection.json `
  --output outputs/reports/final_evaluation
```

Hệ thống xuất JSON/CSV, thống kê từng ảnh, ảnh so sánh và các metric PSNR, SSIM, Delta E 2000, AB-MAE, chroma và LPIPS. Khi so sánh model, phải dùng cùng protocol, cùng thứ tự ảnh và cùng code metric.

## Kết quả nghiên cứu đã kiểm chứng

Các số dưới đây lấy từ audit trực tiếp trên cùng 100 ảnh lịch sử, có LPIPS. Mũi tên thể hiện hướng tốt hơn.

| Model | PSNR ↑ | SSIM ↑ | Delta E ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | LPIPS ↓ | Chroma ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| VietColorNet Master V4 | 26,4872 | 0,96523 | 8,9553 | 6,6245 | 7,2257 | 0,13543 | 0,7752 |
| VietColorNet V5-E | 26,9891 | 0,96441 | 8,6027 | 6,4369 | 6,7911 | 0,12539 | 0,8669 |
| **VietColorNet V6 refined** | **27,2488** | 0,96413 | **8,0965** | **6,0719** | **6,6808** | **0,11873** | 0,8868 |
| VietColorNet V7.1-A | 27,0112 | 0,96292 | 8,4898 | 6,3129 | 6,9918 | 0,12307 | **0,9277** |
| VietColorNet V7.1-B | 27,1716 | 0,96249 | 8,3044 | 6,2082 | 6,7910 | 0,12260 | 0,8649 |
| VietColorNet V7.1-C | 27,0446 | 0,96314 | 8,4422 | 6,2819 | 6,9641 | 0,12229 | 0,9142 |

V6 refined cải thiện rõ Delta E, AB-MAE, lỗi màu ở biên và LPIPS so với V4, nhưng không đạt ba cổng phân bố chroma:

- Chroma ratio phải nằm trong 0,90–1,10: thực tế `0,8868`.
- Số ảnh dưới 80% chroma mục tiêu phải dưới 30/100: thực tế `34/100`.
- Số ảnh trên 125% chroma mục tiêu tối đa 15/100: thực tế `24/100`.

V7 không vượt V6 refined. V7.1 xác nhận DINOv2 bridge có thể thay đổi phân bố chroma, nhưng các learned checkpoint đều làm xấu ít nhất một nhóm metric chất lượng hoặc vẫn không đạt cổng phân bố. Vì vậy V7/V7.1 không được phát hành.

### VietColorNet V8 (ImageNet-100K replay + dữ liệu Việt Nam)

V8 dùng lại kiến trúc V7 (`vietcolornet_v8` ánh xạ tới `VietColorNetV7`); điểm mới nằm ở công thức huấn luyện: khởi tạo từ V6 refined, trộn 100.000 ảnh ImageNet replay với nguồn Việt Nam và hai bộ craw (Viet-Cultural-VQA, KTVIC), 20 epoch × 8.000 mẫu, BF16. Config: `configs/vietcolornet_v8_imagenet100k_vietnam.json`.

Kết quả trên cùng 100 ảnh lịch sử:

| Model | PSNR ↑ | SSIM ↑ | Delta E ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | LPIPS ↓ | <80% chroma | >125% chroma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VietColorNet V6 refined | 27,2488 | 0,96413 | 8,0965 | 6,0719 | 6,6808 | 0,11873 | 34 | 24 |
| **V8 epoch 1** | **27,4224** | **0,96479** | **8,0367** | **6,0255** | **6,5969** | **0,11794** | 37 | 28 |
| V8 epoch 20 (`latest.pth`) | 26,8392 | 0,96360 | 8,2489 | 6,2505 | 6,9590 | 0,12037 | 38 | 28 |

- Trên validation, không epoch nào vượt điểm source-balanced của baseline V6 (0,9904; tốt nhất là epoch 1 với 1,0618), nên `best.pth` của run này chính là V6 refined.
- Epoch 1 nhỉnh hơn V6 refined trên mọi metric chất lượng trong audit, nhưng mức chênh nhỏ, và tập 100 ảnh này đã được dùng trong các quyết định trước nên chỉ là development set.
- Từ epoch 1 đến epoch 20, mô hình xấu dần; video lịch sử bị tô quá màu (chroma ratio ≈ 1,4–1,8 so với 1,31 của baseline).
- V8 vẫn không đạt cổng phân bố chroma (37/100 ảnh dưới 80%, 28/100 ảnh trên 125%).
- V8 epoch 1 được giữ làm phương án dự phòng sau V9.

### VietColorNet V9 (màu hiện đại)

Đánh giá V8 cho thấy nguyên nhân màu "cổ", ám olive/sepia nằm ở dữ liệu: các nguồn phim/ảnh lịch sử mang ám màu (vietnam_video a = −4,3; historical_video a = +8,8, b = −10,2; historical still chroma chỉ 9), và checkpoint được chọn bằng validation gồm toàn các nguồn nhợt màu này (chroma 5–13). V9 thay đổi dữ liệu, không đổi kiến trúc (vẫn V7):

- Protocol `vietnam_modern_v9` (`scripts/prepare/prepare_v9_modern_protocol.py`): train 23.473 ảnh màu hiện đại (Viet-Cultural-VQA, KTVIC, Wikimedia Commons); validation 800 ảnh và test 757 ảnh từ split val/test chính thức của VQA và KTVIC; loại trùng chính xác và gần trùng (dHash) giữa train và tập đánh giá.
- Augmentation suy giảm chất lượng trên kênh L đầu vào (phai tương phản, mờ, hạ phân giải, nhiễu, JPEG), giữ ab đích sạch, để mô hình quen với ảnh scan cũ (`input_degradation` trong `configs/vietcolornet_v9_modern.json`).
- Khởi tạo từ V8 epoch 1; dừng sớm sau 4 epoch. Chọn epoch 3 vì chroma ratio validation tốt nhất (0,927 so với 0,852 của epoch 1) khi điểm chọn tổng hợp chỉ chênh 0,003.

Tập test hiện đại 757 ảnh (màu thô của mô hình, ảnh resize tối đa 512 px):

| Model | Delta E 2000 ↓ | Chroma ratio (trung vị) | Ảnh < 80% chroma |
|---|---:|---:|---:|
| VietColorNet V6 refined | 12,661 | 0,605 | 520 |
| VietColorNet V8 epoch 1 | 12,564 | 0,641 | 493 |
| VietColorNet V9 epoch 1 | 13,114 | 0,856 | 343 |
| **VietColorNet V9 epoch 3** | 13,095 | **0,945** | **286** |

V9 khắc phục hiện tượng tô nhạt (chroma ratio từ 0,64 lên 0,95), đổi lại Delta E tăng khoảng 0,5 vì metric này phạt việc dự đoán màu đậm khi có nhiều đáp án hợp lý. Trên ảnh lịch sử thực tế, trời, cây, da người và vùng trắng có màu tự nhiên hơn; quân phục và xe quân sự vẫn thường bị tô nâu thay vì xanh olive vì dữ liệu hiện đại gần như không có các đối tượng này.

Hậu xử lý chống lem (`refine_ab`): màu dự đoán ở 256 px bị nhòe thành một dải quanh mọi biên khi phóng lên độ phân giải gốc. Pixel trong dải đó bị coi là không tin cậy; mỗi superpixel SLIC lấy màu từ pixel tin cậy của chính nó, superpixel sát biên chỉ mượn màu từ superpixel lân cận có độ sáng tương tự. Trên ảnh test hiện đại, bước này giảm Delta E (12,509 → 12,418). Giới hạn: hai vùng có độ sáng gần như nhau thì ảnh xám không đủ thông tin để tách.

### VietColorNet V10 → V15 (màu tự nhiên, bớt ám vàng nâu)

Các phiên bản sau V9 đều khởi tạo từ phiên bản trước, giữ kiến trúc V7 (trừ V14) và cùng validation/test hiện đại của V9 để so sánh trực tiếp. Mọi số liệu "test" dưới đây là số đo sau khi đã chọn epoch bằng validation.

| Bản | Thay đổi | Kết quả chính | Quyết định |
|---|---|---|---|
| V10 | Thêm 36.797 ảnh ImageNet màu của 32 lớp quân sự/cảnh Việt Nam (replay) | Trên ImageNet validation: ΔE ảnh quân sự 12,64 → 10,48, cảnh Việt Nam 13,10 → 10,35; độ tươi ảnh quân sự 1,74 → 1,07 (V9 tô quá đậm) | Dùng |
| V11 | Thêm 220 ảnh quân đội Việt Nam | ΔE trên 165 ảnh quân đội chưa thấy 11,14 → 10,60 (−5%) nhưng điểm validation hiện đại kém hơn (0,870 so với 0,849); màu quân phục không đổi | Không dùng |
| V12 | 384 px (thay 256) + 667 ảnh quân đội; chọn epoch 6 | Sai số màu tại biên 12,72 → 12,51; sửa lỗi tóc bạc nhuốm màu trời; epoch 6 tươi hơn epoch 3 (chroma 0,909 so với 0,866) | Dùng |
| V13 | 512 px + cân bằng màu hiếm (Zhang et al., 2016) trong loss hồi quy | Không cải thiện: ΔE 12,89 so với 12,65 của V12, độ lệch màu gần như không đổi | Không dùng |
| V14 | Đầu ra phân loại 259 ô màu thay cho hồi quy | Cờ đỏ ra đúng màu, bớt ám vàng vùng trung tính, nhưng quân phục thành đỏ cam và ΔE kém (12,92); tăng nhiệt độ giải mã không sửa được | Không dùng |
| **V15** | V12 + 1.235 khung hình ViCoW đã hiệu chỉnh (chia theo phim), epoch 3 | Xem bảng dưới | **Mặc định** |

**Nguyên nhân ám vàng nâu.** So màu mô hình tô với màu thật trên 267 ảnh validation hiện đại cho V12: cây xanh lá (thật a = −14,0; b = +14,8) bị tô (a = +2,5; b = +16,6), vùng trung tính (thật b = +0,2) bị ám b = +8,4, trời xanh (thật b = −19,7) bị tô b = +12,2. Mô hình hồi quy dự đoán màu trung bình của các khả năng, và trung bình của xanh lá với đỏ là vàng nâu. Cân bằng màu hiếm (V13) chỉ dịch được rất ít; đầu ra phân loại (V14) làm mô hình chọn màu dứt khoát nhưng chọn sai màu quân phục. Khoảng cách này chưa được đóng, xem mục Giới hạn.

Kết quả V15 trên tập test (ảnh resize tối đa 640 px, màu thô của mô hình, mỗi hai ảnh lấy một; `scripts/evaluate/compare_checkpoints.py`, báo cáo `outputs/reports/v15_comparison.json`):

| Tập | Model | ΔE 2000 ↓ | Edge AB-MAE ↓ | Chroma ratio | Cây xanh (a, b) tô → thật | Vùng trung tính b tô (thật) |
|---|---|---:|---:|---:|---|---|
| Ảnh hiện đại (379) | V12 epoch 6 | 12,511 | 12,717 | 0,876 | (+2,2; +17,4) → (−14,0; +19,0) | +8,2 (+0,3) |
| | **V15 epoch 3** | **12,113** | **12,410** | 0,723 | (+1,4; +14,1) → (−14,0; +19,0) | **+6,4** (+0,3) |
| Phim *Hà Nội 12 ngày đêm* (218) | V12 epoch 6 | 10,961 | 10,130 | 1,644 | (+1,3; +19,6) → (−9,4; +9,8) | +9,7 (−0,1) |
| | **V15 epoch 3** | **9,571** | **7,669** | 1,178 | (−1,2; +13,6) → (−9,4; +9,8) | **+5,8** (−0,1) |

Trên một lần chạy (seed 2026), V15 giảm sai số màu tại biên trên phim lịch sử 24% so với V12 (nơi màu hay lem; số liệu độ ổn định theo seed ở mục Đánh giá thống kê), giảm ΔE trên cả hai tập và bớt ám vàng, đổi lại tô nhạt hơn V12 trên ảnh hiện đại (0,72 so với 0,88); bước "tăng độ tươi" của chế độ Hiện đại bù lại một phần. Chọn epoch 3 dựa trên validation (loss phim lịch sử thấp nhất, điểm hiện đại ngang epoch 1); số test chỉ dùng để báo cáo. Train dừng sớm sau epoch 3 vì điểm validation hiện đại không vượt được mốc của V12.

**Chống lem màu.** Pipeline suy luận (`colorization/enhance.py`, `colorization/segmentation.py`): (1) phục hồi kênh L (khử nhiễu theo mức đo được, kéo dải sáng giữ chi tiết vùng tối, làm nét); (2) dự đoán ab ở 256/384 px; (3) `refine_ab`: pixel gần biên được coi là không tin cậy vì màu bị nhòe khi phóng ảnh, mỗi superpixel lấy màu từ lõi của chính nó và chỉ mượn màu từ superpixel lân cận cùng độ sáng; (4) Segment Anything tách vật thể để màu không được mượn qua ranh giới vật thể, kể cả khi hai vùng có độ sáng giống nhau (tóc bạc cạnh trời); (5) `modernize_ab`: khử ám màu chỉ trên vùng gần trung tính, không đẩy pixel xám sang màu đối diện, và tăng độ tươi theo kiểu vibrance. Trên 134 ảnh validation hiện đại, `refine_ab` giảm ΔE (13,34 → 13,17) và sai số tại biên (12,85 → 12,69); SAM giữ độ tươi tốt hơn (0,955 so với 0,909) nhưng không giảm thêm các số đo này vì ảnh hiện đại hiếm khi có hai vật cùng độ sáng sát nhau. Lợi ích của SAM thấy rõ ở ảnh lịch sử (mặt người không nhuốm màu tường, tóc bạc không nhuốm màu trời) và chưa có số đo khách quan trên bộ ảnh lịch sử có đáp án màu.

### Đánh giá thống kê và độ ổn định

Theo yêu cầu kiểm tra độ ổn định, mọi so sánh dưới đây dùng toàn bộ tập test (757 ảnh hiện đại, 436 khung phim *Hà Nội 12 ngày đêm*), ảnh resize tối đa 640 px, đầu vào là ảnh chỉ còn độ sáng. Mỗi độ đo có khoảng tin cậy 95% bootstrap theo khối (các khung liền nhau của một phim được lấy mẫu theo khối 12 khung), chênh lệch cặp có khoảng tin cậy và kiểm định Wilcoxon. Script: `scripts/evaluate/statistical_comparison.py`; báo cáo trong `outputs/reports/`. Baseline là **DDColor gốc**, không phải DDColor V6 đã tinh chỉnh.

**Độ lệch theo seed.** Giai đoạn tinh chỉnh V12 → "Lịch sử" được lặp lại với seed 1 và 2 (`scripts/evaluate/seed_runs.py`; cả hai dừng sớm sau epoch 3 như bản phát hành, seed gốc là 2026). Báo cáo: `outputs/reports/seed_variance.{md,json}`.

| Tập | Độ đo | Seed 2026 | Seed 1 | Seed 2 | Trung bình ± độ lệch chuẩn | Tươi | DDColor gốc |
|---|---|---|---|---|---|---|---|
| Phim (436) | ΔE2000 ↓ | 9,56 | 10,64 | 10,42 | 10,21 ± 0,57 | 11,03 | 11,43 |
| | Sai số biên ↓ | 7,74 | 8,89 | 8,31 | 8,31 ± 0,58 | 10,22 | 12,08 |
| Ảnh hiện đại (757) | ΔE2000 ↓ | 12,19 | 12,72 | 12,25 | 12,39 ± 0,29 | 12,61 | 11,77 |
| | Sai số biên ↓ | 12,42 | 12,67 | 12,41 | 12,50 ± 0,15 | 12,75 | 12,91 |

- Trên phim, cả ba seed thắng DDColor gốc có ý nghĩa thống kê (ΔE thấp hơn 0,80 đến 1,87; sai số biên thấp hơn 3,2 đến 4,4; p < 10⁻⁶). Seed phát hành (2026) là seed tốt nhất trong ba, nên nên báo cáo trung bình 10,21 ± 0,57 thay vì 9,56.
- So với "Tươi", lợi thế về ΔE trên phim nhỏ (seed 1 chỉ hơn 0,39, khoảng tin cậy chạm 0), còn lợi thế về sai số tại biên ổn định ở cả ba seed.
- Trên ảnh hiện đại, DDColor gốc có ΔE thấp hơn ở cả ba seed (0,42 đến 0,95); về sai số biên hai seed tốt hơn nhẹ và seed 1 hòa (p = 0,06).
- Độ lệch seed chỉ phản ánh giai đoạn tinh chỉnh; checkpoint "Tươi" xuất phát dùng chung nên chưa được lặp lại. Ba seed là mẫu nhỏ.

**Ảnh hưởng của pipeline suy luận.** Cùng ba mô hình và cùng tập test, so ba cấu hình áp dụng giống hệt cho mọi mô hình (`pipeline_raw`, `pipeline_refined`, `pipeline_refined_sam`; ΔE2000, chế độ natural):

| Tập | Cấu hình | Lịch sử | Tươi | DDColor gốc |
|---|---|---|---|---|
| Phim | thô | 9,56 | 11,03 | 11,43 |
| | + hậu xử lý | 9,44 | 10,93 | 11,07 |
| | + hậu xử lý + SAM | 9,51 | 11,03 | 11,18 |
| Ảnh hiện đại | thô | 12,19 | 12,61 | 11,77 |
| | + hậu xử lý | 12,16 | 12,57 | 11,60 |
| | + hậu xử lý + SAM | 12,15 | 12,58 | 11,48 |

Hậu xử lý giúp cả ba mô hình một chút nhưng không đổi thứ hạng, nên khác biệt giữa các mô hình đến từ mạng. **SAM không cải thiện ΔE hay sai số tại biên** (sai số biên của "Lịch sử" trên phim nhích từ 7,72 lên 7,83). Các độ đo điểm ảnh không chứng minh được lợi ích của SAM; nếu cần khẳng định, phải đo thêm bằng độ đo lem màu riêng hoặc khảo sát người dùng.

**Lưu ý phương pháp.** Chỉ số chroma ratio không đáng tin vì nhiều ảnh gần xám, nên không dùng. Ảnh lịch sử không có đáp án màu độc lập; "màu thật" của phim là màu phim cũ. DDColor gốc không được huấn luyện trên dữ liệu Việt Nam, còn VietColorNet dùng một tập con ImageNet nhỏ ở 64 px, nên đây không phải so sánh cùng ngân sách huấn luyện.

## Thay checkpoint vận hành

Hệ thống chạy dưới dạng ứng dụng web: giao diện tùy biến FastAPI bằng `python server.py --open-browser` (cổng 8000) hoặc giao diện Gradio bằng `demo.py`; không còn bản cài đặt Windows. Khi thay checkpoint baseline DDColor:

1. Dừng `demo.py`.
2. Sao chép checkpoint mới bằng tên tạm.
3. Kiểm tra file và SHA-256.
4. Đổi tên thành `best.pth`.
5. Cập nhật đồng thời đường dẫn và checksum trong `baseline_selection.json`.
6. Giữ checkpoint cũ để rollback.

## Cấu trúc dự án

Mọi lệnh trong README chạy từ thư mục gốc `KHOA_LUAN/`.

```text
KHOA_LUAN/
├── README.md
├── requirements.txt
├── start_demo.bat                 # bấm đúp để mở ứng dụng web
│
├── demo.py                        # ỨNG DỤNG WEB (Gradio)
├── train.py                       # HUẤN LUYỆN: baseline ImageNet
├── fine_tune_vietnam.py           # HUẤN LUYỆN: VietColorNet trên dữ liệu Việt Nam
├── evaluate.py                    # ĐÁNH GIÁ: ImageNet / Việt Nam
├── evaluate_image_failures.py     # ĐÁNH GIÁ: audit 100 ảnh lịch sử, worst cases
├── select_model.py                # tiện ích: chọn checkpoint từ các run hoàn chỉnh
├── verify_product.py              # tiện ích: kiểm tra selection, checksum, runtime
├── doctor.py                      # tiện ích: kiểm tra môi trường và dữ liệu ImageNet
│
├── colorization/                  # LÕI: mô hình, dữ liệu, suy luận, chống lem, SAM, metric
├── configs/                       # cấu hình từng phiên bản (DDColor, VietColorNet V5 … V14)
├── scripts/
│   ├── prepare/                   # thu thập dữ liệu, dựng protocol, tải trọng số
│   ├── evaluate/                  # đánh giá theo từng vòng nghiên cứu
│   └── legacy/                    # pipeline V5/V7/V7.1, resume V8, sinh báo cáo docx
├── tests/                         # unit tests
├── third_party/DDColor/           # mã nguồn DDColor upstream
├── examples/                      # ảnh mẫu cho giao diện
├── docs/                          # báo cáo .docx, tài liệu .md, giấy phép bên thứ ba
│
├── data/                          # TOÀN BỘ DỮ LIỆU (không lưu trên Git)
│   ├── sources/                   # dữ liệu gốc, mỗi nguồn một thư mục (xem data/sources/README.md)
│   │   ├── vicow/                 #   4 phim tư liệu lịch sử (ViCoW)
│   │   ├── historical_vietnam_images/   # ảnh lịch sử cho audit 100 ảnh
│   │   ├── viet_cultural_vqa/  ktvic/   # ảnh hiện đại Việt Nam
│   │   ├── wikimedia_vietnam/  wikimedia_vietnam_military/
│   │   └── imagenet64/            #   ImageNet 64×64 gốc
│   ├── processed/                 # ImageNet đã xử lý (shards, nhãn, manifest)
│   └── protocols/                 # chia train/val/test V1 … V12 (CSV + manifest SHA-256)
│
├── checkpoints/                   # TOÀN BỘ MÔ HÌNH (không lưu trên Git)
│   ├── pretrained/                # SAM ViT-B, DDColor gốc
│   └── runs/                      # các bản đã train: V6 refined … V14, DDColor V6
│
├── outputs/                       # KẾT QUẢ SINH RA
│   ├── reports/                   # số liệu, audit, selection (baseline_selection.json)
│   └── demo_outputs/              # ảnh tạo từ ứng dụng web
│
└── .venv/                         # môi trường Python
```

`.venv/`, `data/`, `checkpoints/`, `outputs/` và mọi file `*.pth`/`*.pt` được bỏ qua để tránh đưa dữ liệu hoặc artefact lớn vào Git ngoài ý muốn. Repository chỉ giữ `outputs/reports/baseline_selection.json` và `data/sources/README.md`; các report khác phải được thêm có chủ đích nếu cần công bố.

## Giới hạn

- Tô màu là bài toán một đầu vào có nhiều đáp án hợp lý; PSNR/SSIM không đủ để kết luận chất lượng. Cần đọc cùng Delta E, AB-MAE, LPIPS, chroma và ảnh mẫu.
- ImageNet dùng trong protocol là bản 64×64 rồi nội suy lên kích thước huấn luyện; nội suy không tạo thêm chi tiết thật.
- Checkpoint DDColor chính thức đã học ImageNet. Fine-tune từ checkpoint này là transfer learning, không phải huấn luyện từ đầu.
- Tập dữ liệu đã tham gia lựa chọn kiến trúc/hyperparameter chỉ là development set; kết luận cuối cần nguồn độc lập chưa từng dùng trong thiết kế.
- **Màu vẫn lệch vàng nâu:** cây cỏ bị tô gần như không xanh (a ≈ 0 thay vì −14), trời xanh bị tô nhạt, quân phục và cờ đỏ đôi khi sai màu vì đỏ, xanh đậm và nâu có cùng độ sáng trên ảnh xám. Các thử nghiệm V13 và V14 chưa đóng được khoảng cách này; cần thêm dữ liệu có nhãn ngữ nghĩa hoặc điều kiện hóa màu bằng ngữ cảnh.
- Màu "chuẩn" của ViCoW là màu phim cũ; điểm số trên phim này đo độ giống phim, không đo độ đẹp hay tự nhiên. Chưa có tập ảnh lịch sử có đáp án màu hiện đại độc lập; `Hà Nội 12 ngày đêm` là development test từ V1 đến V8 nhưng chưa từng được dùng để train hay chọn mô hình từ V9.
- Ảnh ImageNet chỉ 64×64 rồi phóng lên 384 px nên là nguồn mờ nhất trong dữ liệu train; augmentation suy giảm chất lượng giúp mô hình quen với ảnh mờ nhưng không tạo thêm chi tiết.
- Ảnh hư hỏng nặng, mất mảng hoặc trầy xước lớn cần thêm bước restoration/inpainting.
- Video dài trên GPU laptop cần nhiều thời gian và bộ nhớ; tính năng video trong demo mặc định bị tắt.
- Kết quả định lượng chỉ có ý nghĩa khi protocol, checkpoint, checksum và mã đánh giá được cố định.

## Phần mềm bên thứ ba

DDColor:

- Repository: <https://github.com/piddnad/DDColor>
- Commit tích hợp: `2adb63f2656ac41cbdf7b894cddd94121a3faf13`
- Giấy phép: Apache License 2.0; file `LICENSE` gốc được giữ trong `third_party/DDColor/`.
- Bài báo: Xia, Z., Shen, X., Huang, J., Chen, J., & Ji, R. (2023), “DDColor: Towards Photo-Realistic and Consistent Image Colorization with Dual Decoders”, ICCV 2023.

Segment Anything (SAM ViT-B):

- Repository: <https://github.com/facebookresearch/segment-anything>; giấy phép Apache License 2.0.
- Chỉ dùng để tách ranh giới vật thể khi suy luận; không đóng góp vào dự đoán màu. Tải trọng số bằng `scripts/prepare/download_sam.py` (kiểm tra SHA-256).
- Bài báo: Kirillov, A. et al. (2023), “Segment Anything”, ICCV 2023.

DINOv2 (Oquab et al., 2023, Apache-2.0) và ResNet-34 (He et al., 2016; trọng số torchvision) được dùng làm backbone trích đặc trưng cho VietColorNet.

Các dependency khác giữ nguyên giấy phép tương ứng của từng dự án. Thư mục demo vendored của DDColor thuộc upstream và không được ứng dụng Sắc Việt gọi trực tiếp.
