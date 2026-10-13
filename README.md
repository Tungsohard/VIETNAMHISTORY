# Sắc Việt — Phục chế màu ảnh tư liệu lịch sử Việt Nam

Sắc Việt là hệ thống nghiên cứu và triển khai tô màu ảnh, video tư liệu lịch sử Việt Nam bằng học sâu. Dự án phát triển **VietColorNet** như kiến trúc đề xuất của khóa luận, đồng thời dùng **DDColor** làm baseline/đối chứng và teacher trong các thí nghiệm chưng cất tri thức.

> Trạng thái hiện tại: checkpoint vận hành được đóng băng trong `ImageNet/reports/production_selection.json` là **DDColor V6**. VietColorNet V6 refined là ứng viên nghiên cứu tốt nhất của nhánh VietColorNet hiện có và được ưu tiên khi chạy local nếu checkpoint tồn tại, nhưng chưa đạt toàn bộ cổng phát hành. V7 và V7.1 là các ablation đã hoàn tất, không thay thế V6 refined.

## Mục lục

- [Mục tiêu và phạm vi](#mục-tiêu-và-phạm-vi)
- [Vai trò các mô hình](#vai-trò-các-mô-hình)
- [Kiến trúc VietColorNet](#kiến-trúc-vietcolornet)
- [Dữ liệu và protocol](#dữ-liệu-và-protocol)
- [Cài đặt](#cài-đặt)
- [Chạy ứng dụng Sắc Việt](#chạy-ứng-dụng-sắc-việt)
- [Huấn luyện và đánh giá](#huấn-luyện-và-đánh-giá)
- [Kết quả nghiên cứu đã kiểm chứng](#kết-quả-nghiên-cứu-đã-kiểm-chứng)
- [Đóng gói Windows](#đóng-gói-windows)
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
- Đóng gói ứng dụng Windows bằng PyInstaller/Inno Setup.

Hệ thống dùng không gian màu CIE Lab: mô hình nhận kênh độ sáng `L` và dự đoán hai kênh màu `a/b`. Ảnh kết quả được ghép lại và chuyển về RGB, đồng thời giữ kích thước đầu vào khi suy luận.

## Vai trò các mô hình

### VietColorNet

- Là **mô hình đề xuất do nhóm tự thiết kế và triển khai** trong `ImageNet/colorization/model.py`.
- Không phải là một biến thể của DDColor.
- Có thể học từ DDColor bằng knowledge distillation; điều đó không thay đổi nguồn gốc kiến trúc.
- Các nhánh V5, V6, V7 và V7.1 là các vòng nghiên cứu/ablation kế tiếp của VietColorNet.

### DDColor

- Là kiến trúc bên thứ ba của Xia và cộng sự, dùng làm **baseline/đối chứng** và teacher.
- Mã nguồn upstream được tích hợp tại commit `2adb63f2656ac41cbdf7b894cddd94121a3faf13` theo Apache License 2.0.
- `ImageNet/colorization/ddcolor_adapter.py` chỉ chuyển đổi giao diện tensor CIE Lab để DDColor dùng chung data loader, runtime và metric của dự án.
- Việc fine-tune DDColor trên dữ liệu Việt Nam không biến DDColor thành kiến trúc do nhóm tự xây dựng.

### Trạng thái lựa chọn model

| Phạm vi | Model/checkpoint | Trạng thái |
|---|---|---|
| Vận hành đóng băng | DDColor V6, epoch 7 | Được ghi trong `production_selection.json` và kiểm tra SHA-256 khi nạp |
| VietColorNet local testing | VietColorNet V6 refined, epoch 4 | Ứng viên nghiên cứu tốt nhất; chưa đạt cổng phát hành |
| Mốc rollback VietColorNet | VietColorNet Master V4 | Giữ để tái lập và so sánh |
| V7/V7.1 | DINOv2, palette loss và MoE ablations | Đã đánh giá, không phát hành |

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
- Windows PowerShell cho các script build và launcher hiện có.

Từ thư mục dự án:

```powershell
cd ImageNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài bản PyTorch/TorchVision hỗ trợ CUDA phù hợp từ trang chính thức của PyTorch trước, sau đó:

```powershell
python -m pip install -r requirements.txt
python doctor.py
```

Các dependency trực tiếp gồm NumPy, OpenCV, Pillow, scikit-image, LPIPS, Hugging Face Hub, PyYAML, SciPy, Requests, Gradio, imageio-ffmpeg và pywebview. Phiên bản cụ thể được giới hạn trong `ImageNet/requirements.txt`.

### Dữ liệu và trọng số không lưu trên Git

Repository chứa mã nguồn hệ thống nhưng không chứa dataset, môi trường ảo, checkpoint huấn luyện hoặc binary đã build vì các artefact này có dung lượng từ hàng trăm MB đến hàng chục GB. Sau khi clone, cần chuẩn bị dữ liệu bằng các script `prepare_*.py` và đặt checkpoint vào đúng đường dẫn được khai báo trong selection/config tương ứng.

Checkpoint DDColor chính thức có thể tải bằng:

```powershell
python download_ddcolor.py --model ddcolor_paper
```

Checkpoint DDColor V6 và VietColorNet đã fine-tune là artefact của quá trình huấn luyện, không nằm trong Git. `verify_product.py` sẽ báo rõ checkpoint hoặc checksum còn thiếu trước khi chạy giao diện.

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

- Tự nạp DDColor từ `ImageNet/reports/production_selection.json` nếu checkpoint hợp lệ.
- Tìm VietColorNet theo thứ tự ưu tiên V6 refined → Master V4 → V3 → các checkpoint cũ hơn.
- Chọn VietColorNet làm model mặc định khi checkpoint này nạp thành công; nếu không sẽ dùng model khả dụng đầu tiên.
- Hỗ trợ tô màu ảnh, so sánh A/B DDColor–VietColorNet, lưu PNG vào `ImageNet/demo_outputs/` và giữ độ phân giải ảnh gốc.
- Có giới hạn 40 megapixel cho ảnh đầu vào.
- Khi bật video, dùng optical flow để làm dịu biến động chroma và reset khi phát hiện chuyển cảnh.

Có thể chỉ định checkpoint VietColorNet khác:

```powershell
.\.venv\Scripts\python.exe demo.py --checkpoint runs\vietcolornet_v6_refined\best.pth
```

## Huấn luyện và đánh giá

### Chuẩn bị ImageNet

```powershell
python prepare_data.py
```

Script từ chối ghi đè manifest đã tồn tại. Khi cần checkpoint DDColor chính thức:

```powershell
python download_ddcolor.py --model ddcolor_paper
python register_pretrained.py --output runs/ddcolor_official
```

Không dùng `ddcolor_artistic` cho baseline ImageNet-only vì model đó được huấn luyện thêm trên dữ liệu riêng.

### Smoke test và huấn luyện

```powershell
python train.py `
  --config configs/ddcolor_smoke.json `
  --output runs/smoke_ddcolor `
  --train-limit 16 `
  --val-limit 8
```

Run có `--train-limit` hoặc `--val-limit` chỉ dùng chẩn đoán và không được tham gia chọn mô hình.

Ví dụ huấn luyện DDColor và các baseline:

```powershell
python train.py --config configs/ddcolor_tiny_budgeted.json --output runs/ddcolor_tiny_budgeted
python train.py --config configs/resnet18.json --output runs/resnet18
python train.py --config configs/resnet34.json --output runs/resnet34
```

Tiếp tục run bị gián đoạn bằng đúng config và thư mục cũ:

```powershell
python train.py `
  --config configs/ddcolor_tiny.json `
  --output runs/ddcolor_tiny `
  --resume runs/ddcolor_tiny/latest.pth
```

### Domain adaptation Việt Nam

```powershell
python fine_tune_vietnam.py `
  --config configs/ddcolor_vietnam_adaptation_v2.json `
  --output runs/ddcolor_vietnam_adaptation_v2
```

Pipeline hỗ trợ protocol riêng, checkpoint nguồn, teacher checkpoint, distillation, dữ liệu Wikimedia, dữ liệu lịch sử và resume. Mỗi run phải giữ nguyên config, manifest và seed để tái lập.

### Chọn và đánh giá mô hình

Chỉ đưa các run hoàn chỉnh vào bước chọn:

```powershell
python select_model.py runs/ddcolor_tiny runs/resnet18 runs/resnet34
```

Đánh giá checkpoint đã chọn:

```powershell
python evaluate.py `
  --selection reports/model_selection.json `
  --output reports/final_evaluation
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

## Đóng gói Windows

Build ứng dụng từ thư mục `ImageNet`:

```powershell
.\build_windows_app.ps1
```

Build sạch:

```powershell
.\build_windows_app.ps1 -Clean
```

Kết quả được tạo trong `ImageNet/release/SacViet`. Model và cấu hình được đặt cạnh `SacViet.exe`, cho phép cập nhật checkpoint mà không build lại mã nguồn.

Khi thay checkpoint DDColor:

1. Đóng ứng dụng.
2. Sao chép checkpoint mới bằng tên tạm.
3. Kiểm tra file và SHA-256.
4. Đổi tên thành `best.pth`.
5. Cập nhật đồng thời đường dẫn và checksum trong `production_selection.json`.
6. Giữ checkpoint cũ để rollback.

## Cấu trúc dự án

```text
KHOA_LUAN/
├── .gitignore
├── README.md
└── ImageNet/
    ├── colorization/          # model, dataset, inference, metric, temporal runtime
    ├── configs/               # cấu hình DDColor, VietColorNet và ablation
    ├── data/                  # raw data, protocol và manifest
    ├── examples/              # ảnh mẫu cho giao diện
    ├── pretrained/            # checkpoint pretrained
    ├── reports/               # selection, audit và kết quả đánh giá
    ├── runs/                  # checkpoint và lịch sử huấn luyện
    ├── tests/                 # unit tests
    ├── third_party/DDColor/   # mã nguồn DDColor upstream
    ├── demo.py                # ứng dụng Gradio Sắc Việt
    ├── train.py               # huấn luyện ImageNet/baseline
    ├── fine_tune_vietnam.py   # domain adaptation và distillation
    ├── evaluate.py            # đánh giá ImageNet/Việt Nam
    ├── verify_product.py      # kiểm tra selection, checksum và runtime
    └── requirements.txt
```

Các thư mục `.venv/`, `data/`, `runs/`, `pretrained/`, `build/`, `dist/`, `release/`, `demo_outputs/` và checkpoint `*.pth`/`*.pt` được bỏ qua để tránh đưa dữ liệu hoặc artefact lớn vào Git ngoài ý muốn. Repository chỉ giữ `reports/production_selection.json`; các report khác phải được thêm có chủ đích nếu cần công bố.

## Giới hạn

- Tô màu là bài toán một đầu vào có nhiều đáp án hợp lý; PSNR/SSIM không đủ để kết luận chất lượng. Cần đọc cùng Delta E, AB-MAE, LPIPS, chroma và ảnh mẫu.
- ImageNet dùng trong protocol là bản 64×64 rồi nội suy lên kích thước huấn luyện; nội suy không tạo thêm chi tiết thật.
- Checkpoint DDColor chính thức đã học ImageNet. Fine-tune từ checkpoint này là transfer learning, không phải huấn luyện từ đầu.
- Tập dữ liệu đã tham gia lựa chọn kiến trúc/hyperparameter chỉ là development set; kết luận cuối cần nguồn độc lập chưa từng dùng trong thiết kế.
- Ảnh hư hỏng nặng, mất mảng hoặc trầy xước lớn cần thêm bước restoration/inpainting.
- Video dài trên GPU laptop cần nhiều thời gian và bộ nhớ; tính năng video trong demo mặc định bị tắt.
- Kết quả định lượng chỉ có ý nghĩa khi protocol, checkpoint, checksum và mã đánh giá được cố định.

## Phần mềm bên thứ ba

DDColor:

- Repository: <https://github.com/piddnad/DDColor>
- Commit tích hợp: `2adb63f2656ac41cbdf7b894cddd94121a3faf13`
- Giấy phép: Apache License 2.0; file `LICENSE` gốc được giữ trong `ImageNet/third_party/DDColor/`.
- Bài báo: Xia, Z., Shen, X., Huang, J., Chen, J., & Ji, R. (2023), “DDColor: Towards Photo-Realistic and Consistent Image Colorization with Dual Decoders”, ICCV 2023.

Các dependency khác giữ nguyên giấy phép tương ứng của từng dự án. Thư mục demo vendored của DDColor thuộc upstream và không được ứng dụng Sắc Việt gọi trực tiếp.
