# FCIL-AndMal: Học Tăng Cường Phân Lớp Phân Tán Cho Phát Hiện Họ Mã Độc Android

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: FCIL](https://img.shields.io/badge/Framework-FCIL%20%7C%20MalFSCIL-green.svg)](#)

Nền tảng nghiên cứu khoa học toàn diện cho bài toán **Federated Class-Incremental Learning (FCIL)** và **Few-Shot Class-Incremental Learning (MalFSCIL)** áp dụng trong phân loại các họ mã độc Android trên bộ dữ liệu chuẩn **CIC-AndMal-2020**.

---

## 📌 Tổng Quan

Hệ thống phát hiện mã độc Android thực tế đối mặt với 2 thách thức cốt lõi:
1. **Sự Tiến Hóa Theo Thời Gian (Temporal Evolution)**: Các họ mã độc mới liên tục xuất hiện theo thời gian. Mô hình huấn luyện truyền thống bị hiện tượng **quên lãng thảm khốc (catastrophic forgetting)** khi cập nhật dữ liệu mới.
2. **Quyền Riêng Tư & Phân Tán Dữ Liệu (Spatial Privacy & Distribution)**: Dữ liệu mã độc nằm rải rác trên các thiết bị đầu cuối, máy chủ doanh nghiệp và không thể thu thập tập trung do quy định bảo mật riêng tư.

**FCIL-AndMal** tích hợp các phương pháp tiên tiến về Continual Learning (CIL), Few-Shot Learning (FSCIL) và Học Phân Tán (FedAvg, FedNova) trong điều kiện dữ liệu lệch nhãn phi IID (Dirichlet $\alpha = 0.5$).

---

## ✨ Tính Năng Nổi Bật

- **Kịch Bản 5 Task Học Tăng Cường**: 15 lớp dữ liệu (Benign + 14 họ mã độc) chia thành 5 task nối tiếp.
- **Biểu Diễn Dữ Liệu Đa Mô Thức (Multi-Modal Fusion)**: Hỗ trợ đặc trưng **Static**, **Dynamic**, và **Fused** (kết hợp căn chỉnh static + dynamic).
- **Phân Chịu Dữ Liệu Phân Tán Phi IID**: Phân bố Dirichlet ($\alpha = 0.5$) với sự tham gia tăng dần của client ($12 \to 14 \to 16 \to 18 \to 20$ client active).
- **Thuật Toán Tiên Tiến**:
  - *CIL Tập Trung*: Fine-tuning, Joint Training, Elastic Weight Consolidation (**EWC**), Learning without Forgetting (**LwF**), Exemplar Replay, và **SPCIL**.
  - *Few-Shot CIL*: **MalFSCIL** (Variational Reconstruction + Angular Margin ArcFace + Graph Attention Prototypes).
  - *Tập Hợp Phân Tán*: **FedAvg** và **FedNova**.
- **Báo Cáo Tự Động Rõ Ràng**: Xuất các chỉ số đánh giá chuẩn (Macro-F1, Accuracy, Average Forgetting, Malware F1) ra file Excel (`evaluation_results.xlsx`) và checkpoint mô hình PyTorch.

---

## 📂 Cấu Trúc Dự Án

```
FCIL-AndMal/
├── config/                 # Cấu hình hệ thống & định nghĩa task
│   ├── config.py           # Dataclass configs (ScenarioConfig, ModelConfig, FLConfig, v.v.)
│   ├── task_config.py      # Ánh xạ nhãn 5 task (15 lớp mã độc/lành tính)
│   └── paths.py            # Tiện ích quản lý đường dẫn
├── data/                   # Xử lý & nạp dữ liệu
│   ├── prepare_dataset.py  # Stage 1: Gộp, chuẩn hóa & chia held-out test split
│   ├── partition.py        # Stage 2: Phân chia dữ liệu client Dirichlet non-IID
│   ├── dataset.py          # PyTorch TabularMalwareDataset & FLTaskDataset loaders
│   ├── schema.py           # Kiểm tra schema đặc trưng & làm sạch metadata
│   └── synthetic_generator.py # Bộ tạo dữ liệu giả lập cho phát triển/kiểm thử
├── models/                 # Kiến trúc mạng & backbone
│   ├── fcil_model.py       # Bộ định tuyến mô hình FCIL & feature extractors
│   ├── backbones.py        # Backbone MLP, 1D-CNN, TCN, và Fused
│   └── fused_model.py      # Mạng kết hợp đa mô thức (Static + Dynamic)
├── methods/                # Thuật toán Continual & Few-Shot CIL
│   ├── base.py             # Lớp cơ sở cho các phương pháp CIL
│   ├── ewc.py              # Elastic Weight Consolidation
│   ├── lwf.py              # Learning without Forgetting
│   ├── replay.py           # Exemplar Replay
│   ├── spcil.py            # Self-Paced Class-Incremental Learning
│   └── malfscil.py         # Chiến lược MalFSCIL Few-Shot CIL
├── federated/              # Môi trường giả lập Học Phân Tán
│   ├── client.py           # Client FL tối ưu cục bộ
│   ├── server.py           # Server FL quản lý tập hợp toàn cục
│   └── aggregators/        # Thuật toán tập hợp FedAvg & FedNova
├── training/               # Bộ huấn luyện Tập Trung & Phân Tán
│   ├── trainer.py          # Centralized continual trainer
│   ├── evaluator.py        # Đánh giá ma trận liên tục & theo dõi chỉ số
│   └── checkpoint.py       # Trình quản lý checkpoint mô hình PyTorch
├── main.py                 # File thực thi chính & giao diện CLI
├── requirements.txt        # Các thư viện phụ thuộc Python
├── README.md               # Tài liệu hướng dẫn Tiếng Anh
└── README_VN.md            # Tài liệu hướng dẫn Tiếng Việt
```

---

## 📅 Lịch Trình 5 Task Phân Loại

15 lớp chuẩn được phân chia qua 5 task nối tiếp như sau:

| Task | Nhãn Xuất Hiện Mới | Số Client Active ($K=20$) | Tổng Số Lớp Tích Lũy |
| :---: | :--- | :---: | :---: |
| **Task 1** | `Benign`, `PUA`, `Backdoor` | 12 / 20 | 3 |
| **Task 2** | `Adware`, `TrojanBanker`, `TrojanSpy` | 14 / 20 | 6 |
| **Task 3** | `NoCategory`, `Trojan`, `Riskware` | 16 / 20 | 9 |
| **Task 4** | `FileInfector`, `Ransomware`, `TrojanDropper` | 18 / 20 | 12 |
| **Task 5** | `Scareware`, `ZeroDay`, `TrojanSMS` | 20 / 20 | 15 |

---

## ⚙️ Cài Đặt Môi Trường

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/FCIL-AndMal.git
   cd FCIL-AndMal
   ```

2. **Tạo & Kích Hoạt Môi Trường Ảo**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Cài Đặt Thư Viện Phụ Thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quy Trình Chuẩn Bị Dữ Liệu

### **Stage 1: Xử Lý Dữ Liệu Thô (Prepare Dataset)**
Chuẩn hóa các file CSV thô, làm sạch cột đặc trưng, căn chỉnh static/dynamic và tạo tập test held-out độc lập:

```bash
python3 -m data.prepare_dataset \
  --root /duong/dan/toi/dataset_tho \
  --output_dir /duong/dan/toi/dataset_output \
  --type all \
  --summary
```

### **Stage 2: Phân Chia Dữ Liệu Client (Partition Data)**
Phân chia dữ liệu đã chuẩn bị thành các slice cho client qua 5 task:

```bash
# Phân chia đặc trưng Fused (20 Clients)
python3 -m data.partition \
  --dataset /duong/dan/toi/dataset_output/fused/train.parquet \
  --feature_type fused \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /duong/dan/toi/fl_data_partitions

# Phân chia đặc trưng Dynamic (20 Clients)
python3 -m data.partition \
  --dataset /duong/dan/toi/dataset_output/dynamic/train.parquet \
  --feature_type dynamic \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /duong/dan/toi/fl_data_partitions

# Phân chia đặc trưng Static (20 Clients)
python3 -m data.partition \
  --dataset /duong/dan/toi/dataset_output/static/train.parquet \
  --feature_type static \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /duong/dan/toi/fl_data_partitions
```

---

## 💻 Hướng Dẫn Chạy Thử Nghiệm

Thực thi thử nghiệm qua `main.py`. Thay `--device cuda` bằng `--device cpu` hoặc `--device mps` tùy theo thiết bị của bạn.

### **1. Thử Nghiệm Tập Trung (Centralized Continual Learning)**

#### **Phương Pháp Đề Xuất Chính: MalFSCIL (Đặc Trưng Fused)**
```bash
python3 main.py \
  --mode centralized \
  --feature_type fused \
  --method malfscil \
  --prepared_dir /duong/dan/toi/dataset_output \
  --partition_dir /duong/dan/toi/fl_data_partitions \
  --raw_root /duong/dan/toi/dataset_tho \
  --rounds_per_task 50 \
  --device cpu
```

#### **Các Phương Pháp Baseline Tập Trung (Finetune, EWC, LwF, Replay)**
```bash
# Fine-Tuning Baseline
python3 main.py --mode centralized --feature_type fused --method finetune \
  --prepared_dir /duong/dan/toi/dataset_output --partition_dir /duong/dan/toi/fl_data_partitions --device cpu

# Elastic Weight Consolidation (EWC)
python3 main.py --mode centralized --feature_type fused --method ewc --ewc_lambda 5000.0 \
  --prepared_dir /duong/dan/toi/dataset_output --partition_dir /duong/dan/toi/fl_data_partitions --device cpu

# Learning without Forgetting (LwF)
python3 main.py --mode centralized --feature_type fused --method lwf --lwf_temp 2.0 --lwf_alpha 1.0 \
  --prepared_dir /duong/dan/toi/dataset_output --partition_dir /duong/dan/toi/fl_data_partitions --device cpu

# Exemplar Replay
python3 main.py --mode centralized --feature_type fused --method replay --buffer_size 20 \
  --prepared_dir /duong/dan/toi/dataset_output --partition_dir /duong/dan/toi/fl_data_partitions --device cpu
```

---

### **2. Thử Nghiệm Phân Tán (Federated Class-Incremental Learning - FCIL)**

#### **Học Phân Tán: FedAvg + Exemplar Replay**
```bash
python3 main.py \
  --mode federated \
  --feature_type fused \
  --method replay \
  --aggregator fedavg \
  --n_clients 20 \
  --local_epochs 5 \
  --rounds_per_task 50 \
  --prepared_dir /duong/dan/toi/dataset_output \
  --partition_dir /duong/dan/toi/fl_data_partitions \
  --device cpu
```

#### **Học Phân Tán: FedNova + Exemplar Replay**
```bash
python3 main.py \
  --mode federated \
  --feature_type fused \
  --method replay \
  --aggregator fednova \
  --n_clients 20 \
  --local_epochs 5 \
  --rounds_per_task 50 \
  --prepared_dir /duong/dan/toi/dataset_output \
  --partition_dir /duong/dan/toi/fl_data_partitions \
  --device cpu
```

---

## 📊 Báo Cáo Kết Quả & Artifacts

Tất cả log thực thi, cấu hình, kết quả chỉ số và checkpoint mô hình tự động được lưu trữ tại `./EXPERIMENT/`:

```
EXPERIMENT/
├── evaluation_results.xlsx    # Báo cáo Excel tổng hợp Accuracy, Macro-F1, Malware F1, và Forgetting
└── <TEN_TEN_THU_NGHIEM>/
    ├── experiment_config.json # Snapshot cấu hình siêu tham số
    ├── academic_experiment.log # Log chi tiết quá trình chạy
    └── checkpoints/           # Weight mô hình PyTorch (.pt) qua từng task
```

---

## 📝 Trích Dẫn

Nếu bạn sử dụng benchmark hoặc mã nguồn này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@article{fcil_andmal2026,
  title={Federated Class-Incremental Learning for Android Malware Family Detection},
  author={Research Team},
  journal={Academic Research Platform},
  year={2026}
}
```

---

## 📄 Giấy Phép

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
