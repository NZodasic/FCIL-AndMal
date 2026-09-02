# Hướng dẫn Chạy Benchmark Pipeline (FCIL-AndMal2020)

---

### 1. Di chuyển vào thư mục dự án
```bash
cd /path/to/FCIL-AndMal
```
*(Thay `/path/to/FCIL-AndMal` bằng đường dẫn thực tế đến folder project trên máy Mac của bạn)*

---

### 2. (Tùy chọn) Kích hoạt môi trường Python (nếu dùng venv hoặc conda)
```bash
# Nếu dùng venv:
source venv/bin/activate

# Hoặc nếu dùng Conda:
conda activate <ten_env_cua_ban>
```

---

### 3. Kiểm tra danh sách kịch bản thử nghiệm
```bash
python3 run_all.py --list
```
*Lệnh này hiển thị 2 kịch bản Client (20 clients và 50 clients), cùng danh sách 10 bài toán thực nghiệm cốt lõi.*

---

### 4. Chạy kiểm tra nhanh (Dry Run - không tốn thời gian train)
Lệnh này giúp bạn kiểm tra toàn bộ luồng tự động tìm `./Dataset`, chia data và tạo lệnh mà không mất thời gian chờ train:
```bash
python3 run_all.py --dry_run
```

---

### 5. CÁC LỆNH CHẠY CHÍNH

* **Chạy Kịch bản 20 Clients (10 Experiments cho 20 clients):**
  ```bash
  python3 run_all.py --clients 20
  ```

* **Chạy Kịch bản 50 Clients (10 Experiments cho 50 clients):**
  ```bash
  python3 run_all.py --clients 50
  ```

* **Chạy TOÀN BỘ cả 2 Kịch bản 20 Clients và 50 Clients (20 runs total):**
  ```bash
  python3 run_all.py
  ```

---

### 💡 Các tùy chọn hữu ích khác:

* **Chạy ẩn trong nền (treo máy Mac không lo tắt terminal):**
  ```bash
  nohup python3 run_all.py > run_benchmark.log 2>&1 &
  ```
  *Xem tiến độ live:*
  ```bash
  tail -f run_benchmark.log
  ```

* **Chỉ chạy 4 bài Centralized cho 20 clients:**
  ```bash
  python3 run_all.py --clients 20 --only Centralized
  ```

* **Chỉ chạy các bài FL cho 50 clients:**
  ```bash
  python3 run_all.py --clients 50 --only FL
  ```

---

### 📊 Cấu trúc Kết quả Xuất ra:
- **Kịch bản 20 Clients:**
  - Tệp Excel metrics: `./EXPERIMENT/20clients/evaluation_results.xlsx`
  - Tệp JSON tóm tắt: `./EXPERIMENT/20clients/run_summary.json`
  - Log chi tiết từng bài: `./EXPERIMENT/20clients/_logs/<Case_Name>.stdout.log`

- **Kịch bản 50 Clients:**
  - Tệp Excel metrics: `./EXPERIMENT/50clients/evaluation_results.xlsx`
  - Tệp JSON tóm tắt: `./EXPERIMENT/50clients/run_summary.json`
  - Log chi tiết từng bài: `./EXPERIMENT/50clients/_logs/<Case_Name>.stdout.log`

- **Tóm tắt tổng quát:**
  - `./EXPERIMENT/run_summary.json`