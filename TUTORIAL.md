
---

### 1. Di chuyển vào thư mục dự án
```bash
cd /path/to/FCIL-AndMal
```
*(Thay `/path/to/FCIL-AndMal` bằng đường dẫn thực tế đến folder project của bạn trên Mac)*

---

### 2. (Tùy chọn) Kích hoạt môi trường Python (nếu dùng venv hoặc conda)
```bash
# Nếu dùng venv:
source venv/bin/activate

# Hoặc nếu dùng Conda:
conda activate <ten_env_cua_ban>
```

---

### 3. Kiểm tra danh sách 16 bài thử nghiệm (để xác nhận)
```bash
python3 run_all.py --list
```

---

### 4. Chạy kiểm tra nhanh (Dry Run - không tốn thời gian train)
Lệnh này giúp bạn kiểm tra toàn bộ luồng tự động tìm `./Dataset`, chia data và tạo lệnh mà không mất thời gian đợi train:
```bash
python3 run_all.py --dry_run
```

---

### 5. LỆNH CHÍNH — Chạy toàn bộ Benchmark (Stage 0, 1, 2 + 16 Experiments)
```bash
python3 run_all.py
```

---

### 💡 Các lệnh tiện ích khác (chạy từng phần nếu muốn):

* **Chạy ẩn trong nền (để treo máy Mac chạy không lo tắt terminal):**
  ```bash
  nohup python3 run_all.py > run_benchmark.log 2>&1 &
  ```
  *Xem tiến độ live:*
  ```bash
  tail -f run_benchmark.log
  ```

* **Chỉ chạy 4 bài Centralized:**
  ```bash
  python3 run_all.py --only Centralized
  ```

* **Chỉ chạy các bài FL 20 Client:**
  ```bash
  python3 run_all.py --only K20
  ```

* **Chỉ chạy các bài FL 50 Client:**
  ```bash
  python3 run_all.py --only K50
  ```

---

### 📊 Kết quả sau khi chạy xong:
- **Tệp Excel tổng hợp full metrics:** `./EXPERIMENT/evaluation_results.xlsx`
- **Tệp JSON tóm tắt:** `./EXPERIMENT/run_summary.json`
- **Log chi tiết từng experiment:** `./EXPERIMENT/_logs/<Case_Name>.stdout.log`