# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Hoàng Công Thành |
| MSSV | 2A202601662 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/stephhoang30/K4-Track2-Day21-CI-CD-for-AI-Systems-HoangCongThanh-2A202601662 |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 5 | **0.7149** | 0.8740 |
| 2 | 100 | 0.1 | 3 | 0.7109 | **0.8780** |
| 3 | 50 | 0.1 | 2 | 0.6193 | 0.8500 |

<sub>Hai lần chạy còn lại (300/0.05/4 → F1 0.7070 và 200/0.05/3 → F1 0.7014) xem ảnh `01-mlflow-ui.png`.</sub>

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này cho F1 cao nhất. Đáng chú ý là lần có accuracy cao nhất (lần 2) lại không
phải lần có F1 cao nhất: accuracy chênh 0.004 theo chiều này thì F1 chênh 0.004 theo chiều
ngược lại. Accuracy đã bão hòa quanh 0.87, không còn phân biệt được các mô hình, nên chọn
theo nó là chọn nhầm. Hạ `learning_rate` xuống 0.05 buộc phải tăng `n_estimators` để bù,
nhưng 300 cây vẫn thua 200 cây ở `learning_rate` 0.1. Giới hạn thật nằm ở `max_depth`: cây
sâu 2 tầng không biểu diễn nổi tương tác giữa học vấn, hôn nhân và số giờ làm, khiến F1 rơi
xuống 0.6193 — dưới ngưỡng — dù accuracy vẫn 0.85.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult chỉ có 24.8% mẫu thu nhập trên 50K. Một mô hình luôn trả lời "thu nhập thấp" vẫn
đạt accuracy 0.752, chỉ kém mô hình tốt nhất của em 0.12 điểm dù không học được gì — phần
lớn giá trị của accuracy đến từ việc đoán đúng lớp đa số, thứ vốn không cần mô hình.

F1 của lớp dương là trung bình điều hòa của precision và recall chỉ trên lớp thu nhập cao,
đúng nhóm bài toán quan tâm; mô hình đoán bừa có recall bằng 0 nên F1 bằng 0, phơi bày ngay
thứ accuracy che giấu. Không dùng `average="weighted"` vì nó lấy trung bình có trọng số theo
số mẫu, để lớp đa số chiếm ưu thế và tái lập đúng vấn đề của accuracy; cũng không dùng
`average="macro"` vì nó tính cả F1 lớp âm rồi chia đôi, làm loãng tín hiệu cần đo.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `import mlflow` báo thiếu `pkg_resources` | mlflow 2.13 còn dùng `pkg_resources`, venv pip đời mới không kèm `setuptools` | Thêm `setuptools<81` vào `requirements.txt` |
| `pytest` ghi run rác vào `mlflow.db` thật | Test gọi `train()` nên MLflow ghi vào tracking store mặc định | Thêm `tests/conftest.py` trỏ MLflow vào thư mục tạm |
| `git push` không kích hoạt workflow | Repo là fork, GitHub chặn đến khi chủ repo xác nhận | Loại trừ (`workflow_dispatch` chạy được, `push` thì không) rồi bật Actions |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (`train_batch1`, 22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`, 44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** F1 tăng 0.0205 còn accuracy chỉ tăng 0.0080. Hai nửa dữ liệu cùng phân phối
nên dữ liệu mới không mang thêm thông tin mới về mặt cấu trúc; mức tăng đến từ việc `max_depth=5`
đủ chỗ để tận dụng thêm mẫu, và khoảng 5.500 mẫu thu nhập cao bổ sung rơi đúng vào lớp mà
số mẫu đang là ràng buộc. Đó cũng là lý do F1 nhích nhiều hơn accuracy gần ba lần. Điều
được kiểm chứng ở đây không phải con số cao hơn mà là vòng tự động chạy đúng: một commit
dữ liệu đi hết từ `dvc push` đến model đang phục vụ trên VM, không ai can thiệp.

<sub>Prefix DVC trên S3 là `dvc-k4/` thay vì `dvc/` để tách dữ liệu lab này khỏi lab cũ dùng chung bucket.</sub>

---

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1 - DagsHub: chưa thực hiện.
- [x] Bonus 2 - Ngưỡng quyết định: quét 0.10–0.90; ngưỡng 0.30 cho F1 0.7368, hơn mặc định 0.50 là +0.0219.
- [x] Bonus 3 - Precision/recall: `outputs/detail.txt`. Ở ngưỡng 0.5 mô hình bỏ sót 49 người thu nhập cao nhưng chỉ gán nhầm 12 — bỏ sót tốn kém hơn, nên hạ ngưỡng đổi precision lấy recall là đúng.
- [x] Bonus 4 - Hoàn trả phiên bản trước: Train đẩy lên `artifacts/candidate/`, quality gate so F1 mới với `artifacts/current/report.json`, qua mới promote.
- [x] Bonus 5 - Cảnh báo lệch dữ liệu: cảnh báo nếu tỷ lệ lớp dương lệch quá 5 điểm phần trăm so với 24.8%, ghi vào `outputs/report.json`.

<sub>Ảnh `04-curl-api.png` là kết xuất output thật của `curl` kèm mốc thời gian UTC, không phải ảnh chụp cửa sổ terminal.</sub>
