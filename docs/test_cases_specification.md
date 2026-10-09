# Ma Trận Đặc Tả Bộ Test Case Toàn Diện Cho Dataset OpenArm

**Nguồn dữ liệu:** [`Tuyen062004/data_openarm_8_10`](https://huggingface.co/datasets/Tuyen062004/data_openarm_8_10)  
**Mục tiêu:** 
1. Đảm bảo training không bị lỗi (Zero-crash training).
2. Đảm bảo kết quả đầu ra không bị lỗi (Safe & Valid robot control).
3. Đảm bảo máy yếu chạy được (Low-spec RAM / 1x 4090 / CPU verification).
4. Không có gì mock (100% Real HDF5 binary, real tensors, real kinematics).
5. Bao phủ tối đa các trường hợp biên (Edge Cases).

---

## 1. Danh Mục Test Cases Chi Tiết

| Mã TC | Tên Test Case | Nhóm kiểm thử | Mục đích & Rủi ro ngăn chặn | Phương pháp kiểm tra (Không Mock) |
| :--- | :--- | :--- | :--- | :--- |
| **TC-DATA-01** | HDF5 Schema & Type Verifier | Data Integrity | File HDF5 thiếu key hoặc sai shape (`[N, 240, 424, 3]` uint8, `[N, 16]` float32) làm văng DataLoader. | Đọc file `.hdf5` thật bằng `h5py`, assert đúng dtypes và dims. |
| **TC-DATA-02** | NaN Quarantine | Data Integrity | `joint_commands` chứa `NaN` lọt sang `action` hoặc `qpos` làm gradient nổ thành NaN. | Quét `np.isnan()` trên `action` và `qpos`, kiểm tra bộ lọc loại bỏ episode lỗi. |
| **TC-DATA-03** | Corrupt Episode Rejection | Data Integrity | Episode bị đứt đoạn hoặc hư hỏng ghi vào file txt gây crash train giữa chừng. | Bơm lỗi nhị phân vào episode, kiểm tra script tiền xử lý tự động loại bỏ khỏi danh sách. |
| **TC-DATA-04** | Zero-Depth Sensor Blindspot | Data Integrity | Cảm biến độ sâu bị mù (0 mm) gây lỗi chia cho 0 hoặc giá trị âm khi normalize. | Kiểm tra các pixel độ sâu $= 0$, chuẩn hóa dạng `np.where(d > 0, d/1000, 0)`. |
| **TC-DATA-05** | Timestamp Monotonicity | Data Integrity | Gói tin mạng bị đảo thứ tự làm quỹ đạo nhảy cóc lùi thời gian. | Kiểm tra $\Delta t = t_{i+1} - t_i > 0$, tự động sắp xếp lại index nếu bị đảo. |
| **TC-KIN-01** | 16-DoF Split & Delta Mapping | Kinematics | Khớp tay cần dùng delta ($\Delta q$), gripper cần dùng giá trị tuyệt đối ($0..1$). | Cắt lát 14 khớp tay và 2 khớp kẹp, kiểm tra công thức $\Delta q$ và absolute gripper. |
| **TC-KIN-02** | Canonical 55-Dim Padding | Kinematics | LingBot-VLA dùng không gian 55 chiều, nếu padding sai vị trí model sẽ điều khiển sai khớp. | Map 16-DoF vào đúng vị trí `arm.position` (14) và `effector.position` (2), các chiều khác $= 0$. |
| **TC-KIN-03** | Extreme Angle Clipping | Kinematics | Nhiễu cảm biến góc khớp vọt lên $10^6$ làm hỏng bảng thống kê phân phối. | Áp dụng thuật toán 99-percentile bounds để kẹp giá trị ngoại lai an toàn. |
| **TC-HW-01** | Lazy Streaming (RAM Budget) | Hardware | Đọc cùng lúc 45 file HDF5 (~16 GB) làm tràn RAM hệ thống của máy yếu. | Kiểm tra truy xuất HDF5 dạng slice (`dset[i:i+1]`), RAM mỗi lần đọc $< 400\text{ KB}$. |
| **TC-HW-02** | Micro-Batch Size 1 & 2 | Hardware | Batch size lớn gây CUDA OOM trên GPU 24GB hoặc máy yếu. | Chạy forward/backward mô phỏng trên batch 1 và 2 với gradient accumulation. |
| **TC-HW-03** | CPU Fallback Execution | Hardware | Máy yếu không có GPU Nvidia vẫn phải chạy được smoke test tiền xử lý và validation. | Chạy toàn bộ lớp chiếu Linear và xử lý tensor trên CPU `device('cpu')`. |
| **TC-HW-04** | Multi-Worker Shared Memory | Hardware | DataLoader nạp đa luồng bị leak file descriptors hoặc tràn `/dev/shm`. | Kiểm tra đóng kết nối HDF5 sau mỗi epoch (`f.close()`). |
| **TC-TRAIN-01** | Flow Matching Interpolation | Training | Công thức $x_t = (1-t)x_0 + t x_1$ sinh giá trị vô định hoặc gradient NaN. | Tính vận tốc mục tiêu $v_t = x_1 - x_0$ trên tensor thực, kiểm tra finite. |
| **TC-TRAIN-02** | Constant Joint Zero-Std Protection | Training | Khớp robot không cử động (std $= 0$) gây lỗi chia cho $0$ trong lớp Normalization. | Chuẩn hóa mảng có std $= 0$ với $\epsilon = 10^{-6}$, đảm bảo không sinh NaN. |
| **TC-TRAIN-03** | Short Episode Padding | Training | Episode ngắn hơn độ dài action chunk ($N < 30$) làm văng index slicing. | Tự động lặp lại frame cuối cùng để bù đủ độ dài chunk $H=30$. |
| **TC-TRAIN-04** | Mixed Precision Gradient Parity | Training | Action Expert dùng bfloat16 nhưng head dùng fp32 bị lệch kiểu khi backward. | Kiểm tra tính tương thích khi backward giữa bfloat16 và float32 linear projection. |
| **TC-OUT-01** | Gripper Boundary Conformance | Output Quality | Model dự đoán giá trị kẹp âm hoặc vượt quá 1 làm động cơ cố kéo đứt dây cáp. | Kẹp giá trị `torch.clamp(action, 0.0, 1.0)` đảm bảo vật lý an toàn. |
| **TC-OUT-02** | Velocity Jerk Limit (Motor Safety) | Output Quality | Mô hình xuất lệnh đổi hướng đột ngột làm giật gãy tay robot ngoài đời. | Kiểm tra đạo hàm bậc 1 của quỹ đạo: $\max \|\Delta q\| < 0.20\text{ rad/step}$ ở tần số 50Hz. |
| **TC-OUT-03** | Output Dimension Determinism | Output Quality | Đầu ra dự đoán sai chiều không thể gửi về proxy điều khiển robot. | Assert `output.shape == (B, H, 16)` cho mọi batch. |

---

## 2. Hướng Dẫn Chạy Test Suite

Chạy trực tiếp file test (sử dụng thư viện `unittest` chuẩn của Python):
```powershell
python tests/test_openarm_suite.py
```
Hoặc dùng `pytest` để có báo cáo chi tiết:
```powershell
pytest -v -s tests/test_openarm_suite.py
```

