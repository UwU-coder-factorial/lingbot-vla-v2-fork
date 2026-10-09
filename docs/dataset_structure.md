# Cấu Trúc Dataset Hiện Tại (OpenArm)

Dữ liệu được lưu trữ dạng Episode theo cấu trúc HDF5/Dict tương tự LeRobot / ACT:

| Dataset Field | Shape / Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `observations/images/chest_rgb` | `[N, 240, 424, 3]` `uint8` | Ảnh màu RGB từ camera gắn ngực (chest camera). |
| `observations/images/chest_depth` | `[N, 240, 424]` `uint16` | Độ sâu đo bằng **mm**, kẹp trong khoảng [200, 1200]. Giá trị 0 là camera không đo được. |
| `observations/qpos` | `[N, 16]` `float32` | Vị trí góc đo được của 16 khớp (State). |
| `observations/qvel` | `[N, 16]` `float32` | Vận tốc khớp; với gripper là tốc độ mở/đóng. |
| `observations/effort` | `[N, 16]` `float32` | Mô-men xoắn / lực đo được của khớp. |
| `action` | `[N, 16]` `float32` | **Nhãn target để train**. Chính là quỹ đạo vị trí khớp `qpos` đo được (thay vì lệnh thô). |
| `joint_commands` | `[N, 16]` `float32` | Lệnh teleop gửi qua proxy (có thể chứa NaN khi mất lệnh hoặc trễ > 0.2s). |
| `timestamp_ns` | `[N]` `int64` | Mốc thời gian lấy mẫu (nanosecond). |

### Đặc điểm quan trọng:
1. **Camera**: 1 góc quay (`chest_rgb`), độ phân giải `240x424`.
2. **Depth**: Có sẵn kênh depth vật lý (`chest_depth`), không bắt buộc phải chạy model sinh Depth teacher (MoRGBD).
3. **Action / State space**: 16 chiều (OpenArm dual-arm: 2 tay x 7 DoF + 2 Grippers = 16 DoF).
4. **Nhãn train**: Dùng `qpos` làm action label (chuẩn imitation learning / behavior cloning).

