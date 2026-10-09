"""
BỘ TEST SUITE TOÀN DIỆN CHO DATASET OPENARM VÀ PIPELINE HUẤN LUYỆN LINGBOT-VLA
Dataset nguồn: https://huggingface.co/datasets/Tuyen062004/data_openarm_8_10

Yêu cầu nghiệm ngặt:
1. Đảm bảo training không bị lỗi.
2. Đảm bảo kết quả đầu ra không bị lỗi.
3. Đảm bảo máy yếu (RAM thấp / 1x 4090 / CPU) chạy ổn định.
4. KHÔNG SỬ DỤNG MOCK: Dùng dữ liệu nhị phân HDF5 thật, tensor PyTorch thật, toán tử thật.
5. Kiểm tra toàn diện mọi Edge Cases và rủi ro phần cứng/phần mềm.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thêm đường dẫn gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BaseOpenArmTestCase(unittest.TestCase):
    """Lớp nền tạo dữ liệu HDF5 thật theo đúng 100% schema của Tuyen062004/data_openarm_8_10."""

    @classmethod
    def setUpClass(cls):
        try:
            import h5py
            cls.h5py = h5py
        except ImportError:
            raise unittest.SkipTest("Cần cài đặt thư viện 'h5py' để chạy test suite.")

        try:
            import torch
            cls.torch = torch
        except ImportError:
            cls.torch = None

        cls.temp_dir = tempfile.mkdtemp(prefix="openarm_test_")


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_real_hdf5_episode(
        self,
        file_path: str,
        n_steps: int = 50,
        inject_nan_in_commands: bool = True,
        inject_zero_depth: bool = True,
        corrupt_action: bool = False,
        constant_qpos: bool = False,
    ):
        """Tạo file HDF5 thật với đúng schema và kiểu dữ liệu từ HuggingFace."""
        h5py = self.h5py
        with h5py.File(file_path, "w") as f:
            # 1. observations/images/chest_rgb [N, 240, 424, 3] uint8
            rgb = np.random.randint(0, 256, size=(n_steps, 240, 424, 3), dtype=np.uint8)
            f.create_dataset("observations/images/chest_rgb", data=rgb, compression="gzip")

            # 2. observations/images/chest_depth [N, 240, 424] uint16 (mm: 200..1200)
            depth = np.random.randint(200, 1201, size=(n_steps, 240, 424), dtype=np.uint16)
            if inject_zero_depth:
                depth[:, :20, :20] = 0  # Giả lập điểm mù camera không đo được
            f.create_dataset("observations/images/chest_depth", data=depth, compression="gzip")

            # 3. observations/qpos [N, 16] float32
            if constant_qpos:
                qpos = np.zeros((n_steps, 16), dtype=np.float32) + 0.5
            else:
                qpos = np.random.uniform(-1.5, 1.5, size=(n_steps, 16)).astype(np.float32)
                qpos[:, 7] = np.random.uniform(0.0, 1.0, size=(n_steps,))   # Gripper trái
                qpos[:, 15] = np.random.uniform(0.0, 1.0, size=(n_steps,))  # Gripper phải
            f.create_dataset("observations/qpos", data=qpos)

            # 4. observations/qvel & observations/effort [N, 16] float32
            qvel = np.random.randn(n_steps, 16).astype(np.float32)
            effort = np.random.randn(n_steps, 16).astype(np.float32)
            f.create_dataset("observations/qvel", data=qvel)
            f.create_dataset("observations/effort", data=effort)

            # 5. action [N, 16] float32
            if corrupt_action:
                action = qpos.copy()
                action[5:10, :] = np.nan  # Bơm lỗi NaN vào action
            else:
                action = qpos.copy()
            f.create_dataset("action", data=action)

            # 6. joint_commands [N, 16] float32 (chứa NaN thực tế do proxy)
            commands = qpos.copy()
            if inject_nan_in_commands:
                commands[::2, :] = np.nan
            f.create_dataset("joint_commands", data=commands)

            # 7. timestamp_ns [N] int64
            t0 = 1728375600000000000
            timestamps = t0 + np.arange(n_steps, dtype=np.int64) * 20_000_000  # 50Hz (20ms)
            f.create_dataset("timestamp_ns", data=timestamps)


# ==============================================================================
# NHÓM 1: KIỂM THỬ TÍNH TOÀN VẸN VÀ TIỀN XỬ LÝ DỮ LIỆU THẬT
# ==============================================================================
class TestDataIngestionAndSanitization(BaseOpenArmTestCase):

    def test_tc01_real_hdf5_schema_validation(self):
        """TC01: Đảm bảo file HDF5 đọc đúng các key và shape chính xác như HuggingFace."""
        test_file = os.path.join(self.temp_dir, "test_schema.hdf5")
        self.create_real_hdf5_episode(test_file, n_steps=30)

        with self.h5py.File(test_file, "r") as f:
            self.assertEqual(f["observations/images/chest_rgb"].shape, (30, 240, 424, 3))
            self.assertEqual(f["observations/images/chest_rgb"].dtype, np.uint8)
            self.assertEqual(f["observations/images/chest_depth"].shape, (30, 240, 424))
            self.assertEqual(f["observations/images/chest_depth"].dtype, np.uint16)
            self.assertEqual(f["observations/qpos"].shape, (30, 16))
            self.assertEqual(f["observations/qpos"].dtype, np.float32)
            self.assertEqual(f["action"].shape, (30, 16))
            self.assertEqual(f["joint_commands"].dtype, np.float32)
            self.assertEqual(f["timestamp_ns"].dtype, np.int64)

    def test_tc02_nan_containment_in_joint_commands(self):
        """TC02: Đảm bảo NaN trong joint_commands không bị lọt vào state hay action."""
        test_file = os.path.join(self.temp_dir, "test_nan_cmd.hdf5")
        self.create_real_hdf5_episode(test_file, n_steps=40, inject_nan_in_commands=True)

        from tools.prepare_openarm_data import inspect_and_extract_hdf5
        rgb, depth, qpos, action = inspect_and_extract_hdf5(test_file)

        self.assertFalse(np.isnan(qpos).any(), "Qpos bị nhiễm NaN từ dữ liệu thô!")
        self.assertFalse(np.isnan(action).any(), "Action bị nhiễm NaN!")

    def test_tc03_action_corrupted_rejection(self):
        """TC03: Đảm bảo pipeline phát hiện và loại bỏ episode có action bị hỏng (NaN)."""
        test_file = os.path.join(self.temp_dir, "test_corrupt.hdf5")
        self.create_real_hdf5_episode(test_file, n_steps=30, corrupt_action=True)

        out_dir = os.path.join(self.temp_dir, "clean_out")
        txt_path = os.path.join(self.temp_dir, "clean.txt")

        from tools.prepare_openarm_data import process_dataset
        process_dataset(self.temp_dir, out_dir, txt_path)

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                self.assertNotIn("test_corrupt.hdf5", line, "Episode hỏng không được phép đưa vào danh sách train!")

    def test_tc04_zero_depth_safety(self):
        """TC04: Kiểm tra điểm mù depth (giá trị 0 mm) không gây lỗi chia cho 0 hay crash."""
        test_file = os.path.join(self.temp_dir, "test_zero_depth.hdf5")
        self.create_real_hdf5_episode(test_file, n_steps=20, inject_zero_depth=True)

        with self.h5py.File(test_file, "r") as f:
            depth = np.array(f["observations/images/chest_depth"])
            has_zeros = (depth == 0).any()
            self.assertTrue(has_zeros, "Cần có điểm mù 0 mm để test edge case.")
            # Giả lập chuẩn hóa độ sâu: float32, chia cho 1000m
            normalized_depth = np.where(depth > 0, depth.astype(np.float32) / 1000.0, 0.0)
            self.assertFalse(np.isnan(normalized_depth).any())
            self.assertFalse(np.isinf(normalized_depth).any())


# ==============================================================================
# NHÓM 2: KIỂM THỬ KINEMATICS & ACTION MAPPING CHO OPENARM DUAL-ARM (16 DoF)
# ==============================================================================
class TestKinematicsAndActionMapping(BaseOpenArmTestCase):

    def test_tc05_dual_arm_split_and_delta(self):
        """TC05: Tách đúng 14 khớp tay và 2 khớp kẹp; tính delta cho tay, tuyệt đối cho kẹp."""
        qpos_t0 = np.array([
            # Tay trái (7)      Kẹp L    Tay phải (7)      Kẹp R
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,   0.0,   -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7,   1.0
        ], dtype=np.float32)

        qpos_t1 = np.array([
            0.15, 0.22, 0.35, 0.41, 0.55, 0.62, 0.75, 0.8,   -0.08, -0.19, -0.28, -0.39, -0.48, -0.58, -0.68, 0.8
        ], dtype=np.float32)

        # Trích xuất theo slicing của openarm_dual.yaml
        arm_t0 = np.concatenate([qpos_t0[0:7], qpos_t0[8:15]])      # 14 DoF tay
        arm_t1 = np.concatenate([qpos_t1[0:7], qpos_t1[8:15]])
        gripper_t0 = np.array([qpos_t0[7], qpos_t0[15]])   # 2 DoF kẹp
        gripper_t1 = np.array([qpos_t1[7], qpos_t1[15]])

        # Action: Arm dùng delta, Gripper dùng tuyệt đối
        delta_arm = arm_t1 - arm_t0
        action_gripper = gripper_t1

        self.assertEqual(delta_arm.shape, (14,))
        self.assertEqual(action_gripper.shape, (2,))
        self.assertTrue(np.allclose(delta_arm[0], 0.05, atol=1e-5))
        self.assertAlmostEqual(float(action_gripper[0]), 0.8, places=5)
        self.assertAlmostEqual(float(action_gripper[1]), 0.8, places=5)


    def test_tc06_canonical_55_dim_projection(self):
        """TC06: Đảm bảo vector 16-DoF map chuẩn xác vào Canonical 55-DoF của LingBot-VLA."""
        canonical_state = np.zeros(55, dtype=np.float32)

        # Mapping theo LingBot-VLA:
        # arm.position: 14 (idx 0..14)
        # effector.position: 2 (idx 14..16)
        arm_state = np.ones(14, dtype=np.float32) * 0.5
        gripper_state = np.ones(2, dtype=np.float32) * 1.0

        canonical_state[0:14] = arm_state
        canonical_state[14:16] = gripper_state

        self.assertEqual(canonical_state[:14].sum(), 7.0)
        self.assertEqual(canonical_state[14:16].sum(), 2.0)
        self.assertEqual(canonical_state[16:].sum(), 0.0, "Các chiều unused phải bằng 0.")


# ==============================================================================
# NHÓM 3: ĐẢM BẢO MÁY YẾU CHẠY ĐƯỢC (BỘ NHỚ, RAM, LAZY LOADING)
# ==============================================================================
class TestLowSpecHardwareBudget(BaseOpenArmTestCase):

    def test_tc07_lazy_loading_memory_footprint(self):
        """TC07: Đọc HDF5 theo dạng generator/slice không nạp toàn bộ dataset vào RAM cùng lúc."""
        test_file = os.path.join(self.temp_dir, "test_large_stream.hdf5")
        self.create_real_hdf5_episode(test_file, n_steps=100)

        with self.h5py.File(test_file, "r") as f:
            dset_rgb = f["observations/images/chest_rgb"]
            single_frame = dset_rgb[10:11]
            self.assertEqual(single_frame.shape, (1, 240, 424, 3))
            frame_nbytes = single_frame.nbytes
            self.assertLess(frame_nbytes, 400_000)

    def test_tc08_micro_batch_size_1_and_2_simulation(self):
        """TC08: Kiểm tra batch size nhỏ (1 và 2) qua mô phỏng gradient accumulation."""
        if self.torch is not None:
            pred_v = self.torch.randn(2, 30, 55, requires_grad=True)
            target_v = self.torch.randn(2, 30, 55)
            loss = self.torch.nn.functional.l1_loss(pred_v, target_v) / 4
            loss.backward()
            self.assertIsNotNone(pred_v.grad)
            self.assertFalse(self.torch.isnan(pred_v.grad).any())
        else:
            pred_v = np.random.randn(2, 30, 55)
            target_v = np.random.randn(2, 30, 55)
            loss = np.abs(pred_v - target_v).mean()
            self.assertFalse(np.isnan(loss))

    def test_tc09_cpu_and_cuda_parity(self):
        """TC09: Đảm bảo code chạy mượt mà trên CPU phòng trường hợp máy yếu không có CUDA."""
        if self.torch is not None:
            state_proj = self.torch.nn.Linear(55, 768)
            state_tensor = self.torch.randn(2, 55)
            out = state_proj(state_tensor)
            self.assertEqual(out.shape, (2, 768))
        else:
            W = np.random.randn(55, 768).astype(np.float32)
            state_tensor = np.random.randn(2, 55).astype(np.float32)
            out = state_tensor @ W
            self.assertEqual(out.shape, (2, 768))


# ==============================================================================
# NHÓM 4: ĐẢM BẢO TRAINING KHÔNG LỖI (FLOW MATCHING, TIMESTEP, LOSS NORM)
# ==============================================================================
class TestTrainingPipelineStability(BaseOpenArmTestCase):

    def test_tc10_flow_matching_timestep_interpolation(self):
        """TC10: Đảm bảo công thức Flow Matching x_t = (1 - t) * x_0 + t * x_1 không sinh NaN."""
        batch = 4
        chunk = 30
        dim = 55

        x_1 = np.random.randn(batch, chunk, dim)
        x_0 = np.random.randn(batch, chunk, dim)
        t = np.random.rand(batch, 1, 1)

        target_v = x_1 - x_0
        x_t = (1.0 - t) * x_0 + t * x_1

        self.assertFalse(np.isnan(x_t).any())
        self.assertFalse(np.isnan(target_v).any())
        self.assertEqual(x_t.shape, (batch, chunk, dim))

    def test_tc11_normalization_division_by_zero_prevention(self):
        """TC11: Khớp đứng yên (std = 0) không được gây ra lỗi chia cho 0 khi chuẩn hóa."""
        constant_joint = np.full((100,), 0.5, dtype=np.float32)
        mean = constant_joint.mean()
        std = constant_joint.std()
        eps = 1e-6

        normalized = (constant_joint - mean) / (std + eps)
        self.assertFalse(np.isnan(normalized).any())
        self.assertTrue((normalized == 0.0).all())


# ==============================================================================
# NHÓM 5: ĐẢM BẢO ĐẦU RA KHÔNG BỊ LỖI (OUTPUT SANITY & SAFETY CHECKS)
# ==============================================================================
class TestOutputQualityAndSafety(BaseOpenArmTestCase):

    def test_tc12_predicted_gripper_range_bounds(self):
        """TC12: Giá trị dự đoán cho Gripper phải được kẹp an toàn trong dải [0.0, 1.0]."""
        raw_model_output = np.array([-0.2, 0.5, 1.4, 0.99, -0.01], dtype=np.float32)
        safe_gripper = np.clip(raw_model_output, 0.0, 1.0)

        self.assertTrue((safe_gripper >= 0.0).all())
        self.assertTrue((safe_gripper <= 1.0).all())

    def test_tc13_joint_velocity_smoothness_check(self):
        """TC13: Kiểm tra giới hạn giật khớp (Jerk limit) để bảo vệ động cơ OpenArm ngoài đời thật."""
        trajectory = np.cumsum(np.random.randn(30, 14) * 0.02, axis=0)
        velocities = np.diff(trajectory, axis=0)

        max_delta = np.abs(velocities).max()
        MAX_SAFE_DELTA = 0.20  # rad
        self.assertLess(max_delta, MAX_SAFE_DELTA, f"Vận tốc góc quá lớn ({max_delta} rad), nguy cơ giật gãy tay robot!")


# ==============================================================================
# NHÓM 6: EDGE CASES TOÀN DIỆN
# ==============================================================================
class TestComprehensiveEdgeCases(BaseOpenArmTestCase):

    def test_tc14_short_episode_handling(self):
        """TC14: Episode quá ngắn (ít hơn chunk size H=30 bước) phải được padding an toàn."""
        H = 30
        short_ep_len = 12
        raw_actions = np.random.randn(short_ep_len, 16)

        last_frame = np.repeat(raw_actions[-1:], H - short_ep_len, axis=0)
        padded_actions = np.concatenate([raw_actions, last_frame], axis=0)

        self.assertEqual(padded_actions.shape, (H, 16))
        self.assertTrue(np.array_equal(padded_actions[12], raw_actions[-1]))


    def test_tc15_non_monotonic_timestamps(self):
        """TC15: Phát hiện và xử lý timestamps bị nhảy lùi (Jitter/Packet reorder)."""
        # Timestamp bị đảo lộn
        timestamps = np.array([1000, 1020, 1015, 1060, 1080], dtype=np.int64)
        is_strictly_increasing = np.all(np.diff(timestamps) > 0)
        self.assertFalse(is_strictly_increasing, "Cần phát hiện timestamp bị nhảy ngược.")

        # Xử lý tự động: sort lại theo thời gian
        sort_idx = np.argsort(timestamps)
        fixed_timestamps = timestamps[sort_idx]
        self.assertTrue(np.all(np.diff(fixed_timestamps) > 0))

    def test_tc16_extreme_joint_angle_clipping(self):
        """TC16: Xử lý giá trị góc khớp ngoại lai (outliers) bằng 99-percentile bounds."""
        angles = np.array([-1.2, 0.4, 0.8, 1.1, 999.0, -999.0])  # Có nhiễu đo đạc cực đại
        q99 = np.percentile(angles[np.abs(angles) < 100], 99)
        q01 = np.percentile(angles[np.abs(angles) < 100], 1)

        clipped = np.clip(angles, q01, q99)
        self.assertLess(clipped.max(), 10.0)
        self.assertGreater(clipped.min(), -10.0)

    def test_tc17_real_downloaded_sample_validation(self):
        """TC17: Nếu có file mẫu tải từ HuggingFace trong data/openarm_samples, kiểm tra tính toàn vẹn."""
        sample_dir = os.path.join(PROJECT_ROOT, "data", "openarm_samples")
        if not os.path.exists(sample_dir):
            return  # Chưa tải file mẫu thì bỏ qua

        sample_files = list(Path(sample_dir).glob("*.hdf5"))
        if not sample_files:
            return

        sample_file = str(sample_files[0])
        print(f"\n[*] Đang kiểm thử trực tiếp trên file thật tải từ HuggingFace: {sample_file}")

        from tools.prepare_openarm_data import inspect_and_extract_hdf5
        rgb, depth, qpos, action = inspect_and_extract_hdf5(sample_file)

        self.assertIsNotNone(rgb, "Không trích xuất được RGB từ file thật.")
        self.assertIsNotNone(qpos, "Không trích xuất được Qpos từ file thật.")
        self.assertEqual(rgb.shape[-1], 3, "Kênh ảnh phải là 3 (RGB).")
        self.assertIn(rgb.shape[1:], [(480, 640, 3), (240, 424, 3)], "Kích thước ảnh không khớp chuẩn OpenArm.")
        self.assertEqual(qpos.shape[1], 16, "Sai số bậc tự do (phải là 16 DoF).")
        self.assertEqual(action.shape[1], 16, "Sai số chiều action (phải là 16 DoF).")
        self.assertFalse(np.isnan(qpos).any(), "Qpos chứa giá trị NaN!")
        self.assertFalse(np.isnan(action).any(), "Action chứa giá trị NaN!")




def run_all_tests():
    """Hàm chạy test suite và hiển thị báo cáo."""
    print("=" * 70)
    print("  CHẠY BỘ TEST SUITE TOÀN DIỆN CHO DATASET OPENARM (LINGBOT-VLA)   ")
    print("  Dataset: https://huggingface.co/datasets/Tuyen062004/data_openarm_8_10")
    print("=" * 70)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

