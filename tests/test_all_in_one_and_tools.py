"""
BỘ TEST KIỂM THỬ SCRIPT ALL-IN-ONE VÀ CÁC TOOLS THÀNH PHẦN (ƯU TIÊN LINUX/BASH)

Mục tiêu kiểm thử:
1. Đảm bảo bash script chạy được trên Linux (cú pháp bash, gọi thực thi, cờ tham số).
2. Đảm bảo script master train_all_in_one.py xử lý đối số, luồng tương tác prompt và logic an toàn.
3. Đảm bảo toàn bộ các công cụ (tools) thành phần chạy ổn định:
   - tools/check_environment.py
   - tools/prepare_openarm_data.py
   - scripts/download_hf_model.py
   - tools/download_sample_data.py
4. Đảm bảo chuỗi tích hợp end-to-end các tool hoạt động trơn tru.
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess
from pathlib import Path

# Đảm bảo UTF-8 cho stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BaseToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import h5py
            cls.h5py = h5py
        except ImportError:
            raise unittest.SkipTest("Cần thư viện 'h5py' để chạy test suite.")

        try:
            import numpy
            cls.np = numpy
        except ImportError:
            raise unittest.SkipTest("Cần thư viện 'numpy' để chạy test suite.")

        cls.temp_dir = tempfile.mkdtemp(prefix="test_tools_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_dummy_hdf5(self, path: str, n_frames: int = 20, has_nan: bool = False):
        """Tạo file HDF5 thật để test công cụ tiền xử lý."""
        np = self.np
        with self.h5py.File(path, "w") as f:
            rgb = np.random.randint(0, 256, (n_frames, 240, 424, 3), dtype=np.uint8)
            f.create_dataset("observations/images/chest_rgb", data=rgb)
            qpos = np.random.uniform(-1.0, 1.0, (n_frames, 16)).astype(np.float32)
            if has_nan:
                qpos[5, :] = np.nan
            f.create_dataset("observations/qpos", data=qpos)
            action = qpos.copy()
            f.create_dataset("action", data=action)


# ==============================================================================
# NHÓM 1: KIỂM THỬ SCRIPT BASH (LINUX COMPATIBILITY)
# ==============================================================================
class TestBashScriptsLinux(BaseToolsTest):

    def test_tc_bash_01_syntax_validation(self):
        """TC_BASH_01: Kiểm tra cú pháp bash của tất cả script shell bằng bash -n."""
        scripts_to_check = [
            "scripts/train_all_in_one.sh",
            "scripts/train_remote_openarm.sh",
        ]

        has_bash = shutil.which("bash") is not None
        if not has_bash:
            self.skipTest("Hệ thống không có lệnh 'bash' (bỏ qua trên môi trường non-bash).")

        for rel_script in scripts_to_check:
            script_path = PROJECT_ROOT / rel_script
            self.assertTrue(script_path.exists(), f"Không tìm thấy file: {script_path}")
            # Dùng relative POSIX path để tương thích cả Linux và Git Bash trên Windows
            res = subprocess.run(
                ["bash", "-n", rel_script],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(res.returncode, 0, f"Lỗi cú pháp bash trong {rel_script}:\n{res.stderr}")

    def test_tc_bash_02_train_all_in_one_help_flag(self):
        """TC_BASH_02: Chạy bash scripts/train_all_in_one.sh --help đảm bảo Linux chạy được."""
        has_bash = shutil.which("bash") is not None
        if not has_bash:
            self.skipTest("Hệ thống không có lệnh 'bash'.")

        res = subprocess.run(
            ["bash", "scripts/train_all_in_one.sh", "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(res.returncode, 0, f"Lỗi khi chạy bash script --help:\n{res.stderr}")
        self.assertIn("train_all_in_one.py", res.stdout)
        self.assertIn("--data-dir", res.stdout)


# ==============================================================================
# NHÓM 2: KIỂM THỬ SCRIPT MASTER TRAIN_ALL_IN_ONE.PY
# ==============================================================================
class TestMasterScriptLogic(BaseToolsTest):

    def test_tc_master_01_cli_arguments_parsing(self):
        """TC_MASTER_01: Kiểm tra parser tham số --data-dir, --model-dir, --use-mirror, -y."""
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_all_in_one.py"), "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0)
        self.assertIn("--data-dir", res.stdout)
        self.assertIn("--model-dir", res.stdout)
        self.assertIn("--use-mirror", res.stdout)
        self.assertIn("-y", res.stdout)

    def test_tc_master_02_interactive_prompt_simulation(self):
        """TC_MASTER_02: Đảm bảo hàm ask_user_confirm nhận diện chính xác 'y' và 'n'."""
        from scripts.train_all_in_one import ask_user_confirm
        import io

        orig_stdin = sys.stdin
        try:
            # Giả lập input 'y'
            sys.stdin = io.StringIO("y\n")
            self.assertTrue(ask_user_confirm("Test question?"))

            # Giả lập input 'n'
            sys.stdin = io.StringIO("n\n")
            self.assertFalse(ask_user_confirm("Test question?"))
        finally:
            sys.stdin = orig_stdin


# ==============================================================================
# NHÓM 3: KIỂM THỬ CÔNG CỤ CHECK_ENVIRONMENT.PY
# ==============================================================================
class TestToolCheckEnvironment(BaseToolsTest):

    def test_tc_tool_check_env_execution(self):
        """TC_TOOL_01: Chạy python tools/check_environment.py kiểm tra quét package không crash."""
        cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "check_environment.py")]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertIn(res.returncode, [0, 1], f"check_environment.py văng lỗi bất thường:\n{res.stderr}")
        self.assertIn("KIỂM TRA MÔI TRƯỜNG & THƯ VIỆN", res.stdout)
        self.assertIn("Python:", res.stdout)


# ==============================================================================
# NHÓM 4: KIỂM THỬ CÔNG CỤ PREPARE_OPENARM_DATA.PY
# ==============================================================================
class TestToolPrepareOpenArmData(BaseToolsTest):

    def test_tc_tool_prepare_data_e2e(self):
        """TC_TOOL_02: Chạy tools/prepare_openarm_data.py trên file HDF5 thật, lọc bỏ NaN và ghi danh sách."""
        raw_dir = os.path.join(self.temp_dir, "raw_episodes")
        os.makedirs(raw_dir, exist_ok=True)
        out_dir = os.path.join(self.temp_dir, "processed_episodes")
        txt_path = os.path.join(self.temp_dir, "test_train.txt")

        # 1. Tạo 1 file hợp lệ và 1 file chứa NaN
        valid_file = os.path.join(raw_dir, "valid_ep.hdf5")
        nan_file = os.path.join(raw_dir, "nan_ep.hdf5")
        self.create_dummy_hdf5(valid_file, n_frames=20, has_nan=False)
        self.create_dummy_hdf5(nan_file, n_frames=20, has_nan=True)

        # 2. Chạy tool
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "prepare_openarm_data.py"),
            "--input_dir", raw_dir,
            "--output_dir", out_dir,
            "--txt_path", txt_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0, f"prepare_openarm_data.py bị lỗi:\n{res.stderr}")

        # 3. Xác nhận output
        self.assertTrue(os.path.exists(txt_path), "File txt danh sách không được tạo!")
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("openarm", content)
        self.assertIn("episode_000000.h5", content)
        processed_files = list(Path(out_dir).glob("*.h5"))
        self.assertEqual(len(processed_files), 1, "Số file sau khi lọc không khớp (file NaN phải bị bỏ).")


# ==============================================================================
# NHÓM 5: KIỂM THỬ CÔNG CỤ DOWNLOAD_HF_MODEL.PY (LOGIC FILTER & MIRROR)
# ==============================================================================
class TestToolDownloadHfModel(BaseToolsTest):

    def test_tc_tool_download_model_filter_logic(self):
        """TC_TOOL_03: Kiểm tra logic cắt giảm 9GB weights của Qwen và 4GB teacher của LingBot."""
        import inspect
        from scripts.download_hf_model import download_model

        sig = inspect.signature(download_model)
        self.assertIn("repo_id", sig.parameters)
        self.assertIn("local_dir", sig.parameters)
        self.assertIn("use_mirror", sig.parameters)

        script_code = (PROJECT_ROOT / "scripts" / "download_hf_model.py").read_text(encoding="utf-8")
        self.assertIn("*.safetensors", script_code, "Thiếu logic lọc bỏ safetensors của Qwen!")
        self.assertIn("depth/*", script_code, "Thiếu logic bỏ qua depth teacher của LingBot!")
        self.assertIn("hf-mirror.com", script_code, "Thiếu cấu hình mirror hf-mirror.com!")


# ==============================================================================
# NHÓM 6: KIỂM THỬ TÍCH HỢP CHUỖI CÁC BƯỚC (DRY-RUN INTEGRATION)
# ==============================================================================
class TestIntegrationDryRun(BaseToolsTest):

    def test_tc_integ_01_step_pipeline_validation(self):
        """TC_INTEG_01: Đảm bảo các hàm import và cấu trúc hằng số của master script toàn vẹn."""
        from scripts.train_all_in_one import REQUIRED_PACKAGES, MODELS_INFO
        self.assertIn("torch", REQUIRED_PACKAGES)
        self.assertIn("h5py", REQUIRED_PACKAGES)
        self.assertIn("lingbot-vla-v2-6b", MODELS_INFO)
        self.assertIn("Qwen3-VL-4B-Instruct", MODELS_INFO)
        self.assertGreater(MODELS_INFO["lingbot-vla-v2-6b"]["size_gb"], 20.0)


def run_tools_tests():
    print("=" * 70)
    print("  CHẠY BỘ TEST KIỂM THỬ ALL-IN-ONE VÀ CÁC TOOLS (LINUX & WINDOWS)  ")
    print("=" * 70)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    ok = run_tools_tests()
    sys.exit(0 if ok else 1)
