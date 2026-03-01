"""
Measure segmentation inference efficiency on the validation set.
Reports per-case: image size, running time, max GPU memory, total GPU (area under GPU-time curve).
"""
import os
import sys
import json
import time
import threading

import numpy as np
import torch
import nibabel as nib

BASE = os.path.dirname(os.path.abspath(__file__))

os.environ["nnUNet_raw"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

RESULT_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_results", "Dataset001_Pancreas",
    "nnUNetTrainerMultiTask__nnUNetPlans__3d_fullres",
)
IMAGES_TR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_raw", "Dataset001_Pancreas", "imagesTr",
)
SPLITS_FILE = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_preprocessed", "Dataset001_Pancreas",
    "splits_final.json",
)


class GPUMemoryMonitor:
    """Sample GPU memory in a background thread to compute area-under-curve."""
    def __init__(self, interval_s=0.05):
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            t = time.perf_counter() - t0
            self.samples.append((t, mem_mb))
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def max_gpu_mb(self):
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def total_gpu_mb(self):
        """Area under GPU memory-time curve (MB*s), reported in MB."""
        if len(self.samples) < 2:
            return 0.0
        area = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_mem = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            area += dt * avg_mem
        return area


def main():
    device = torch.device("cuda")
    splits = json.load(open(SPLITS_FILE))
    val_cases = sorted(splits[0]["val"])

    print(f"Validation cases: {len(val_cases)}")
    print(f"Setting up nnUNetPredictor...")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=RESULT_DIR,
        use_folds=(0,),
        checkpoint_name="checkpoint_latest.pth",
    )

    net = predictor.network
    if hasattr(net, "_do_cls"):
        net._do_cls = False
    elif hasattr(net, "module") and hasattr(net.module, "_do_cls"):
        net.module._do_cls = False

    tmp_out = os.path.join(BASE, "_efficiency_tmp")
    os.makedirs(tmp_out, exist_ok=True)

    monitor = GPUMemoryMonitor(interval_s=0.05)
    results = []

    # Warm-up run (first case is always slower due to CUDA kernel caching)
    warmup_img = os.path.join(IMAGES_TR, f"{val_cases[0]}_0000.nii.gz")
    if os.path.isfile(warmup_img):
        print("Warm-up run...")
        predictor.predict_from_files(
            [[warmup_img]],
            [os.path.join(tmp_out, f"{val_cases[0]}.nii.gz")],
        )
        torch.cuda.synchronize()

    print(f"\n{'Case ID':<18} {'Image Size':<22} {'Time (s)':>10} {'Max GPU (MB)':>14} {'Total GPU (MB)':>16}")
    print("-" * 84)

    for case_id in val_cases:
        img_path = os.path.join(IMAGES_TR, f"{case_id}_0000.nii.gz")
        if not os.path.isfile(img_path):
            print(f"  SKIP: {case_id} (image not found)")
            continue

        img_nii = nib.load(img_path)
        img_shape = img_nii.shape

        out_path = os.path.join(tmp_out, f"{case_id}.nii.gz")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor.start()
        t_start = time.perf_counter()

        predictor.predict_from_files(
            [[img_path]],
            [out_path],
        )

        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start
        monitor.stop()

        max_gpu = monitor.max_gpu_mb()
        total_gpu = monitor.total_gpu_mb()

        shape_str = f"({img_shape[0]}, {img_shape[1]}, {img_shape[2]})"
        results.append({
            "case_id": case_id,
            "shape": shape_str,
            "time_s": t_elapsed,
            "max_gpu_mb": max_gpu,
            "total_gpu_mb": total_gpu,
        })

        print(f"{case_id:<18} {shape_str:<22} {t_elapsed:>10.2f} {max_gpu:>14.1f} {total_gpu:>16.1f}")

    # Summary
    times = [r["time_s"] for r in results]
    maxgpus = [r["max_gpu_mb"] for r in results]
    totgpus = [r["total_gpu_mb"] for r in results]

    print("-" * 84)
    print(f"{'MEAN':<18} {'':<22} {np.mean(times):>10.2f} {np.mean(maxgpus):>14.1f} {np.mean(totgpus):>16.1f}")
    print(f"{'STD':<18} {'':<22} {np.std(times):>10.2f} {np.std(maxgpus):>14.1f} {np.std(totgpus):>16.1f}")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_out, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
