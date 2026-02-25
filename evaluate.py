"""
evaluate.py — Validation evaluation for the multitask nnUNet model.

Computes:
  Segmentation (per label group, averaged over cases):
    - DSC, NSD (tau=1mm), F_beta (beta=2)
  Classification (over all validation samples):
    - Macro F1, Balanced Accuracy, per-class LR+
    - ECE_KDE, CWCE, RBS (calibration; requires softmax probabilities)
"""

import os
import json
import pickle

import blosc2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from MetricsReloaded.metrics.calibration_measures import (
    CalibrationMeasures as CM,
)
from MetricsReloaded.metrics.pairwise_measures import (
    BinaryPairwiseMeasures as BPM,
    MultiClassPairwiseMeasures as MCPM,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultitask import (
    CrossAttentionPooling,
    MultiTaskWrapper,
    nnUNetTrainerMultiTask,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_results", "Dataset001_Pancreas",
    "nnUNetTrainerMultiTask__nnUNetPlans__3d_fullres",
)
FOLD_DIR = os.path.join(RESULT_DIR, "fold_0")
CHECKPOINT = os.path.join(FOLD_DIR, "checkpoint_latest.pth")
SEG_PRED_DIR = os.path.join(FOLD_DIR, "validation")

LABEL_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_raw", "Dataset001_Pancreas", "labelsTr",
)
PREPROCESSED_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_preprocessed", "Dataset001_Pancreas",
    "nnUNetPlans_3d_fullres",
)
SPLITS_FILE = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_preprocessed", "Dataset001_Pancreas",
    "splits_final.json",
)
PLANS_FILE = os.path.join(RESULT_DIR, "plans.json")

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def parse_subtype(case_id: str) -> int:
    """quiz_{subtype}_{caseid} -> int(subtype)"""
    return int(case_id.split("_")[1])


def center_crop_or_pad(data: np.ndarray, target_shape: list) -> np.ndarray:
    """Center-crop or zero-pad *data* (C, D, H, W) to *target_shape* (D, H, W)."""
    C = data.shape[0]
    result = np.zeros((C, *target_shape), dtype=data.dtype)
    src_slices = [slice(None)]
    tgt_slices = [slice(None)]
    for d in range(3):
        s, t = data.shape[d + 1], target_shape[d]
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            tgt_slices.append(slice(None))
        else:
            start = (t - s) // 2
            src_slices.append(slice(None))
            tgt_slices.append(slice(start, start + s))
    result[tuple(tgt_slices)] = data[tuple(src_slices)]
    return result


def build_model(plans: dict, checkpoint_path: str, device: torch.device):
    """Reconstruct the MultiTaskWrapper and load trained weights."""
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]

    base_network = get_network_from_plans(
        arch["network_class_name"],
        arch["arch_kwargs"],
        arch["_kw_requires_import"],
        1,   # num_input_channels (CT)
        3,   # num_output_channels (bg + pancreas + lesion)
        allow_init=True,
        deep_supervision=False,
    )
    model = MultiTaskWrapper(base_network, num_classes_cls=3)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["network_weights"])
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plans = json.load(open(PLANS_FILE))
    splits = json.load(open(SPLITS_FILE))
    val_cases = sorted(splits[0]["val"])
    patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

    print(f"Device: {device}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Patch size: {patch_size}")

    # ── A. Classification inference ──────────────────────────
    print("\n[A] Loading model for classification inference …")
    model = build_model(plans, CHECKPOINT, device)
    model._do_cls = True

    cls_preds_list, cls_probs_list, cls_targets_list = [], [], []

    for i, case_id in enumerate(val_cases):
        data = blosc2.open(
            os.path.join(PREPROCESSED_DIR, f"{case_id}.b2nd")
        )[:]
        data = center_crop_or_pad(data, patch_size)
        x = torch.from_numpy(data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, cls_logits = model(x)
        probs = F.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        target = parse_subtype(case_id)

        cls_preds_list.append(pred)
        cls_probs_list.append(probs)
        cls_targets_list.append(target)
        print(f"  [{i+1:>2}/{len(val_cases)}] {case_id}  "
              f"pred={pred}  target={target}  probs={probs.round(3)}")

    cls_preds = np.array(cls_preds_list)
    cls_probs = np.array(cls_probs_list)       # (N, 3)
    cls_targets = np.array(cls_targets_list)

    del model
    torch.cuda.empty_cache()

    # ── B. Segmentation metrics ──────────────────────────────
    print("\n[B] Computing segmentation metrics …")
    seg_results = {"whole_pancreas": [], "lesion": []}

    for i, case_id in enumerate(val_cases):
        pred_path = os.path.join(SEG_PRED_DIR, f"{case_id}.nii.gz")
        ref_path = os.path.join(LABEL_DIR, f"{case_id}.nii.gz")

        if not os.path.isfile(pred_path):
            print(f"  WARNING: prediction missing for {case_id}, skipping")
            continue

        pred_nii = nib.load(pred_path)
        ref_nii = nib.load(ref_path)
        pred_data = np.asarray(pred_nii.dataobj).astype(np.uint8)
        ref_data = np.asarray(ref_nii.dataobj).astype(np.uint8)
        pixdim = list(pred_nii.header.get_zooms()[:3])

        # Whole pancreas (label > 0)
        bpm_wp = BPM(
            (pred_data > 0).astype(np.uint8),
            (ref_data > 0).astype(np.uint8),
            measures=["dsc", "nsd", "fbeta"],
            pixdim=pixdim,
            dict_args={"nsd": 1.0, "beta": 2.0},
        )
        seg_results["whole_pancreas"].append(bpm_wp.to_dict_meas())

        # Lesion only (label == 2)
        bpm_les = BPM(
            (pred_data == 2).astype(np.uint8),
            (ref_data == 2).astype(np.uint8),
            measures=["dsc", "nsd", "fbeta"],
            pixdim=pixdim,
            dict_args={"nsd": 1.0, "beta": 2.0},
        )
        seg_results["lesion"].append(bpm_les.to_dict_meas())
        print(f"  [{i+1:>2}/{len(val_cases)}] {case_id}  "
              f"WP-DSC={seg_results['whole_pancreas'][-1]['dsc']:.4f}  "
              f"Les-DSC={seg_results['lesion'][-1]['dsc']:.4f}")

    # ── C. Classification metrics ────────────────────────────
    print("\n[C] Computing classification metrics …")

    # Macro F1 via per-class BPM (beta=1)
    per_class_f1 = []
    for c in [0, 1, 2]:
        bpm_c = BPM(
            (cls_preds == c).astype(int),
            (cls_targets == c).astype(int),
            dict_args={"beta": 1},
        )
        per_class_f1.append(bpm_c.fbeta())
    macro_f1 = float(np.nanmean(per_class_f1))

    # Balanced accuracy
    mcpm = MCPM(cls_preds, cls_targets, list_values=[0, 1, 2], measures=["ba"])
    ba = mcpm.balanced_accuracy()

    # Per-class LR+
    lr_plus = {}
    for c in [0, 1, 2]:
        bpm_c = BPM(
            (cls_preds == c).astype(int),
            (cls_targets == c).astype(int),
        )
        lr_plus[c] = bpm_c.positive_likelihood_ratio()

    # Calibration
    cm = CM(
        pred_proba=cls_probs,
        ref=cls_targets,
        measures=["ece_kde", "cwece", "rbs"],
    )
    cal = cm.to_dict_meas()

    # ── Print results ────────────────────────────────────────
    print("\n" + "=" * 62)
    print(" SEGMENTATION METRICS  (mean over validation cases)")
    print("=" * 62)
    for group in ["whole_pancreas", "lesion"]:
        if not seg_results[group]:
            print(f"\n  {group.upper()}: no predictions found")
            continue
        vals = {m: np.nanmean([r[m] for r in seg_results[group]])
                for m in ["dsc", "nsd", "fbeta"]}
        print(f"\n  {group.upper()}:")
        print(f"    DSC          {vals['dsc']:.4f}")
        print(f"    NSD (tau=1)  {vals['nsd']:.4f}")
        print(f"    F_beta (β=2) {vals['fbeta']:.4f}")

    print("\n" + "=" * 62)
    print(" CLASSIFICATION METRICS")
    print("=" * 62)
    print(f"  Macro F1              {macro_f1:.4f}")
    print(f"  Balanced Accuracy     {ba:.4f}")
    for c in [0, 1, 2]:
        print(f"  LR+ class {c}           {lr_plus[c]:.4f}")
    print(f"  ECE_KDE               {cal.get('ece_kde', float('nan')):.4f}")
    print(f"  CWCE                  {cal.get('cwece', float('nan')):.4f}")
    print(f"  RBS                   {cal.get('rbs', float('nan')):.4f}")

    # ── Per-case segmentation table ──────────────────────────
    print("\n" + "=" * 62)
    print(" PER-CASE SEGMENTATION (DSC)")
    print("=" * 62)
    print(f"  {'Case':<16} {'WP-DSC':>8} {'Les-DSC':>8}")
    print("  " + "-" * 34)
    for j, case_id in enumerate(val_cases):
        if j < len(seg_results["whole_pancreas"]):
            wp = seg_results["whole_pancreas"][j]["dsc"]
            les = seg_results["lesion"][j]["dsc"]
            print(f"  {case_id:<16} {wp:>8.4f} {les:>8.4f}")


if __name__ == "__main__":
    main()
