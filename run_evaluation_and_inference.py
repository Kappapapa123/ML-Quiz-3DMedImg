"""
run_evaluation_and_inference.py
================================
Phase 1: Generate validation results CSV (seg from summary.json + classification inference)
Phase 2: Copy best/worst masks for technical report
Phase 3: Run inference on test dataset -> segmentation masks + subtype_results.csv
"""

import os
import sys
import json
import csv
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import blosc2
from scipy.ndimage import zoom

BASE = os.path.dirname(os.path.abspath(__file__))

os.environ["nnUNet_raw"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(BASE, "nnUNet_storage", "nnUNet_results")

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultitask import (
    CrossAttentionPooling, MultiTaskWrapper, nnUNetTrainerMultiTask,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

RESULT_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_results", "Dataset001_Pancreas",
    "nnUNetTrainerMultiTask__nnUNetPlans__3d_fullres",
)
FOLD_DIR = os.path.join(RESULT_DIR, "fold_0")
CHECKPOINT = os.path.join(FOLD_DIR, "checkpoint_latest.pth")
VAL_PRED_DIR = os.path.join(FOLD_DIR, "validation")
SUMMARY_JSON = os.path.join(VAL_PRED_DIR, "summary.json")

LABEL_DIR = os.path.join(BASE, "nnUNet_storage", "nnUNet_raw", "Dataset001_Pancreas", "labelsTr")
PREPROCESSED_DIR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_preprocessed", "Dataset001_Pancreas", "nnUNetPlans_3d_fullres",
)
SPLITS_FILE = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_preprocessed", "Dataset001_Pancreas", "splits_final.json",
)
PLANS_FILE = os.path.join(RESULT_DIR, "plans.json")
TEST_IMAGES_DIR = os.path.join(BASE, "nnUNet_storage", "nnUNet_raw", "Dataset001_Pancreas", "imagesTs")


def parse_subtype(case_id: str) -> int:
    return int(case_id.split("_")[1])


def center_crop_or_pad(data: np.ndarray, target_shape: list) -> np.ndarray:
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
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]
    base_network = get_network_from_plans(
        arch["network_class_name"],
        arch["arch_kwargs"],
        arch["_kw_requires_import"],
        1, 3,
        allow_init=True,
        deep_supervision=False,
    )
    model = MultiTaskWrapper(base_network, num_classes_cls=3)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["network_weights"])
    model.to(device)
    model.eval()
    return model


def preprocess_raw_image(nifti_path, target_spacing, intensity_props, patch_size):
    """Manually preprocess a raw NIfTI CT image for classification inference."""
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj).astype(np.float32)
    spacing = np.array(img.header.get_zooms()[:3])

    zoom_factors = spacing / np.array(target_spacing)
    if not np.allclose(zoom_factors, 1.0, atol=0.01):
        data = zoom(data, zoom_factors, order=3)

    lower = intensity_props["percentile_00_5"]
    upper = intensity_props["percentile_99_5"]
    mean = intensity_props["mean"]
    std = intensity_props["std"]
    data = np.clip(data, lower, upper)
    data = (data - mean) / max(std, 1e-8)

    data = data[np.newaxis]  # (1, D, H, W)
    data = center_crop_or_pad(data, patch_size)
    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plans = json.load(open(PLANS_FILE))
    splits = json.load(open(SPLITS_FILE))
    val_cases = sorted(splits[0]["val"])
    patch_size = plans["configurations"]["3d_fullres"]["patch_size"]
    target_spacing = plans["configurations"]["3d_fullres"]["spacing"]
    intensity_props = plans["foreground_intensity_properties_per_channel"]["0"]

    summary = json.load(open(SUMMARY_JSON))

    print(f"Device: {device}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Patch size: {patch_size}")

    # ================================================================
    # PHASE 1: Validation results CSV
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Generating validation results CSV")
    print("=" * 60)

    seg_metrics = {}
    for entry in summary["metric_per_case"]:
        pred_file = entry["prediction_file"]
        case_id = os.path.basename(pred_file).replace(".nii.gz", "")
        seg_metrics[case_id] = {
            "wp_dice": entry["metrics"]["1"]["Dice"],
            "wp_iou": entry["metrics"]["1"]["IoU"],
            "lesion_dice": entry["metrics"]["2"]["Dice"],
            "lesion_iou": entry["metrics"]["2"]["IoU"],
            "fg_mean_dice": (entry["metrics"]["1"]["Dice"] + entry["metrics"]["2"]["Dice"]) / 2,
        }

    print("Loading model for classification inference...")
    model = build_model(plans, CHECKPOINT, device)
    model._do_cls = True

    cls_results = {}
    for i, case_id in enumerate(val_cases):
        data = blosc2.open(os.path.join(PREPROCESSED_DIR, f"{case_id}.b2nd"))[:]
        data = center_crop_or_pad(data, patch_size)
        x = torch.from_numpy(data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, cls_logits = model(x)
        probs = F.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        target = parse_subtype(case_id)

        cls_results[case_id] = {
            "gt_subtype": target,
            "pred_subtype": pred,
            "cls_correct": pred == target,
            "prob_0": float(probs[0]),
            "prob_1": float(probs[1]),
            "prob_2": float(probs[2]),
        }
        print(f"  [{i+1:>2}/{len(val_cases)}] {case_id}: pred={pred} target={target} probs={probs.round(3)}")

    rows = []
    for case_id in val_cases:
        seg = seg_metrics.get(case_id, {})
        cls = cls_results.get(case_id, {})
        rows.append({
            "case_id": case_id,
            "gt_subtype": cls.get("gt_subtype", ""),
            "pred_subtype": cls.get("pred_subtype", ""),
            "cls_correct": cls.get("cls_correct", ""),
            "prob_0": f"{cls.get('prob_0', 0):.4f}",
            "prob_1": f"{cls.get('prob_1', 0):.4f}",
            "prob_2": f"{cls.get('prob_2', 0):.4f}",
            "wp_dice": f"{seg.get('wp_dice', 0):.4f}",
            "wp_iou": f"{seg.get('wp_iou', 0):.4f}",
            "lesion_dice": f"{seg.get('lesion_dice', 0):.4f}",
            "lesion_iou": f"{seg.get('lesion_iou', 0):.4f}",
            "fg_mean_dice": f"{seg.get('fg_mean_dice', 0):.4f}",
        })

    csv_path = os.path.join(BASE, "validation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nValidation CSV saved to: {csv_path}")

    # ================================================================
    # PHASE 2: Copy best/worst masks for technical report
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Copying best/worst masks for technical report")
    print("=" * 60)

    ranked = sorted(rows, key=lambda r: float(r["fg_mean_dice"]), reverse=True)
    best_2 = ranked[:2]
    worst_2 = ranked[-2:]

    report_dir = os.path.join(BASE, "report_masks")
    os.makedirs(report_dir, exist_ok=True)

    print("\nBest 2 cases (by foreground mean Dice):")
    for r in best_2:
        cid = r["case_id"]
        score = r["fg_mean_dice"]
        src_pred = os.path.join(VAL_PRED_DIR, f"{cid}.nii.gz")
        src_gt = os.path.join(LABEL_DIR, f"{cid}.nii.gz")
        dst_pred = os.path.join(report_dir, f"BEST_{cid}_pred.nii.gz")
        dst_gt = os.path.join(report_dir, f"BEST_{cid}_gt.nii.gz")
        if os.path.isfile(src_pred):
            shutil.copy2(src_pred, dst_pred)
        if os.path.isfile(src_gt):
            shutil.copy2(src_gt, dst_gt)
        print(f"  {cid}: fg_mean_dice={score} (wp={r['wp_dice']}, les={r['lesion_dice']})")

    print("\nWorst 2 cases (by foreground mean Dice):")
    for r in worst_2:
        cid = r["case_id"]
        score = r["fg_mean_dice"]
        src_pred = os.path.join(VAL_PRED_DIR, f"{cid}.nii.gz")
        src_gt = os.path.join(LABEL_DIR, f"{cid}.nii.gz")
        dst_pred = os.path.join(report_dir, f"WORST_{cid}_pred.nii.gz")
        dst_gt = os.path.join(report_dir, f"WORST_{cid}_gt.nii.gz")
        if os.path.isfile(src_pred):
            shutil.copy2(src_pred, dst_pred)
        if os.path.isfile(src_gt):
            shutil.copy2(src_gt, dst_gt)
        print(f"  {cid}: fg_mean_dice={score} (wp={r['wp_dice']}, les={r['lesion_dice']})")

    # ================================================================
    # PHASE 3: Test inference
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Running inference on test dataset")
    print("=" * 60)

    output_dir = os.path.join(BASE, "Keishi_Suzuki_results")
    os.makedirs(output_dir, exist_ok=True)

    # 3a. Segmentation using nnUNetPredictor
    print("\n[3a] Running segmentation inference with nnUNetPredictor...")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=RESULT_DIR,
        use_folds=(0,),
        checkpoint_name='checkpoint_latest.pth',
    )

    net = predictor.network
    if hasattr(net, '_do_cls'):
        net._do_cls = False
    elif hasattr(net, 'module') and hasattr(net.module, '_do_cls'):
        net.module._do_cls = False

    predictor.predict_from_files(
        list_of_lists_or_source_folder=TEST_IMAGES_DIR,
        output_folder_or_list_of_truncated_output_files=output_dir,
    )
    print(f"Segmentation masks saved to: {output_dir}")

    # Clean up non-.nii.gz files from output (nnUNet may produce json/pkl)
    for f in os.listdir(output_dir):
        if not f.endswith(".nii.gz"):
            os.remove(os.path.join(output_dir, f))

    # 3b. Classification inference on test images
    print("\n[3b] Running classification inference on test images...")
    model._do_cls = True

    test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith("_0000.nii.gz")])

    test_cls_results = []
    for i, fname in enumerate(test_images):
        case_name = fname.replace("_0000.nii.gz", "")
        img_path = os.path.join(TEST_IMAGES_DIR, fname)

        data = preprocess_raw_image(img_path, target_spacing, intensity_props, patch_size)
        x = torch.from_numpy(data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, cls_logits = model(x)
        probs = F.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

        test_cls_results.append({
            "Names": f"{case_name}.nii.gz",
            "Subtype": pred,
        })
        print(f"  [{i+1:>2}/{len(test_images)}] {case_name}: pred={pred} probs={probs.round(3)}")

    # Save subtype_results.csv inside the results folder
    csv_test_path = os.path.join(output_dir, "subtype_results.csv")
    with open(csv_test_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Names", "Subtype"])
        writer.writeheader()
        writer.writerows(test_cls_results)

    # Also save a copy at root level
    csv_root_path = os.path.join(BASE, "subtype_results.csv")
    shutil.copy2(csv_test_path, csv_root_path)

    print(f"\nClassification CSV saved to: {csv_test_path}")
    print(f"Copy also at: {csv_root_path}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"  Validation CSV:      {os.path.join(BASE, 'validation_results.csv')}")
    print(f"  Report masks:        {report_dir}")
    print(f"  Test seg masks:      {output_dir}")
    print(f"  Test cls CSV:        {csv_test_path}")
    print(f"\nTo create the zip for submission:")
    print(f"  cd \"{output_dir}\"")
    print(f"  # Compress the folder contents into Keishi_Suzuki_results.zip")


if __name__ == "__main__":
    main()
