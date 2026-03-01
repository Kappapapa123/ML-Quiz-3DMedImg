"""
Convert NIfTI label masks in report_masks to ITK-SNAP compatible format
(uint8) and copy the corresponding CT images so that masks can be loaded
as segmentation overlays on top of the original scans.

Output layout in report_mask_ITK/:
    BEST_quiz_0_184_ct.nii.gz   <- original CT (main image)
    BEST_quiz_0_184_gt.nii.gz   <- ground-truth segmentation overlay
    BEST_quiz_0_184_pred.nii.gz <- predicted segmentation overlay

ITK-SNAP usage:
  1. File -> Open Main Image  ->  *_ct.nii.gz
  2. Segmentation -> Open Segmentation  ->  *_gt.nii.gz  or  *_pred.nii.gz
"""
import os
import re
import shutil

import numpy as np
import nibabel as nib

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "report_masks")
DST_DIR = os.path.join(BASE, "report_mask_ITK")
IMAGES_TR = os.path.join(
    BASE, "nnUNet_storage", "nnUNet_raw", "Dataset001_Pancreas", "imagesTr",
)

CASE_RE = re.compile(r"(BEST|WORST)_(quiz_\d+_\d+)_(gt|pred)\.nii\.gz")


def convert_mask(src: str, dst: str):
    """Convert a label mask to uint8 with a clean NIfTI header."""
    img = nib.load(src)
    data = np.asarray(img.dataobj)
    data = np.clip(data, 0, 255).astype(np.uint8)

    out = nib.Nifti1Image(data, img.affine)
    zooms = img.header.get_zooms()[:3]
    out.header.set_zooms(zooms)
    out.set_data_dtype(np.uint8)
    nib.save(out, dst)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    if not os.path.isdir(SRC_DIR):
        print(f"Source directory not found: {SRC_DIR}")
        return

    mask_files = sorted(
        f for f in os.listdir(SRC_DIR)
        if f.endswith((".nii.gz", ".nii"))
        and os.path.isfile(os.path.join(SRC_DIR, f))
    )
    if not mask_files:
        print(f"No NIfTI files found in {SRC_DIR}")
        return

    seen_cases: set[str] = set()

    print(f"Converting {len(mask_files)} mask(s) from report_masks -> report_mask_ITK")
    for f in mask_files:
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(DST_DIR, f)
        try:
            convert_mask(src, dst)
            print(f"  OK  (mask):  {f}")
        except Exception as e:
            print(f"  FAIL (mask): {f} - {e}")
            continue

        m = CASE_RE.match(f)
        if m:
            prefix, case_id, _ = m.groups()
            key = f"{prefix}_{case_id}"
            if key not in seen_cases:
                seen_cases.add(key)
                ct_src = os.path.join(IMAGES_TR, f"{case_id}_0000.nii.gz")
                ct_dst = os.path.join(DST_DIR, f"{prefix}_{case_id}_ct.nii.gz")
                if os.path.isfile(ct_src):
                    shutil.copy2(ct_src, ct_dst)
                    print(f"  OK  (CT):    {prefix}_{case_id}_ct.nii.gz")
                else:
                    print(f"  WARN: CT not found for {case_id} at {ct_src}")

    print(f"\nDone. Output in: {DST_DIR}")
    print(
        "\nITK-SNAP usage:\n"
        "  1. File -> Open Main Image        -> *_ct.nii.gz\n"
        "  2. Segmentation -> Open Segmentation -> *_gt.nii.gz or *_pred.nii.gz"
    )


if __name__ == "__main__":
    main()
