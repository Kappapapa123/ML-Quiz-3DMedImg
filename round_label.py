import nibabel as nib
import numpy as np
import os

# Paths to your folders
folders_to_fix = [
    r'C:\Users\Keishi Suzuki\Projects\ML-Quiz-3DMedImg\nnUNet_storage\nnUNet_raw\Dataset001_Pancreas\labelsTr'
]

for label_dir in folders_to_fix:
    if not os.path.exists(label_dir):
        continue
        
    print(f"Cleaning: {label_dir}")
    for filename in os.listdir(label_dir):
        if filename.endswith('.nii.gz'):
            path = os.path.join(label_dir, filename)
            img = nib.load(path)
            data = img.get_fdata()
            
            # Use int64 to match nnU-Net's strict expectation
            clean_data = np.round(data).astype(np.int64)
            
            new_img = nib.Nifti1Image(clean_data, img.affine, img.header)
            
            # Update the header to reflect 64-bit integers
            new_img.set_data_dtype(np.int64) 
            
            nib.save(new_img, path)
            
print("All files converted to 64-bit integers. Your fingerprint extraction should run smoothly now!")