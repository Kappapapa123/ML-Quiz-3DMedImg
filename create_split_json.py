import json
import os

# 1. Define your paths
# Use the folders where you kept your files separated to identify the IDs
train_img_dir = r'C:\Users\Keishi Suzuki\Projects\ML-Quiz-3DMedImg\nnUNet_storage\nnUNet_raw\Dataset001_Pancreas\imagesTr'
val_img_dir = r'C:\Users\Keishi Suzuki\Projects\ML-Quiz-3DMedImg\nnUNet_storage\nnUNet_raw\Dataset001_Pancreas\imagesVal'

# This is where the output file MUST go
output_dir = r'C:\Users\Keishi Suzuki\Projects\ML-Quiz-3DMedImg\nnUNet_storage\nnUNet_preprocessed\Dataset001_Pancreas'
output_file = os.path.join(output_dir, 'splits_final.json')

def get_identifiers_from_folder(folder):
    # Get filenames, remove '_0000.nii.gz' or '.nii.gz'
    ids = []
    for f in os.listdir(folder):
        if f.endswith('.nii.gz'):
            # Remove extension and the _0000 suffix if it exists
            identifier = f.replace('.nii.gz', '').replace('_0000', '')
            ids.append(identifier)
    return sorted(list(set(ids)))

# 2. Collect the IDs
train_ids = get_identifiers_from_folder(train_img_dir)
val_ids = get_identifiers_from_folder(val_img_dir)

# 3. Create the split structure (Fold 0)
# nnU-Net expects a list of dictionaries
splits = [
    {
        'train': train_ids,
        'val': val_ids
    }
]

# 4. Save the file
os.makedirs(output_dir, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"Successfully created splits_final.json at {output_file}")
print(f"Train cases: {len(train_ids)} | Val cases: {len(val_ids)}")