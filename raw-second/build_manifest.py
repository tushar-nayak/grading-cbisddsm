import os
import glob
import pandas as pd

def generate_paired_manifest():
    # Define paths based on your directory structure
    base_dir = "/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-proj"
    jpeg_dir = os.path.join(base_dir, "jpeg")
    output_csv = "/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv"
    
    # 1. Map all available JPEGs
    # Your dataset structure shows jpegs are stored in folders named after DICOM UIDs
    print("Scanning for JPEG images...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
    
    # Map the unique folder ID (UID) to the actual file path
    uid_to_jpg = {}
    for jpg in all_jpegs:
        uid = os.path.basename(os.path.dirname(jpg))
        uid_to_jpg[uid] = jpg
        
    print(f"Found {len(uid_to_jpg)} unique images in the jpeg directory.")

    # 2. Load the CBIS-DDSM Metadata CSVs
    mass_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(base_dir, "calc_case_description_train_set.csv")
    
    dfs = []
    if os.path.exists(mass_csv):
        dfs.append(pd.read_csv(mass_csv))
    if os.path.exists(calc_csv):
        dfs.append(pd.read_csv(calc_csv))
        
    if not dfs:
        raise FileNotFoundError("Could not find the CBIS-DDSM metadata CSVs. Check your dataset folder.")
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()  # Clean up whitespace in CBIS-DDSM column names

    # 3. Match Metadata to JPEGs
    print("Matching metadata to physical JPEG files...")
    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str): 
            return None
        # The DICOM path in the CSV contains the UID which is our folder name
        for uid, jpg_path in uid_to_jpg.items():
            if uid in dicom_path:
                return jpg_path
        return None

    df['jpg_path'] = df['image file path'].apply(get_jpg_path)
    df_clean = df.dropna(subset=['jpg_path'])

    # 4. Pair CC and MLO views
    print("Pairing CC and MLO views for each patient...")
    # Group by patient, breast side, and BI-RADS assessment to ensure accurate clinical pairing
    paired = df_clean.pivot_table(
        index=['patient_id', 'left or right breast', 'assessment'], 
        columns='image view', 
        values='jpg_path', 
        aggfunc='first'
    ).reset_index()

    # Drop any records that don't have BOTH a CC and an MLO view
    if 'CC' not in paired.columns or 'MLO' not in paired.columns:
        raise ValueError("Failed to find both CC and MLO views in the data.")
        
    paired = paired.dropna(subset=['CC', 'MLO'])

    # 5. Format for the Spatial Alignment Pipeline
    final_df = paired[['CC', 'MLO', 'assessment']].copy()
    final_df = final_df.rename(columns={
        'CC': 'cc_image_path', 
        'MLO': 'mlo_image_path',
        'assessment': 'birads_label'
    })

    # Ensure labels are integers
    final_df['birads_label'] = final_df['birads_label'].astype(int)

    # Save to the exact location your initial1.py script expects
    final_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! paired manifest saved to {output_csv}")
    print(f"Total valid CC/MLO pairs ready for training: {len(final_df)}")

if __name__ == "__main__":
    generate_paired_manifest()