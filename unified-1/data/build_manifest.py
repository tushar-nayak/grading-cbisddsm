import os
import glob
import pandas as pd

def generate_paired_manifest():
    base_dir = "/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-proj"
    jpeg_dir = os.path.join(base_dir, "jpeg")
    output_csv = "../data/spatial_manifest.csv"
    
    print("Scanning for JPEG images...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
    uid_to_jpg = {os.path.basename(os.path.dirname(jpg)): jpg for jpg in all_jpegs}

    mass_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(base_dir, "calc_case_description_train_set.csv")
    
    dfs = [pd.read_csv(f) for f in [mass_csv, calc_csv] if os.path.exists(f)]
    if not dfs:
        raise FileNotFoundError("Could not find CBIS-DDSM metadata CSVs.")
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str): return None
        for uid, jpg_path in uid_to_jpg.items():
            if uid in dicom_path: return jpg_path
        return None

    df['jpg_path'] = df['image file path'].apply(get_jpg_path)
    df_clean = df.dropna(subset=['jpg_path'])

    paired = df_clean.pivot_table(
        index=['patient_id', 'left or right breast', 'assessment'], 
        columns='image view', values='jpg_path', aggfunc='first'
    ).reset_index()

    paired = paired.dropna(subset=['CC', 'MLO'])

    # Mocking mask and bbox paths for the new pipeline structure
    # In reality, you'd link your ground truth mask paths here
    final_df = paired[['CC', 'MLO', 'assessment']].copy()
    final_df = final_df.rename(columns={'CC': 'cc_image_path', 'MLO': 'mlo_image_path', 'assessment': 'birads_label'})
    final_df['cc_mask_path'] = final_df['cc_image_path'].str.replace('.jpg', '_mask.jpg')
    final_df['mlo_mask_path'] = final_df['mlo_image_path'].str.replace('.jpg', '_mask.jpg')
    final_df['cc_bbox'] = "50,50,100,100" # Placeholder
    final_df['mlo_bbox'] = "60,60,110,110" # Placeholder

    final_df['birads_label'] = final_df['birads_label'].astype(int)
    final_df.to_csv(output_csv, index=False)
    print(f"Success! Paired manifest saved to {output_csv}. Total valid pairs: {len(final_df)}")

if __name__ == "__main__":
    generate_paired_manifest()