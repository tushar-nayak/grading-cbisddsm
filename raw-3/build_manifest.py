import os
import glob
import pandas as pd

def generate_paired_manifest():
    # UPDATED PATHS
    base_dir = "/root/host_dir/spatial_alignment/dataset/raw/cbisddsm-kaggle"
    jpeg_dir = os.path.join(base_dir, "jpeg")
    csv_dir = os.path.join(base_dir, "csv")
    output_csv = "/root/host_dir/spatial_alignment/raw-fourths/dicom_clean_train.csv"
    
    # 1. Map all available JPEGs
    print("Scanning for JPEG images...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
    uid_to_jpg = {os.path.basename(os.path.dirname(jpg)): jpg for jpg in all_jpegs}

    # 2. Load and Combine Metadata (Looking inside the csv/ folder)
    mass_csv = os.path.join(csv_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(csv_dir, "calc_case_description_train_set.csv")
    
    dfs = []
    if os.path.exists(mass_csv):
        m_df = pd.read_csv(mass_csv).rename(columns={'breast_density': 'breast density'})
        dfs.append(m_df)
    if os.path.exists(calc_csv):
        dfs.append(pd.read_csv(calc_csv))
        
    if not dfs:
        raise FileNotFoundError(f"Could not find CSVs in {csv_dir}")
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip() 

    # 3. Match Metadata to JPEGs
    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str): return None
        for uid, jpg_path in uid_to_jpg.items():
            if uid in dicom_path: return jpg_path
        return None

    df['jpg_path'] = df['image file path'].apply(get_jpg_path)
    df_clean = df.dropna(subset=['jpg_path'])

    # 4. Aggregate Abnormality Labels to Image Level
    img_level = df_clean.groupby(
        ['patient_id', 'left or right breast', 'image view', 'jpg_path']
    ).agg(
        assessment=('assessment', 'max')
    ).reset_index()

    # 5. Pair CC and MLO views
    print("Pairing CC and MLO views for each patient...")
    paired = img_level.pivot_table(
        index=['patient_id', 'left or right breast'], 
        columns='image view', 
        values=['jpg_path', 'assessment'], 
        aggfunc='first'
    )
    
    paired.columns = [f"{col[0]}_{col[1]}" for col in paired.columns]
    paired = paired.reset_index()
    paired = paired.dropna(subset=['jpg_path_CC', 'jpg_path_MLO'])

    # 6. Format for the Spatial Alignment Pipeline
    final_df = paired[['jpg_path_CC', 'jpg_path_MLO', 'assessment_CC', 'left or right breast']].copy()
    final_df = final_df.rename(columns={
        'jpg_path_CC': 'cc_image_path', 
        'jpg_path_MLO': 'mlo_image_path',
        'assessment_CC': 'birads_label',
        'left or right breast': 'breast_side'
    })

    final_df['birads_label'] = final_df['birads_label'].astype(int)
    final_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Cleaned manifest saved to {output_csv}")
    print(f"Total valid CC/MLO pairs ready for STN: {len(final_df)}")

if __name__ == "__main__":
    generate_paired_manifest()