import glob
import os
from pathlib import Path

import pandas as pd


def generate_paired_manifest():
    base_dir = "/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-proj"
    jpeg_dir = os.path.join(base_dir, "jpeg")
    output_csv = Path(__file__).resolve().parent / "dicom_clean_train.csv"

    print("Scanning for JPEG images...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
    uid_to_jpg = {os.path.basename(os.path.dirname(jpg)): jpg for jpg in all_jpegs}

    mass_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(base_dir, "calc_case_description_train_set.csv")

    dfs = []
    if os.path.exists(mass_csv):
        mass_df = pd.read_csv(mass_csv).rename(columns={"breast_density": "breast density"})
        dfs.append(mass_df)
    if os.path.exists(calc_csv):
        dfs.append(pd.read_csv(calc_csv))

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str):
            return None
        for uid, jpg_path in uid_to_jpg.items():
            if uid in dicom_path:
                return jpg_path
        return None

    df["jpg_path"] = df["image file path"].apply(get_jpg_path)
    df_clean = df.dropna(subset=["jpg_path"])

    img_level = df_clean.groupby(
        ["patient_id", "left or right breast", "image view", "jpg_path"]
    ).agg(assessment=("assessment", "max")).reset_index()

    print("Pairing CC and MLO views for each patient...")
    paired = img_level.pivot_table(
        index=["patient_id", "left or right breast"],
        columns="image view",
        values=["jpg_path", "assessment"],
        aggfunc="first",
    )

    paired.columns = [f"{col[0]}_{col[1]}" for col in paired.columns]
    paired = paired.reset_index()
    paired = paired.dropna(subset=["jpg_path_CC", "jpg_path_MLO"])

    final_df = paired[
        [
            "patient_id",
            "left or right breast",
            "jpg_path_CC",
            "jpg_path_MLO",
            "assessment_CC",
            "assessment_MLO",
        ]
    ].copy()
    final_df["birads_label"] = final_df[["assessment_CC", "assessment_MLO"]].max(axis=1).astype(int)
    final_df["sample_id"] = (
        final_df["patient_id"].astype(str)
        + "_"
        + final_df["left or right breast"].astype(str).str.strip().str.upper()
    )

    final_df = final_df.rename(columns={
        "left or right breast": "breast_side",
        "jpg_path_CC": "cc_image_path",
        "jpg_path_MLO": "mlo_image_path",
    })
    final_df = final_df[
        ["patient_id", "sample_id", "cc_image_path", "mlo_image_path", "birads_label", "breast_side"]
    ]

    final_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Cleaned and aggregated manifest saved to {output_csv}")
    print(f"Total valid CC/MLO pairs ready for STN: {len(final_df)}")


if __name__ == "__main__":
    generate_paired_manifest()
