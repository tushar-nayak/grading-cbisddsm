#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined script:
- Builds the CBIS-DDSM paired manifest (CC/MLO pairs) -> dicom_clean_train.csv
- Produces a comprehensive dataset report (counts, distributions, unmatched rows, etc.)
- Prints a short verification block you can paste here for a quick sanity check.

Directory layout expected (adjust if yours differs):
/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-proj/
    jpeg/                     <-- UID-named subfolders containing .jpg files
    mass_case_description_train_set.csv
    calc_case_description_train_set.csv
"""

import os
import glob
import json
import pandas as pd

# ----------------------------------------------------------------------
# 1. Paired manifest generation (your original logic, slightly refactored)
# ----------------------------------------------------------------------
def generate_paired_manifest(
    base_dir: str,
    output_csv: str = "/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv",
) -> pd.DataFrame:
    """
    Build the CC/MLO paired manifest used by your training pipeline.
    Returns the DataFrame (also saved to output_csv).
    """
    jpeg_dir = os.path.join(base_dir, "jpeg")
    print("[Manifest] Scanning for JPEG images...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)

    uid_to_jpg = {}
    for jpg in all_jpegs:
        uid = os.path.basename(os.path.dirname(jpg))
        uid_to_jpg[uid] = jpg   # keep first if multiple (unlikely)

    print(f"[Manifest] Found {len(uid_to_jpg)} unique images in the jpeg directory.")

    # Load metadata CSVs
    mass_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(base_dir, "calc_case_description_train_set.csv")

    dfs = []
    if os.path.exists(mass_csv):
        dfs.append(pd.read_csv(mass_csv))
    if os.path.exists(calc_csv):
        dfs.append(pd.read_csv(calc_csv))

    if not dfs:
        raise FileNotFoundError(
            "Could not find the CBIS-DDSM metadata CSVs. Check your dataset folder."
        )

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()  # clean whitespace

    # Match metadata to JPEGs
    print("[Manifest] Matching metadata to physical JPEG files...")
    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str):
            return None
        for uid, jpg_path in uid_to_jpg.items():
            if uid in dicom_path:
                return jpg_path
        return None

    df["jpg_path"] = df["image file path"].apply(get_jpg_path)
    df_clean = df.dropna(subset=["jpg_path"])

    # Pair CC and MLO views
    print("[Manifest] Pairing CC and MLO views for each patient...")
    paired = df_clean.pivot_table(
        index=["patient_id", "left or right breast", "assessment"],
        columns="image view",
        values="jpg_path",
        aggfunc="first",
    ).reset_index()

    if "CC" not in paired.columns or "MLO" not in paired.columns:
        raise ValueError("Failed to find both CC and MLO views in the data.")

    paired = paired.dropna(subset=["CC", "MLO"])

    # Format for the spatial‑alignment pipeline
    final_df = paired[["CC", "MLO", "assessment"]].copy()
    final_df = final_df.rename(
        columns={
            "CC": "cc_image_path",
            "MLO": "mlo_image_path",
            "assessment": "birads_label",
        }
    )
    final_df["birads_label"] = final_df["birads_label"].astype(int)

    # Save manifest
    final_df.to_csv(output_csv, index=False)
    print(
        f"\n[Manifest] Success! paired manifest saved to {output_csv}"
        f"\n[Manifest] Total valid CC/MLO pairs ready for training: {len(final_df)}"
    )
    return final_df


# ----------------------------------------------------------------------
# 2. Dataset reporting (counts, distributions, etc.)
# ----------------------------------------------------------------------
def generate_dataset_report(
    base_dir: str,
    report_dir: str = "/home/sofa/host_dir/spatial_alignment/dataset_reports",
) -> dict:
    """
    Create a rich report about the CBIS-DDSM training metadata.
    Returns a dictionary with overview statistics.
    """
    jpeg_dir = os.path.join(base_dir, "jpeg")
    os.makedirs(report_dir, exist_ok=True)

    mass_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    calc_csv = os.path.join(base_dir, "calc_case_description_train_set.csv")

    print("[Report] Scanning JPEG directory...")
    all_jpegs = glob.glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)

    uid_to_paths = {}
    for jpg in all_jpegs:
        uid = os.path.basename(os.path.dirname(jpg))
        uid_to_paths.setdefault(uid, []).append(jpg)

    uid_to_jpg = {uid: paths[0] for uid, paths in uid_to_paths.items()}
    duplicate_uid_dirs = {
        uid: paths for uid, paths in uid_to_paths.items() if len(paths) > 1
    }

    print(f"[Report] Found {len(all_jpegs)} jpg files")
    print(f"[Report] Found {len(uid_to_jpg)} unique UID folders")
    if duplicate_uid_dirs:
        print(
            f"[Report] Warning: {len(duplicate_uid_dirs)} UID folders contain >1 jpg"
        )

    # Load metadata
    print("[Report] Loading metadata CSVs...")
    dfs = []
    if os.path.exists(mass_csv):
        mass_df = pd.read_csv(mass_csv)
        mass_df["source_csv"] = "mass_train"
        dfs.append(mass_df)

    if os.path.exists(calc_csv):
        calc_df = pd.read_csv(calc_csv)
        calc_df["source_csv"] = "calc_train"
        dfs.append(calc_df)

    if not dfs:
        raise FileNotFoundError("Could not find training metadata CSVs in base_dir.")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[Report] Loaded {len(df)} total metadata rows")

    # Match to JPEGs (exact path‑component match – safer than substring)
    def get_jpg_path(dicom_path):
        if not isinstance(dicom_path, str):
            return None
        parts = dicom_path.strip().split("/")
        for part in parts:
            if part in uid_to_jpg:
                return uid_to_jpg[part]
        return None

    print("[Report] Matching metadata rows to JPEGs...")
    df["jpg_path"] = df["image file path"].apply(get_jpg_path)
    df["jpg_found"] = df["jpg_path"].notna()

    df_matched = df[df["jpg_found"]].copy()
    df_unmatched = df[~df["jpg_found"]].copy()

    print(
        f"[Report] Matched rows: {len(df_matched)}"
        f", Unmatched rows: {len(df_unmatched)}"
    )

    # Pairing summary (CC/MLO)
    print("[Report] Computing CC/MLO pairing summary...")
    pair_index = ["patient_id", "left or right breast", "assessment"]
    paired = df_matched.pivot_table(
        index=pair_index,
        columns="image view",
        values="jpg_path",
        aggfunc="first",
    ).reset_index()

    has_cc = "CC" in paired.columns
    has_mlo = "MLO" in paired.columns

    if has_cc and has_mlo:
        paired["has_both_views"] = paired["CC"].notna() & paired["MLO"].notna()
        paired_complete = paired[paired["has_both_views"]].copy()
    else:
        paired["has_both_views"] = False
        paired_complete = paired.iloc[0:0].copy()

    # Helper to save value‑counts
    def save_counts(dataframe, column, filename, dropna=True):
        if column not in dataframe.columns:
            return
        s = dataframe[column]
        if dropna:
            s = s.dropna()
        counts = s.value_counts(dropna=not dropna).reset_index()
        counts.columns = [column, "count"]
        counts_path = os.path.join(report_dir, filename)
        counts.to_csv(counts_path, index=False)

    print("[Report] Saving summary CSVs...")
    save_counts(df, "source_csv", "source_csv_counts.csv")
    save_counts(df, "image view", "image_view_counts_all.csv")
    save_counts(df_matched, "image view", "image_view_counts_matched.csv")
    save_counts(df, "left or right breast", "breast_side_counts.csv")
    save_counts(df, "assessment", "assessment_counts.csv")
    save_counts(df, "pathology", "pathology_counts.csv")
    save_counts(df, "abnormality type", "abnormality_type_counts.csv")

    # Density may appear under either name
    if "breast_density" in df.columns:
        save_counts(df, "breast_density", "breast_density_counts.csv")
    if "breast density" in df.columns:
        save_counts(df, "breast density", "breast_density_counts_alt.csv")

    # Lesion‑specific descriptors (may be sparse)
    save_counts(df, "mass shape", "mass_shape_counts.csv")
    save_counts(df, "mass margins", "mass_margin_counts.csv")
    save_counts(df, "calc type", "calc_type_counts.csv")
    save_counts(df, "calc distribution", "calc_distribution_counts.csv")

    # Patient‑level stats
    patient_image_counts = (
        df.groupby("patient_id")
        .size()
        .reset_index(name="num_rows")
        .sort_values("num_rows", ascending=False)
    )
    patient_image_counts.to_csv(
        os.path.join(report_dir, "patient_row_counts.csv"), index=False
    )

    matched_patient_counts = (
        df_matched.groupby("patient_id")
        .size()
        .reset_index(name="num_matched_rows")
        .sort_values("num_matched_rows", ascending=False)
    )
    matched_patient_counts.to_csv(
        os.path.join(report_dir, "patient_matched_row_counts.csv"), index=False
    )

    # Save intermediate tables for debugging
    df_matched.to_csv(
        os.path.join(report_dir, "matched_metadata_with_jpg.csv"), index=False
    )
    df_unmatched.to_csv(
        os.path.join(report_dir, "unmatched_metadata_rows.csv"), index=False
    )
    paired.to_csv(
        os.path.join(report_dir, "pairing_table_all.csv"), index=False
    )
    paired_complete.to_csv(
        os.path.join(report_dir, "pairing_table_complete_only.csv"), index=False
    )

    # Build overview dict
    overview = {
        "base_dir": base_dir,
        "jpeg_dir": jpeg_dir,
        "total_jpg_files": int(len(all_jpegs)),
        "unique_uid_folders": int(len(uid_to_jpg)),
        "uids_with_multiple_jpgs": int(len(duplicate_uid_dirs)),
        "total_metadata_rows": int(len(df)),
        "matched_metadata_rows": int(len(df_matched)),
        "unmatched_metadata_rows": int(len(df_unmatched)),
        "unique_patient_ids_all": int(df["patient_id"].nunique())
        if "patient_id" in df.columns
        else None,
        "unique_patient_ids_matched": int(df_matched["patient_id"].nunique())
        if "patient_id" in df_matched.columns
        else None,
        "pair_groups_total": int(len(paired)),
        "pair_groups_complete_cc_mlo": int(len(paired_complete)),
        "complete_pair_rate_among_groups": float(
            len(paired_complete) / len(paired) if len(paired) > 0 else 0.0
        ),
        "columns_in_metadata": list(df.columns),
    }

    # Save overview as JSON and CSV
    with open(os.path.join(report_dir, "dataset_overview.json"), "w") as f:
        json.dump(overview, f, indent=2)

    overview_rows = [{"metric": k, "value": v} for k, v in overview.items() if k != "columns_in_metadata"]
    pd.DataFrame(overview_rows).to_csv(
        os.path.join(report_dir, "dataset_overview.csv"), index=False
    )

    # Human‑readable summary
    with open(os.path.join(report_dir, "dataset_summary.txt"), "w") as f:
        f.write("CBIS-DDSM TRAIN DATASET REPORT\n")
        f.write("=" * 40 + "\n")
        for k, v in overview.items():
            if k == "columns_in_metadata":
                f.write(f"{k}: {', '.join(map(str, v))}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"\n[Report] Reports saved to: {report_dir}")
    print(f"[Report] Complete CC/MLO pairs: {len(paired_complete)} / {len(paired)}")
    return overview


# ----------------------------------------------------------------------
# 3. Verification summary (print‑only, copy‑paste friendly)
# ----------------------------------------------------------------------
def print_verification_summary(
    report_dir: str = "/home/sofa/host_dir/spatial_alignment/dataset_reports",
):
    """
    Print a compact block you can copy‑paste here for a quick sanity check.
    """
    import json
    import pandas as pd

    overview_json = os.path.join(report_dir, "dataset_overview.json")
    matched_csv = os.path.join(report_dir, "matched_metadata_with_jpg.csv")
    unmatched_csv = os.path.join(report_dir, "unmatched_metadata_rows.csv")
    paired_csv = os.path.join(report_dir, "pairing_table_complete_only.csv")

    if not os.path.exists(overview_json):
        raise FileNotFoundError(f"Missing {overview_json}")

    with open(overview_json, "r") as f:
        overview = json.load(f)

    print("\n" + "=" * 60)
    print("CBIS-DDSM VERIFICATION SUMMARY")
    print("=" * 60)

    keys = [
        "total_jpg_files",
        "unique_uid_folders",
        "uids_with_multiple_jpgs",
        "total_metadata_rows",
        "matched_metadata_rows",
        "unmatched_metadata_rows",
        "unique_patient_ids_all",
        "unique_patient_ids_matched",
        "pair_groups_total",
        "pair_groups_complete_cc_mlo",
        "complete_pair_rate_among_groups",
    ]
    for k in keys:
        print(f"{k}: {overview.get(k)}")

    if os.path.exists(matched_csv):
        dfm = pd.read_csv(matched_csv)
        print("\n--- Matched rows distributions ---")

        for col in [
            "source_csv",
            "image view",
            "left or right breast",
            "assessment",
            "pathology",
            "abnormality type",
        ]:
            if col in dfm.columns:
                print(f"\n{col}:")
                print(
                    dfm[col]
                    .value_counts(dropna=False)
                    .sort_index()
                    .to_string()
                )

        density_col = None
        if "breast_density" in dfm.columns:
            density_col = "breast_density"
        elif "breast density" in dfm.columns:
            density_col = "breast density"

        if density_col:
            print(f"\n{density_col}:")
            print(
                dfm[density_col]
                .value_counts(dropna=False)
                .sort_index()
                .to_string()
            )

    if os.path.exists(unmatched_csv):
        dfu = pd.read_csv(unmatched_csv)
        print("\n--- Unmatched sample rows ---")
        cols = [c for c in ["patient_id", "image view", "image file path"] if c in dfu.columns]
        if len(dfu) == 0:
            print("No unmatched rows")
        else:
            print(dfu[cols].head(10).to_string(index=False))

    if os.path.exists(paired_csv):
        dfp = pd.read_csv(paired_csv)
        print("\n--- Complete pair sample ---")
        cols = [c for c in ["patient_id", "left or right breast", "assessment", "CC", "MLO"] if c in dfp.columns]
        if len(dfp) == 0:
            print("No complete CC/MLO pairs found")
        else:
            print(dfp[cols].head(10).to_string(index=False))

    print("\n" + "=" * 60)


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = "/home/sofa/host_dir/spatial_alignment/dataset/raw/cbisddsm-proj"
    MANIFEST_OUT = "/home/sofa/host_dir/spatial_alignment/dicom_clean_train.csv"
    REPORT_DIR = "/home/sofa/host_dir/spatial_alignment/dataset_reports"

    # 1. Build paired manifest (your original need)
    generate_paired_manifest(base_dir=BASE_DIR, output_csv=MANIFEST_OUT)

    # 2. Generate detailed report
    generate_dataset_report(base_dir=BASE_DIR, report_dir=REPORT_DIR)

    # 3. Print verification block you can copy‑paste here
    print_verification_summary(report_dir=REPORT_DIR)