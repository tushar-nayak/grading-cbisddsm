import argparse
from pathlib import Path

from unified_mammo_pipeline.pipeline import UnifiedMammoPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the unified mammography pipeline trial")
    parser.add_argument("--manifest-csv", default=str(Path(__file__).resolve().parents[1] / "dicom_clean_train.csv"))
    parser.add_argument("--classifier-checkpoint", default=str(Path(__file__).resolve().parents[1] / "output" / "cnn_attentional_weights.pth"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "trial_run"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = UnifiedMammoPipeline(
        manifest_csv=args.manifest_csv,
        classifier_checkpoint=args.classifier_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    out_path, payload = pipeline.run(limit=args.limit)
    print(f"Saved {len(payload)} trial results to {out_path}")
    if payload:
        first = payload[0]
        print(
            f"First sample: {first['sample_id']} | true={first['true_birads']} | "
            f"pred={first['predicted_birads']} | source={first['classifier_source']}"
        )


if __name__ == "__main__":
    main()
