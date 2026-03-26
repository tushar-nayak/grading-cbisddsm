import argparse
from pathlib import Path

try:
    from unified_mammo_pipeline.pipeline import UnifiedMammoPipeline
except ImportError:
    from pipeline import UnifiedMammoPipeline


def default_classifier_checkpoint() -> str:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "output" / "cnn_attentional_weights.pth",
        root / "unified-init" / "output" / "cnn_attentional_weights.pth",
        root / "raw-5" / "output" / "cnn_attentional_weights.pth",
        root / "raw-4" / "output" / "cnn_attentional_weights.pth",
        root / "raw-4-og" / "output" / "cnn_attentional_weights.pth",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Run the unified mammography pipeline trial")
    parser.add_argument("--manifest-csv", default=str(Path(__file__).resolve().parents[1] / "dicom_clean_train.csv"))
    parser.add_argument("--classifier-checkpoint", default=default_classifier_checkpoint())
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
