# Spatial Alignment Code Report

## Current Summary

This repository implements a CBIS-DDSM mammography workflow for paired CC/MLO breast views. The code builds paired manifests from the Kaggle-style CBIS-DDSM export, loads and preprocesses mammogram images, segments likely lesion regions, detects bounding boxes, aligns MLO views to CC views, estimates cross-view correspondence, and produces binary BI-RADS classification outputs.

The main inference and evaluation scripts are currently runnable after the directory move. The training scripts are structurally present, but they are blocked in the current Python environment because OpenCV is not installed.

## What The Code Is Doing

### Data Loading

- `code/data.py` defines manifest samples and view data.
- It reads paired manifest CSV files with CC/MLO image paths.
- It loads grayscale mammograms with PIL.
- It flips right-breast images so orientation is consistent.
- It performs contrast enhancement and extracts a breast mask.

### Segmentation

- `code/segmentation.py` provides a heuristic lesion segmentation path.
- `code/localizer.py` defines a small U-Net model and optional checkpoint-based localization.
- If no localizer checkpoint is available, the pipeline can still run using heuristic segmentation.

### Detection

- `code/detection.py` converts lesion-like regions into bounding boxes.
- It uses image intensity and connected-component logic rather than a heavy detection model.
- This makes the pipeline lightweight and runnable without trained detection weights.

### Alignment

- `code/alignment.py` aligns MLO output to CC output using mask centroids.
- It computes the shift needed to bring MLO lesion masks closer to CC lesion masks.
- The alignment stage outputs before/after metrics and overlay images in the full pipeline scripts.

### Correspondence

- `code/correspondence.py` estimates whether CC and MLO regions correspond.
- It combines mask overlap, centroid distance, area ratio, and image-quality estimates.
- The output is a structured correspondence result with confidence-like fields.

### Classification

- `code/classification.py` defines a ResNet50 + cross-attention classifier.
- If a classifier checkpoint exists, it loads the model and predicts BI-RADS-style outputs.
- If no checkpoint is found, it falls back to a lesion-size heuristic.
- Binary labels are mapped as BI-RADS 0-3 vs BI-RADS 4-5.

### Full Pipeline Scripts

- `code/run_kaggle_full_pipeline.py` builds or reads a Kaggle CBIS-DDSM paired manifest, runs the whole workflow, and writes outputs for detection, segmentation, alignment, correspondence, classification, and metrics.
- `code/run_unified_binary_pipeline.py` wraps the same general workflow with binary classification reporting.
- `code/run_trial.py` runs the older `UnifiedMammoPipeline` wrapper on an existing paired manifest.
- `code/unified_binary_pipeline_full_run.ipynb` is a notebook version of the binary full-run pipeline.

### Training Scripts

- `code/train_localizer.py` trains the small U-Net localizer from paired manifest records and ROI masks.
- `code/train_binary_classifier_cross_attention_only.py` trains a binary ResNet50 + symmetric cross-attention classifier on CC/MLO pairs.
- Both training scripts depend on OpenCV via `cv2`.

## Current Run Status

The following checks passed:

- Python syntax compilation with `python -m compileall -q code`.
- Git whitespace/error check with `git diff --check`.
- One-sample smoke run for `code/run_kaggle_full_pipeline.py`.
- One-sample smoke run for `code/run_unified_binary_pipeline.py`.
- One-sample smoke run for `code/run_trial.py`.
- Repo-wide scan for stale pre-move absolute paths and old Windows paths.

The following checks are blocked:

- `python code/train_localizer.py --help`
- `python code/train_binary_classifier_cross_attention_only.py --help`

Both fail because the active environment does not have OpenCV installed:

```text
ModuleNotFoundError: No module named 'cv2'
```

## What Is Working Really Well

- The core pipeline can run end to end on at least one sample.
- The code has a clear staged design: data loading, segmentation, detection, alignment, correspondence, and classification are separated into modules.
- The full-run scripts produce useful artifacts, including overlays, JSON outputs, CSV metrics, and per-sample reports.
- The pipeline has graceful fallbacks when model checkpoints are missing, especially for segmentation and classification.
- Manifest generation is useful because it reconstructs paired CC/MLO examples from the raw Kaggle CBIS-DDSM metadata and JPEG folders.
- The smoke-run options, especially `--limit` and `--max-long-side`, make it practical to test the pipeline without launching a full dataset run.
- The binary classification report includes richer metrics than plain accuracy, including confusion matrix, class-level precision/recall/F1, and balanced accuracy.
- The code now points to the moved local dataset location rather than stale machine-specific paths.

## What Can Be Improved

### Dependency Management

There is no visible dependency file in the repo. The project should add one of:

- `requirements.txt`
- `environment.yml`
- `pyproject.toml`

At minimum it should include packages such as numpy, pandas, pillow, scipy, scikit-image, scikit-learn, torch, torchvision, tqdm, and OpenCV.

### Training Environment

The training scripts currently cannot start because OpenCV is missing. Install one of:

```bash
pip install opencv-python-headless
```

or:

```bash
conda install -c conda-forge opencv
```

After that, rerun the training help commands and a small one-epoch smoke training run.

### Package Naming

The source folder is named `code`, which can conflict with Python's standard-library `code` module in some import patterns. Direct script execution works, but package-style imports from the repository root can behave badly when libraries such as PyTorch internally import the standard `code` module.

A safer package name would be something like:

- `spatial_alignment`
- `mammo_pipeline`
- `cbisddsm_pipeline`

### Tests

There is no formal test suite. Useful tests would include:

- Manifest builder creates expected columns.
- Image loading handles right-breast flipping.
- Segmentation returns masks matching input shapes.
- Detection returns valid bounding boxes or `None`.
- Alignment improves or preserves centroid distance on synthetic masks.
- Binary metric computation handles one-class and two-class cases.
- Pipeline smoke test runs on one tiny fixture sample.

### Configuration

Some paths are still embedded directly in scripts. They now point to the current local layout, but the more maintainable approach is to centralize paths in a config file or environment variables.

Examples:

- Dataset base
- Output directory
- Classifier checkpoint
- Localizer checkpoint
- Manifest CSV

### Generated Artifacts

The repository contains many generated run outputs and large manifest/result files. These are useful for auditability, but they make diffs noisy. Consider moving generated outputs to an ignored `runs/` or `outputs/` directory, while keeping only small sample fixtures in Git.

### Model Checkpoint Handling

The classifier and localizer code can fall back to heuristics, which is useful, but it can hide the fact that a trained model is not actually being used. Pipeline reports should make the active source very obvious:

- `heuristic`
- `localizer_checkpoint`
- `classifier_checkpoint`

This is already partially present and should be made consistent across all outputs.

### Notebook Hygiene

The notebook is useful for exploratory work, but committed notebooks should usually have outputs cleared to avoid stale paths, huge diffs, and historical logs. The current notebook outputs were cleared for this reason.

## Recommended Next Steps

1. Add a dependency file and install OpenCV.
2. Run:

   ```bash
   python code/train_localizer.py --help
   python code/train_binary_classifier_cross_attention_only.py --help
   ```

3. Run a tiny training smoke test after OpenCV is installed, using `--epochs 1`, `--batch-size 1`, `--num-workers 0`, and a small sample limit where available.
4. Add a small automated test suite around the manifest, metrics, segmentation, detection, and alignment logic.
5. Consider renaming the `code` package to avoid standard-library import conflicts.
6. Move generated experiment artifacts out of tracked source paths or add a clear artifact-management policy.

## Bottom Line

The main inference and evaluation pipeline is working after the directory move. The most important immediate blocker is the missing OpenCV dependency for training. The biggest maintainability improvements are dependency pinning, tests, cleaner package naming, and separating generated run artifacts from source code.
