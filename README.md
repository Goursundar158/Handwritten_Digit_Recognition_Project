# Handwritten Digit and Character Recognition System — Digit Segmentation Prototype

This is a complete, beginner-friendly hands-on project based on **Project 7: Handwritten Digit and Character Recognition System**. It implements the digit pathway: preprocessing, connected-component segmentation, feature extraction and classification.

The dataset is loaded from `sklearn.datasets.load_digits()`, so the project works offline and needs no large download. MNIST, EMNIST and IAM are described in the documentation as larger upgrade paths; letter-level character recognition is not implemented in this prototype.

## What is included

- `Documentation.docx` — paper-style documentation with the required sections and 15 references.
- `DOCUMENTATION.md` — the same documentation in Markdown.
- `main.py` — single entry point; run the whole project with `python main.py`.
- `src/preprocessing.py` — grayscale, denoising, Otsu binarization, polarity correction, crop, deskew and resize.
- `src/segmentation.py` — connected-component segmentation for form-style digit strings.
- `src/features.py` — raw-pixel and NumPy HOG feature extraction.
- `src/train.py` — trains 12 configurations and saves metrics/model/figures.
- `src/predict.py` — command-line prediction for one digit image.
- `src/predict_form.py` — segments and classifies a form image containing several digits.
- `src/generate_samples.py` — creates sample PNGs, a form sample and stage figures.
- `tests/` — 7 unit/smoke tests.
- `.vscode/launch.json` — VS Code **Run and Debug** configurations.
- `docs/make_documentation.py` — regenerates `Documentation.docx`.
- `run_project.bat` — one-click setup, training, testing and demos for Windows.
- `results/` — generated metrics and figures.
- `data/samples/` — example digit and form images.

## Quick start (Windows)

Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
python -m src.predict --image data\samples\digit_8_degraded.png
python -m src.predict_form --image data\samples\form_3014.png --save-figure
```

Alternatively, double-click `run_project.bat`.

`main.py` is the simplest entry point. `python -m src.train` does the same training step.

If `python` is not on PATH, use `py -3` instead.

### VS Code users

Open the project folder in VS Code and press **F5**. Configurations:

1. Run Full Project — trains everything via `main.py`
2. Generate Documentation — rebuilds `Documentation.docx`
3. Predict Sample Digit — single-digit demo
4. Predict Sample Form — segmentation demo
5. Run Tests — runs all 7 tests

## Expected result

Best configuration: **HOG (32×32) + RBF SVM**

- Accuracy: **98.89%** (356 / 360 test images)
- Macro F1: **0.9888**
- Feature count: **324**
- Split: 1,437 training and 360 test images, stratified, `random_state=42`

Form demo:

```text
Image              : data\samples\form_3014.png
Digits segmented   : 4
Recognized string  : 3014
```

See `results/metrics.json`, `results/confusion_matrix.png` and `results/segmentation_boxes.png`.

## Single-digit prediction

```powershell
python -m src.predict --image data\samples\digit_3_1.png
```

Example output:

```text
Image              : data\samples\digit_8_degraded.png
Predicted digit    : 8
Confidence         : 0.741
Model / features   : SVM (RBF) / hog32
```

## Form segmentation and recognition

```powershell
python -m src.predict_form --image data\samples\form_3014.png --save-figure
```

The tool thresholds the image, finds connected components, filters noise, centers each digit on a 128×128 canvas, classifies each crop and prints the recognized digit string.

You can use your own image containing separated digits:

```powershell
python -m src.predict_form --image "C:\path\to\your_form.png" --save-figure
```

## Project flow

```text
Form image
   |
   v
Grayscale -> Otsu binarization -> polarity fix
   |
   v
Morphological closing -> connected components
   |
   v
Filter noise -> split wide components -> center each digit
   |
   v
Resize to 32x32 -> HOG descriptor (324 values)
   |
   v
StandardScaler -> SVM / k-NN / Logistic Regression
   |
   v
Recognized digit string, e.g. 3014
```

## Experiment configurations

| Feature variant | Description | Dimensions |
|---|---|---:|
| `raw8` | Original 8×8 pixels, baseline | 64 |
| `gray32` | Grayscale image resized to 32×32 | 1,024 |
| `hog32` | HOG descriptor from 32×32 image | 324 |
| `hybrid` | 16×16 pixels + HOG | 580 |

Classifiers: k-nearest neighbours (`k=5`), logistic regression and RBF SVM (`C=10`, `gamma='scale'`).

## Limitations and next steps

- This is a digit-only prototype; EMNIST/IAM letter classes are not trained.
- Heavily touching digits may need watershed or contour-based splitting.
- The built-in 8×8 dataset is small; real handwriting accuracy may be lower.
- Good extensions: MNIST/EMNIST training, CNN comparison, augmentation, cross-validation and a GUI/web upload page.

## Citation

Cite `Documentation.docx` and the scikit-learn/UCI references listed there.
