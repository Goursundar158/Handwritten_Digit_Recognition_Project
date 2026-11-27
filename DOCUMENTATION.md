# Handwritten Digit and Character Recognition System (Digit Segmentation and Classification Prototype)

## Abstract

This paper presents a reproducible prototype for Project 7, the Handwritten Digit and Character Recognition System. The implemented digit pathway preprocesses scanned handwritten images through grayscale conversion, denoising, Otsu binarization and polarity correction, then segments individual digits from form-style images using connected components and vertical projection analysis. Each digit is recentered, described by a 324-dimensional Histogram of Oriented Gradients (HOG) descriptor, standardized, and classified with k-nearest neighbours, logistic regression or an RBF support vector machine. Experiments use the scikit-learn digits dataset, a freely available 8×8 subset of the UCI optical recognition corpus, because it enables a fully offline hands-on run; MNIST, EMNIST and IAM are identified as larger upgrade datasets. A stratified 80:20 split is used. The best model, HOG with RBF SVM, reaches 98.89% accuracy and 0.9888 macro-F1. A form image containing the sequence 3014 is segmented into four components and correctly classified as 3014. The deliverable includes a Python CLI, trained model, sample images, seven tests, metrics, confusion matrices and documentation suitable for beginners.

## Keywords

Handwritten digit recognition; Image preprocessing; Connected-component segmentation; Histogram of Oriented Gradients; Support Vector Machine

## 1. Introduction

Handwritten digit and character recognition supports postal-code reading, bank-cheque processing, form digitization and automated data entry. The objective of Project 7 is broader than single-character classification: scanned handwritten images should be preprocessed, cleaned, segmented into individual digits or characters, classified and assembled into a readable result. Variation in stroke width, slant, position and noise makes this difficult.

This work implements the digit pathway of that objective as a complete hands-on prototype. It uses the 8×8 digits dataset distributed with scikit-learn, a copy of the test split of the UCI Optical Recognition of Handwritten Digits dataset [13], [15]. The dataset is deliberately small so that training, testing and documentation generation run offline on a normal laptop. MNIST [14], EMNIST and IAM are discussed as larger upgrade paths for characters and unconstrained handwriting.

The contributions are: (1) preprocessing with optional denoising, Otsu binarization, polarity correction, cropping and deskewing; (2) connected-component segmentation for form-style digit strings; (3) a pure-NumPy HOG descriptor; (4) a fair comparison of four feature variants and three classifiers; and (5) a reproducible repository with a trained model, CLI tools, seven tests and generated reports.

## 2. Related work

Early document-recognition systems relied on careful preprocessing, segmentation and handcrafted features. LeCun et al. introduced gradient-based learning for document recognition and demonstrated the value of convolutional architectures on handwritten characters [1]. The Neocognitron proposed shift-tolerant hierarchical visual processing [3], while backpropagation provided the training foundation for multilayer networks [2].

Segmentation by thresholding, connected components and morphology is a standard tool in digital image processing [4]. Szeliski describes feature extraction and local orientation descriptors used in practical vision systems [5]. Classical classifiers such as k-nearest neighbours, logistic regression and support vector machines are well covered in the statistical learning literature and are implemented consistently in scikit-learn [6], [7], [10].

Modern systems often use convolutional neural networks. Krizhevsky et al., Simonyan and Zisserman, and He et al. showed that deep CNNs can achieve very high accuracy on large image datasets [9], [11], [12]. MNIST became a standard benchmark for digit recognition [14]. EMNIST extends this idea to letters, and IAM provides unconstrained handwriting samples, but both require larger downloads and longer training. This project therefore focuses on a transparent classical pipeline that is easy to inspect and reproduce.

## 3. Preliminaries

### 3.1 Image normalization

A digit image is converted to grayscale using standard colour-channel weights and resized to a fixed 32×32 grid. Fixed size is required because classifiers expect a constant feature length.

### 3.2 Noise reduction and binarization

Median filtering replaces each pixel with the neighbourhood median and is effective against impulse noise [4]. Otsu's method automatically selects a threshold that separates foreground from background by maximizing between-class variance [4]. Polarity correction ensures that the digit is white on black; this is useful when a scan has a white background.

### 3.3 Segmentation

A form image is thresholded, then a small morphological closing operation joins broken strokes. Eight-connected component labeling finds foreground blobs. Components below an area threshold are discarded as noise. Very wide components are inspected with a vertical projection profile and split at empty columns, which handles lightly touching digits. Each retained component is recentered on a 128×128 canvas so its framing matches the training images.

### 3.4 Deskewing

Deskewing estimates skew from second-order moments and rotates the image by `atan(mu11 / mu02)`, clipped to ±25 degrees. It is optional because the built-in digits are already normalized, but it is useful for photographed inputs.

### 3.5 Histogram of Oriented Gradients

For a 32×32 image, Sobel operators from OpenCV estimate horizontal and vertical gradients [8]. Magnitude and orientation are computed as:

`m = sqrt(gx² + gy²)`, `theta = atan2(gy, gx) mod 180°`.

The image is divided into 8×8 cells. Each cell produces a 9-bin orientation histogram. Blocks of 2×2 cells are normalized with L2-Hys style clipping, giving 3×3 blocks × 4 cells × 9 bins = **324 features** [5].

### 3.6 Classifiers and metrics

k-nearest neighbours votes over the five closest training samples. Logistic regression learns a linear decision boundary. The RBF SVM uses the kernel `K(x, z) = exp(-gamma * ||x - z||²)` with `gamma='scale'` and `C=10`. Features are standardized before fitting. Accuracy and macro-F1 measure overall correctness and balanced performance across the ten classes.

## 4. Proposed work

### 4.1 System architecture

The implemented pipeline is:

`form image → grayscale → denoise/Otsu → connected components → digit crops → 32×32 resize → HOG → StandardScaler → classifier → digit string`.

For a single isolated digit, the segmentation step returns one component. Training and inference use the same feature code, which prevents train–test inconsistency. The trained bundle stores the scaler, classifier, feature variant and preprocessing parameters in `models/digit_classifier.joblib`.

### 4.2 Dataset and split

The `load_digits()` loader returns 1,797 samples, ten classes and 8×8 grayscale images with integer intensities from 0 to 16 [13]. A stratified 80:20 split produces 1,437 training and 360 test images with `random_state=42`. This is an offline alternative to MNIST/EMNIST; the code can be extended by replacing the loader.

### 4.3 Feature variants

| Variant | Description | Dimensions |
|---|---|---:|
| `raw8` | Original 8×8 pixels, baseline | 64 |
| `gray32` | Grayscale image resized to 32×32 | 1,024 |
| `hog32` | HOG descriptor from 32×32 image | 324 |
| `hybrid` | 16×16 grayscale pixels + HOG | 580 |

Light normalization is used for training because aggressive thresholding degraded accuracy on these already-clean low-resolution digits. Stronger denoising, binarization, cropping and deskewing remain available for noisy photographs and are demonstrated in the generated pipeline figure.

### 4.4 Reproducibility

`src/train.py` trains every combination, writes `results/metrics.json`, saves the best classifier, creates the confusion matrix and sample predictions, and regenerates example images. `src/predict_form.py` segments and reads a form-style image. `tests/test_pipeline.py` and `tests/test_segmentation.py` verify preprocessing, features and segmentation. `run_project.bat` performs the full workflow in one step.

## 5. Experimental results

All configurations were evaluated on the same 360-image stratified test set. Table 1 reports accuracy and macro-F1.

**Table 1. Experimental comparison (test set, 360 images)**

| Features | Classifier | Accuracy | Macro-F1 | Features |
|---|---|---:|---:|---:|
| HOG32 | SVM (RBF) | **0.9889** | **0.9888** | 324 |
| HOG32 | k-NN (k=5) | 0.9889 | 0.9888 | 324 |
| Hybrid | k-NN (k=5) | 0.9889 | 0.9888 | 580 |
| Hybrid | Logistic Regression | 0.9889 | 0.9887 | 580 |
| Raw8 | SVM (RBF) | 0.9806 | 0.9805 | 64 |
| Gray32 | SVM (RBF) | 0.9778 | 0.9779 | 1,024 |
| Hybrid | SVM (RBF) | 0.9861 | 0.9860 | 580 |
| Raw8 | Logistic Regression | 0.9722 | 0.9719 | 64 |
| Gray32 | k-NN (k=5) | 0.9694 | 0.9692 | 1,024 |
| Raw8 | k-NN (k=5) | 0.9639 | 0.9634 | 64 |

The selected model was HOG32 with RBF SVM: accuracy 98.8889% (356/360 correct), macro precision 0.9890 and macro recall 0.9888. Only four test images were misclassified. The raw-pixel SVM achieved 98.06%, so HOG improved the result by about 0.83 percentage points.

**Table 2. Form segmentation demo**

| Test image | Expected | Segmented components | Predicted | Correct |
|---|---|---:|---|---|
| `data/samples/form_3014.png` | 3014 | 4 | 3014 | Yes |

Figure 1: preprocessing stages for a degraded sample (`results/preprocessing_pipeline.png`).

![Preprocessing pipeline](results/preprocessing_pipeline.png)

Figure 2: confusion matrix of the selected model (`results/confusion_matrix.png`).

![Confusion matrix](results/confusion_matrix.png)

Figure 3: connected-component boxes drawn around the four form digits (`results/segmentation_boxes.png`).

![Segmentation boxes](results/segmentation_boxes.png)

## 6. Conclusion

The completed prototype now matches the core workflow of Project 7. It preprocesses scanned images, removes noise, binarizes foreground, segments individual digits with connected components, and classifies each component using HOG features and a support vector machine. On the offline digits dataset, the selected model achieved 98.89% accuracy, and the sample form 3014 was segmented and read correctly as 3014. The strong classical baseline is easy to inspect and run on a normal laptop without a GPU, large dataset or internet access, which makes it suitable for a hands-on image-processing assignment. Character classes from EMNIST or IAM are not yet implemented, so the system is a digit-only prototype rather than a complete handwritten text recognizer. Future work should add character labels, stricter form-field validation, touching-digit splitting, data augmentation, a CNN comparison and cross-validation. Overall, the project provides a reproducible foundation for learning preprocessing, segmentation, feature extraction, classification and experimental reporting.

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. https://doi.org/10.1109/5.726791
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0
3. Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. *Biological Cybernetics, 36*(4), 193–202. https://doi.org/10.1007/BF00344251
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
5. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-34372-9
6. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
8. OpenCV developers. (2026). *OpenCV 4.x documentation*. https://docs.opencv.org/4.x/ (accessed 27 November 2026)
9. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems 25*, 1097–1105. https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
10. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
11. Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR 2015*. https://arxiv.org/abs/1409.1556
12. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778. https://doi.org/10.1109/CVPR.2016.90
13. Scikit-learn developers. (2026). `sklearn.datasets.load_digits` documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html (accessed 27 November 2026)
14. LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
15. Alpaydin, E., & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49
