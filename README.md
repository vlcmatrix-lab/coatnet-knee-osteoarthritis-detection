# An Efficient Arthritis Diagnostic Model using CoAtNet by Integrating Convolutional and Attention Mechanisms for Enhanced Precision

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICIC63915.2024.11116318-blue)](https://doi.org/10.1109/ICIC63915.2024.11116318)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Ahmad Shehzad, Faheem Akbar, Tanzila Kehkashan, Muhammad Mohsin, Muhammad Zaman, Abdul Rehman Shoukat**

*2024 5th International Conference on Innovative Computing (ICIC), IEEE, 2024*

[Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/11116318) | [Author's accepted version (PDF)](paper/coatnet_knee_osteoarthritis_icic2024.pdf)

This repository contains the official code for the paper.

## Summary

The paper grades the severity of knee osteoarthritis (OA) from X-ray images using CoAtNet, a hybrid model that combines convolutional layers with self-attention. The convolutional layers pick up local detail such as edges and textures in the joint, and the attention layers relate distant parts of the image to each other. Each X-ray is classified into one of the five Kellgren-Lawrence (KL) grades, from grade 0 (no OA) to grade 4 (severe OA). Images are resized, normalized and augmented before training.

The paper reports 96% accuracy, an F1-score of 0.88, a precision of 0.87 and a recall of 0.86, with an AUC of 0.97.

## Repository structure

```
coatnet-knee-osteoarthritis-detection/
├── notebooks/
│   └── coatnet_knee_osteoarthritis_detection.ipynb   # main entry point: full pipeline
├── paper/
│   └── coatnet_knee_osteoarthritis_icic2024.pdf      # author's accepted version
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

The notebook is designed for Kaggle or Google Colab with a GPU (NVIDIA T4 or better). To run it locally instead, use Python 3.11:

```bash
git clone https://github.com/vlcmatrix-lab/coatnet-knee-osteoarthritis-detection.git
cd coatnet-knee-osteoarthritis-detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

CoAtNet is loaded with ImageNet-pretrained weights through PyTorch `timm` (`coatnet_0_rw_224`).

## Dataset

The paper uses the **Kaggle Knee Osteoarthritis Dataset**: knee X-ray images graded by the Kellgren-Lawrence system into five classes (grade 0: no OA, grade 1: probable OA, grade 2: minimal OA, grade 3: moderate OA, grade 4: severe OA). The images are split 60 / 20 / 20 into 4800 training, 1600 validation and 1600 test images.

- Official source: [Kaggle: Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

No data is included in this repository. Please follow the dataset's license terms. The notebook finds the KL grade folders (`0` to `4`) automatically:

- **Kaggle:** attach the dataset to the notebook; it is read from `/kaggle/input`.
- **Colab or local:** the dataset is downloaded with `kagglehub`, which needs your Kaggle API credentials. Alternatively, set `DATA_DIR` in the configuration cell to a folder you downloaded yourself. Never commit `kaggle.json` or other credentials.

## Usage

```bash
jupyter notebook notebooks/coatnet_knee_osteoarthritis_detection.ipynb
```

Run all cells in order. The notebook covers:

1. Collecting the images by KL grade and removing duplicates
2. Stratified 60 / 20 / 20 train / validation / test split (4800 / 1600 / 1600 images)
3. Preprocessing and augmentation (resizing to 224x224, normalization, rotation, horizontal and vertical flips, zoom, shift)
4. Training CoAtNet with AdamW (learning rate 0.001), batch size 32, for up to 50 epochs with cross-entropy loss
5. Training and validation curves, test-set metrics, confusion matrix and per-class ROC curves
6. Comparison with other models (Table I)
7. Grad-CAM heatmaps showing which knee regions drive each prediction

Figures, metrics and the best model checkpoint are saved to `notebooks/outputs/` (created automatically and excluded from git). The notebook is distributed without saved outputs.

## Results

All values below are reported in the paper.

**Comparison with state-of-the-art models (Table I)**

| Model | Accuracy (%) | F1-score | Precision | Recall |
|---|---|---|---|---|
| **CoAtNet (ours)** | **96.0** | **0.88** | **0.87** | **0.86** |
| DenseNet-121 | 77.6 | 0.78 | 0.79 | 0.77 |
| EfficientNet | 84.3 | 0.83 | 0.84 | 0.82 |
| ViT | 85.0 | 0.86 | 0.85 | 0.85 |
| ResNet-50 | 74.4 | 0.75 | 0.77 | 0.76 |

The paper also reports an AUC of 0.97 for CoAtNet (Fig. 5).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{shehzad2024efficient,
  author    = {Shehzad, Ahmad and Akbar, Faheem and Kehkashan, Tanzila and Mohsin, Muhammad and Zaman, Muhammad and Shoukat, Abdul Rehman},
  title     = {An Efficient Arthritis Diagnostic Model using {CoAtNet} by Integrating Convolutional and Attention Mechanisms for Enhanced Precision},
  booktitle = {2024 5th International Conference on Innovative Computing (ICIC)},
  publisher = {IEEE},
  year      = {2024},
  doi       = {10.1109/ICIC63915.2024.11116318}
}
```

## Copyright notice

© 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. The final version is published at DOI: [10.1109/ICIC63915.2024.11116318](https://doi.org/10.1109/ICIC63915.2024.11116318).

The PDF in `paper/` is the authors' accepted version, not the publisher's typeset version.

## License

The code in this repository is released under the [MIT License](LICENSE). The copyright notice above applies to the paper.
