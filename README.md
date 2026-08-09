# Action-recognition-using-Vision-transformers

## Overview

This project investigates video action recognition using three
Vision Transformer (ViT)-based architectures:

- TimeSFormer
- VideoMAE
- ViViT

The project compares their classification performance, computational
cost, robustness, and behaviour under controlled ablation studies.

A spatio-temporal localisation pipeline is also developed using a
frozen VideoMAE backbone and evaluated using genuine per-frame
ground-truth annotations from JHMDB.

---

## Objectives

The main objectives of this project are:

1. Compare TimeSFormer, VideoMAE and ViViT for video action recognition.
2. Evaluate classification performance using Top-1 accuracy, Top-5 accuracy
   and F1 score.
3. Compare the number of model parameters and training time per epoch.
4. Investigate the effect of input frame count.
5. Investigate different fine-tuning strategies.
6. Compare CLS-token and mean-token pooling strategies.
7. Evaluate robustness using independent training seeds.
8. Develop and evaluate a spatio-temporal localisation pipeline.
9. Analyse model errors and learned representations using visualisation
   and interpretability techniques.

---

## Models

### TimeSFormer

TimeSFormer separates spatial and temporal attention and was evaluated
using an 8-frame input configuration.

### VideoMAE

VideoMAE uses masked video-token reconstruction during self-supervised
pre-training. In the main classification experiment, VideoMAE used
16 frames and mean pooling.

### ViViT

ViViT factorises spatial and temporal processing and was evaluated using
32 frames.

---

## Datasets

### HMDB_simp

The classification experiments use the HMDB_simp benchmark containing:

- 25 action classes
- 1,250 video clips
- 70/15/15 train/validation/test split
- 875 training clips
- 187 validation clips
- 188 test clips

### JHMDB

JHMDB is used for spatio-temporal localisation.

The localisation pipeline uses genuine per-frame joint annotations
to derive bounding boxes rather than a fixed placeholder ground-truth
box.

---

## Classification Results

The main classification results are:

| Model | Top-1 (%) | Top-5 (%) | F1 | Parameters | Seconds/Epoch |
|---|---:|---:|---:|---:|---:|
| TimeSFormer | 85.64 | 97.87 | 0.847 | 121.3M | 21.7 |
| VideoMAE | **89.89** | **98.40** | **0.895** | **86.2M** | 29.9 |
| ViViT | 86.17 | 97.34 | 0.851 | 89.3M | 111.7 |

VideoMAE achieved the highest Top-1 accuracy, Top-5 accuracy and F1
score while also having the fewest parameters among the three models.

TimeSFormer had the lowest training time per epoch.

ViViT required substantially more computation per epoch without providing
a comparable accuracy improvement.

---

## Ablation Studies

Three controlled ablation studies were performed.

### 1. Frame Count

TimeSFormer was evaluated using:

- 4 frames
- 8 frames
- 16 frames

Validation accuracy:

| Frames | Validation Accuracy |
|---:|---:|
| 4 | 80.75% |
| 8 | 85.03% |
| 16 | 87.70% |

Increasing the number of input frames improved validation accuracy
under the tested conditions.

### 2. Fine-Tuning Strategy

TimeSFormer was evaluated using:

- Head-only fine-tuning
- Partial fine-tuning of the last 4 layers
- Full fine-tuning

Validation accuracy:

| Strategy | Validation Accuracy |
|---|---:|
| Head-only | 80.21% |
| Partial | **91.98%** |
| Full | 88.24% |

Partial fine-tuning produced the best result in this experiment.

### 3. Pooling Strategy

VideoMAE was evaluated using:

- CLS-token pooling
- Mean pooling

Validation accuracy:

| Pooling | Validation Accuracy |
|---|---:|
| CLS | 91.98% |
| Mean | 91.44% |

The difference between the two pooling strategies was relatively small
under the controlled 5-epoch experiment.

---

## Robustness Evaluation

VideoMAE was independently retrained using three different seeds.

Top-1 accuracy:

- Seed 42: 87.77%
- Seed 123: 86.70%
- Seed 2024: 91.49%

Mean Top-1 accuracy:

**88.65% ± 2.05%**

Mean Top-5 accuracy:

**97.52% ± 0.66%**

This experiment was used to assess whether the classification performance
was dependent on a particular random initialisation.

---

## Spatio-Temporal Localisation

A lightweight frame-wise localisation head was developed using a frozen
VideoMAE backbone.

The localisation experiment was evaluated on the JHMDB test set.

Results:

| Metric | Result |
|---|---:|
| Frame-level mAP@0.5 | **25.04%** |
| Video-level tube mAP@0.5 | **24.57%** |

The localisation ground truth was derived from genuine per-frame joint
annotations.

This replaces an earlier temporally static placeholder ground-truth box
and provides a more realistic evaluation of spatial localisation.

---

## Interpretability and Error Analysis

The project includes several forms of analysis:

- Confusion matrices
- Attention visualisation
- t-SNE feature visualisation
- Correctly classified examples
- Misclassified examples
- Localisation success cases
- Localisation failure cases
- Video-property correlation analysis
- Classification versus localisation correlation

The analysis found a weak positive correlation between classification F1
and localisation frame-AP:

**Pearson r = 0.247**

This indicates that strong whole-video classification does not necessarily
result in strong frame-level spatial localisation.

---

## Repository Structure

```text
Action-recognition-using-Vision-transformers/
│
├── figures/
│   ├── attention_overlay.png
│   ├── confusion_matrices_3models.png
│   ├── epoch_convergence_plot.png
│   ├── error_correct.png
│   ├── error_incorrect.png
│   ├── f1_vs_ap_correlation.png
│   ├── localisation_failures.png
│   ├── localisation_qualitative.png
│   ├── localisation_successes.png
│   ├── tsne_videomae.png
│   └── video_property_correlation.png
│
├── logs/
│   ├── ablation_summary.csv
│   ├── computational_cost_comparison.csv
│   ├── epoch_convergence_ablation.csv
│   ├── part_a_comparison.csv
│   ├── part_a_full_comparison.csv
│   ├── robustness.csv
│   ├── robustness_independent.csv
│   ├── jhmdb_per_class_ap.csv
│   ├── per_class_prf_all_models.csv
│   ├── video_property_analysis.csv
│   ├── confused_pairs_timesformer.csv
│   ├── confused_pairs_videomae.csv
│   └── confused_pairs_vivit.csv
│
├── report/
│   └── EEEM068_report.pdf
│
├── VISION_TRANSFORMERS_.ipynb
├── requirements.txt
├── README.md
└── .gitignore
