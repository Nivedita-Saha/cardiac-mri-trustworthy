# Trustworthy Cardiac MRI: From Segmentation to Explainable Diagnosis

A cardiac MRI pipeline that segments the heart, derives clinical biomarkers from
those segmentations, and produces an explainable diagnosis. The project focuses
on three properties that matter for clinical deployment: interpretability,
uncertainty awareness, and generalisation across scanners.

## Motivation

Many cardiac MRI models report strong segmentation scores but stop short of
clinical usefulness. They rarely explain their predictions, rarely signal when
they are uncertain, and often degrade when applied to scans from a different
hospital or vendor. This project addresses all three.

## Pipeline

1. **Segmentation.** A 2D U-Net segments the left ventricle, myocardium, and
   right ventricle on the ACDC dataset.
2. **Biomarker extraction.** Predicted masks at end diastole and end systole are
   converted into clinical measurements: ejection fractions, ventricular volumes,
   myocardial mass, and wall thickness.
3. **Explainable diagnosis.** A classifier predicts the diagnosis group from
   those biomarkers, with SHAP used to show which measurements drive each
   decision.
4. **Uncertainty.** Monte Carlo dropout flags low confidence segmentations.
5. **Cross scanner generalisation.** The ACDC trained model is evaluated
   zero shot on the multi vendor M&Ms dataset to quantify domain shift, followed
   by a mitigation experiment.

## Datasets

| Dataset | Use | Access |
|---|---|---|
| ACDC (Automated Cardiac Diagnosis Challenge, MICCAI 2017) | Primary training and evaluation | Free, registration required |
| M&Ms (Multi Centre, Multi Vendor, Multi Disease) | Generalisation study only | Free, registration required |

Imaging data is not included in this repository and is excluded via `.gitignore`.

## Repository structure                                                                                                                                                    data/raw/          Original downloaded datasets (not tracked)
data/processed/    Preprocessed slices (not tracked)
notebooks/         Exploration and analysis notebooks
src/data/          Loading and preprocessing
src/models/        Model definitions and training
src/evaluation/    Metrics, SHAP, uncertainty
results/figures/   Output figures
results/metrics/   Numerical results                                                                                                                                                                                                                                                                                                            ## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Work in progress. See the project plan for phase by phase progress.

## References

Bernard, O. et al. (2018) 'Deep learning techniques for automatic MRI cardiac
multi-structures segmentation and diagnosis: is the problem solved?', IEEE
Transactions on Medical Imaging, 37(11), pp. 2514-2525.

Campello, V.M. et al. (2021) 'Multi-centre, multi-vendor and multi-disease
cardiac segmentation: the M&Ms challenge', IEEE Transactions on Medical Imaging,
40(12), pp. 3543-3554.

Ronneberger, O., Fischer, P. and Brox, T. (2015) 'U-Net: convolutional networks
for biomedical image segmentation', in Medical Image Computing and Computer
Assisted Intervention (MICCAI 2015). Cham: Springer, pp. 234-241.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model
predictions', Advances in Neural Information Processing Systems, 30,
pp. 4765-4774.


## Dataset characteristics

Profiled across all 100 ACDC training patients (see
`results/metrics/dataset_summary.csv`):

| Property | Range | Implication |
|---|---|---|
| Diagnosis groups | 20 patients each across 5 groups | Balanced, no class weighting required |
| Image dimensions | 154-428 x 154-512 | Crop and pad to a fixed 256 x 256 |
| Slices per volume | 6 to 18 | Build the dataset as a flat slice list |
| In-plane spacing | 0.70 to 1.92 mm | Resample to a common 1.5 mm |
| Slice thickness | 5 to 10 mm | Strongly anisotropic, motivating 2D over 3D |
| Intensity maximum | 184 to 4025 | Per-image normalisation required |

The 2.7-fold variation in in-plane spacing and the 20-fold variation in
intensity maxima are the two properties that most directly shape the
preprocessing pipeline.

## Preprocessing

Each decision follows from the measured dataset properties above.

| Step | Choice | Reason |
|---|---|---|
| Resampling | 1.5 mm in-plane, bilinear for images, nearest for masks | Measured spacing varies 2.7-fold across patients |
| Normalisation | Per-slice, 1st-99th percentile clip then z-score | MRI intensities have no absolute scale; maxima vary 20-fold |
| Spatial size | Centre crop or zero pad to 256 x 256 | Image dimensions vary from 154 to 512 |
| Splits | Patient level, stratified by diagnosis group, 70/15/15 | Slices from one heart are correlated, so slice-level splitting would leak |
| Caching | Preprocessed once to compressed float16 archives | Avoids repeated decompression and produces a compact artefact for GPU training |
| Augmentation | Rotation, zoom, flips, intensity scale and shift | Geometric transforms applied identically to image and mask |

Two correctness checks are included rather than assumed. `crop_retention`
confirms that centre cropping removes no labelled pixels for any patient,
and the augmentation pipeline was verified to keep image and mask spatially
aligned. Splits are validated to contain no patient overlap.

Resulting dataset: 1356 training, 284 validation, and 262 test slices.

## Segmentation results

A 2D U-Net (6.5M parameters, MONAI) trained for 40 epochs with a combined
Dice and cross entropy loss, AdamW, and a cosine learning rate schedule.
Metrics are computed per volume after reassembling slices into patient and
phase volumes, so they are directly comparable with published ACDC results.

| Structure | Dice | HD95 (mm) | nnU-Net Dice |
|---|---|---|---|
| Right ventricle | 0.833 +/- 0.102 | 10.08 | 0.906 |
| Myocardium | 0.862 +/- 0.044 | 6.87 | 0.902 |
| Left ventricle cavity | 0.931 +/- 0.050 | 4.44 | 0.943 |
| Mean | 0.875 +/- 0.050 | | 0.917 |

nnU-Net figures are the published benchmark (Isensee et al., 2021) and are
included for context, not as a like for like comparison: that system uses
extensive automated configuration and ensembling.

### Post-processing

Keeping only the largest connected component per structure, motivated by the
fact that a heart contains exactly one of each, improved both metrics:

| Structure | Dice before | Dice after | HD95 before | HD95 after |
|---|---|---|---|---|
| Right ventricle | 0.819 | 0.833 | 27.33 | 10.08 |
| Myocardium | 0.859 | 0.862 | 10.66 | 6.87 |
| Left ventricle cavity | 0.916 | 0.931 | 11.16 | 4.44 |

The large Hausdorff reduction reflects small disconnected false positives far
from the heart. Dice is relatively insensitive to these, whereas Hausdorff
distance is dominated by them, which is why both metrics are reported.

### Performance by diagnosis group

DCM 0.894, HCM 0.881, RV 0.874, NOR 0.870, MINF 0.857. Performance is
consistent across pathologies, with myocardial infarction hardest.
