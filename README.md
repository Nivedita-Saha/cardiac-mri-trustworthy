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

