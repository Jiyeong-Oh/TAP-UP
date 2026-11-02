# TAP-UP

Tumor-Anchored Fusion of Paired MRI and PSMA-PET via Unpaired Pretraining for Prostate Cancer Recurrence Prediction.

Code for the paper. TAP-UP predicts biochemical recurrence (BCR) after radical prostatectomy from paired pre-operative
MRI (T2W, ADC) and PSMA-PET. It pretrains one encoder per modality on unpaired data, fuses the frozen encoders with
cross-modal attention and tumor-anchored pooling, and explains each prediction with integrated gradients and Shapley values.

The paired patient cohort is not public, so patient data, labels, splits and trained models are not included.
Public datasets must be downloaded from their own sources. Result tables from the paper are in `results/`.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/wangshansong1/Triad.git external_repos/Triad
git -C external_repos/Triad checkout d05cc1d
git -C external_repos/Triad apply ../../external/triad_in_chans.patch
```

Place the Triad Swin-B weights at `checkpoints_pretrained/Triad-SwinB-MAE.pth`. The comparison methods also need
`model_swinvit.pt` (Swin UNETR) and `mass_base.pth` with the MASS code in `external_repos/MASS`.

## Data

Set these environment variables, or use the defaults under `datasets/`:

| Variable | Contents |
|---|---|
| `TAPUP_INTERNAL` | `<id>/{t2w,adc,pet,gland,tumor}.nii.gz`, co-registered |
| `TAPUP_PICAI` | `<id>/<id>_{t2w,adc_reg,gland,tumor}.nii.gz` |
| `TAPUP_P158_TRAIN`, `TAPUP_P158_TEST` | Prostate158 folders |
| `TAPUP_DEEPPSMA` | `<case>/PSMA/{PET,totseg_24}.nii.gz` |
| `TAPUP_CACHE` | Output folder for preprocessed volumes |

Create these files in `data/`:

| File | Columns |
|---|---|
| `cohort.csv` | `patient_id, bcr_label_binary, cohort` |
| `clinical.csv` | `patient_id, age, psa, volume, isup_gg, bcr_sheet, gleason_sum, ece_epe, lni, g1, g2, stage` |
| `internal_unlabeled.csv` (optional) | `patient_id` of unlabeled patients for pretraining |

`g1`, `g2` and `stage` (prostatectomy Gleason patterns and clinical stage) are used only by the radiomics comparison.

## Reproducing the paper

```bash
# Preprocessing and split
python step1/preprocess_cohort.py
python step2/preprocess_bcr_cohort.py
python step2/build_folds.py

# Stage 1: modality-specific pretraining
python step1/ssl_train_triad.py --modality t2 --epochs 100
python step1/ssl_train_triad.py --modality adc --epochs 100
python step1/ssl_train_triad.py --modality pet --epochs 200

# Stage 2: TAP-UP
python step2/train_bcr_kfold.py --tag tapup --freeze_lora --tumor_pool_weight 0.9 --aux_clinical \
    --rotate_aug --l2_weight 0.01 --save_checkpoints
python step2/threshold_analysis.py tapup

# Interpretation
for f in 1 2 3; do python step3/export_saliency.py --tag tapup --fold $f; done
python step3/deletion_test.py --tag tapup --out_dir outputs/tapup/deletion
python step3/randomization_test.py --tag tapup --out_dir outputs/tapup/randomization
python step3/saliency_overlap.py --tag tapup
python step3/modality_shapley.py --tag tapup
python step3/make_shapley_bars.py --tag tapup
python step3/make_paper_saliency_figure.py --tag tapup --pids "<id>:<row label>" --out outputs/tapup/fig2.png
```

Ablations use the Stage 2 command with one change:

| Variant | Change |
|---|---|
| Modality subsets | `--modalities t2,adc` (or another pair); single modalities also need `--attn_mode joint` |
| Concatenation | `--fusion_type concat_mlp` |
| No tumor-anchored pooling | remove `--tumor_pool_weight` |
| No positional encoding | `--no_pos_encoding` |
| Triad weights | `--no_ssl` |
| No auxiliary targets | remove `--aux_clinical` |

Comparison methods are in `step2/train_baseline_backbones.py`, `step2/papp_radiomics_baseline.py` and
`step2/clinical_baseline_preop.py`. Triad comparisons use `step2/train_bcr_kfold.py` with `--no_ssl`,
`--fusion_type concat_mlp` and either `--linear_probe` or LoRA.
