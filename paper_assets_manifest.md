# Paper Assets Manifest

## Figures Used

| Figure | Label | Source File | Status |
|--------|-------|-------------|--------|
| Architecture diagram | `fig:architecture` | `Images/MultiModalArchitecture-2.png` | **Confirmed** — 204 KB, exists in repo |
| Grad-CAM++ comparison | `fig:gradcam` | `Images/test.png` | **Placeholder** — using sample X-ray image; replace with exported Grad-CAM comparison from `GradCAM++_apply.ipynb` |

## Figures Recommended but Not Included

| Figure | Suggested Source | Action Needed |
|--------|-----------------|---------------|
| Training loss/AUC curves | `train.ipynb` outputs | Export training vs. validation loss/AUC plots as PNG |
| Radiologist comparison | `Images/different_radiologists.png` (5.2 MB) | Could be included but file is very large; crop or compress |
| Bounding ellipse overlay | `plot_ellipse_and_save.ipynb` | Export representative localization figure |
| Sum vs concat saliency side-by-side | `GradCAM++_apply.ipynb` | Export comparison showing corner artifacts (sum) vs clean maps (concat) |

## Tables

| Table | Label | Data Source |
|-------|-------|-------------|
| Data sources summary | `tab:datasources` | README.md and `reflacx_dataset_preprocessing.ipynb` |
| Model variant comparison | `tab:results` | `results/300epochResult/*.txt` and `results/100_epoch_results/*.txt` |
| Per-disease confusion matrix | `tab:perdisease` | `results/300epochResult/concatWithClinical.txt` |

## Notes

- All numerical results in the paper are taken directly from the repository result files.
- The Grad-CAM figure (`fig:gradcam`) currently uses a placeholder image (`Images/test.png`). To produce the intended figure, run `GradCAM++_apply.ipynb` and export a side-by-side comparison of summation vs. concatenation saliency maps.
- The architecture figure (`Images/MultiModalArchitecture-2.png`) exists and is correctly referenced.
