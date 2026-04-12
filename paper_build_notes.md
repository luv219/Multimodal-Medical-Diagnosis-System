# Paper Build Notes

## Compile Instructions

```bash
pdflatex paper_ieee.tex
bibtex paper_ieee
pdflatex paper_ieee.tex
pdflatex paper_ieee.tex
```

Requires: `IEEEtran.cls` (standard IEEE LaTeX class), `IEEEtran.bst` (IEEE BibTeX style).

## Dependencies

- Standard LaTeX packages: `cite`, `amsmath`, `amssymb`, `amsfonts`, `graphicx`, `textcomp`, `xcolor`, `algorithmic`
- Additional packages: `booktabs` (table rules), `multirow` (table cells), `url` (URL formatting)
- All packages are included in standard TeX Live / MiKTeX distributions.

## Assumptions Made

1. **Dataset size**: 675 total samples stated based on `reflacx_with_clinical.csv` row count (674 data rows + header). The split is 80/10/10 (approx. 540/68/67 samples).

2. **Parameter count**: Stated as "approximately 8 million" based on DenseNet-121 having ~7.98M parameters plus the clinical MLP branch.

3. **Training epochs**: The 300-epoch results are used as the primary comparison point. The paper notes that best validation AUC was found at epoch 10, indicating most training beyond that is overfitting.

4. **Learning rate history**: The initial learning rate of 0.02 was derived from CheXNeXt; the paper states this was reduced to 0.0008 based on evidence from `100epoch_imageOnly.txt` (lr=0.02) vs. `withoutClinical.txt` (lr=0.0008).

5. **DeepAUC results**: Taken from `AUC_training_500epoch_with_image_only.txt` and `AUC_training_500epoch_with_clinica_2dropoutl.txt`. These files show 500-epoch runs but test metrics are reported at epoch 300 for the image-only variant.

6. **Data loss percentage**: The "approximately 30%" data loss from version incompatibility is stated in the README and carried into the paper as-is.

7. **Fusion comparison**: The paper states concatenation outperforms summation. Quantitative evidence comes from the 300-epoch result files. Qualitative Grad-CAM evidence is described from README documentation and notebook outputs.

## Placeholders and TODOs

| Item | Location in .tex | Action Required |
|------|-------------------|-----------------|
| Author names | `\author{}` block | Replace with actual author names, affiliations, and emails |
| Grad-CAM figure | `fig:gradcam` | Replace `Images/test.png` with exported comparison from `GradCAM++_apply.ipynb` |

## Known Issues

- The Grad-CAM comparison figure uses a placeholder. The actual comparison requires running the notebook to export a side-by-side visualization.
- Per-disease AUC values are not individually reported because the result files only provide aggregate AUC. Per-disease AUCs could be computed by re-running inference.
- The summation fusion model's exact test AUC is not separately reported in the result files; the paper discusses it qualitatively relative to concatenation.
