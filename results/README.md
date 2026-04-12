# Results

`results/per_id_v2/` contains the final report-ready outputs from the corrected per-ID MLP autoencoder pipeline:

- `*_all_ids_summary.csv`: per-machine summary tables exported from `src.evaluate`.
- `*_mlp_loss.png`: training and validation loss curves.
- `*_mlp_roc.png`: ROC curves with AUC.
- `*_mlp_score_dist.png`: normal vs anomalous reconstruction-error distributions.

The root-level loose `results/*.png` files are older scratch outputs and are ignored by git. Use `results/per_id_v2/` for the final report/demo.
