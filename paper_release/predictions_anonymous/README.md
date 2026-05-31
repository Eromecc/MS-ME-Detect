# Anonymous Prediction Scores

`all_samples_predictions_hash_only.csv` contains only hash-only prediction rows:

```text
sample_hash,label,score_final,score_no_segment,score_qwen14_segment,pred_final_0_5
```

Rules for this directory:

- `sample_hash` is SHA-256 over whitespace-normalized text from the local source prediction file.
- Raw text is not included.
- Original file paths are not included.
- Original sample IDs are not included.
- Personal information or source-specific raw IDs must not be added.

The generated file in this release was derived locally from:

- `artifacts/results/results_incremental_embedding_best_plus_embedding_segment_qwen14_v1/val_composite__all_samples_predictions.csv`
- `results_incremental_current_best_plus_embedding_pool_raw_plateau_v1/val_low_fpr__all_samples_predictions.csv`
- `artifacts/results/results_embedding_segment_qwen14_probe_clean_v1/predictions/embedding_segment_qwen14_probe__sgd_a1e4__all_samples.csv`

Those source files contain raw text and must not be committed.
