# Figure Source Data

This directory contains small plotting tables for the paper-release snapshot.

- `figure2_source_data.csv`: available feature-complementarity source rows from the local Figure 2 source table.
- `figure3_source_data.csv`: external benchmark metrics for Figure 3, matching `tables/table1_external_metrics.csv`.
- `figure4_source_data.csv`: available Qwen14 segment/fusion source rows plus explicit TODO rows for design-level ablations that were not found.

Heavy intermediate matrices are excluded. Missing source-data rows must be regenerated from local caches and summarized into small CSVs before upload. Do not commit raw text, embedding matrices, hidden-state caches, token-loss caches, full feature matrices, checkpoints, or model weights.
