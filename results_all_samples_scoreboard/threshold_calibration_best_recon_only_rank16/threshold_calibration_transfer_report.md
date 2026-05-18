# Dev-Only Threshold and Calibration Transfer

Thresholds and calibrators in this report were selected or fit on public dev predictions only. all_samples labels were used only for final transfer evaluation.

## Low-FPR Transfer

- dev_target_fpr_1pct: all_samples FPR=0.020000, TPR=0.013333, F1=0.025806
- dev_target_fpr_5pct: all_samples FPR=0.053333, TPR=0.140000, F1=0.234637

## Calibration

- uncalibrated: all_samples ECE=0.179622, Brier=0.256891, log_loss=0.719899
- platt_sigmoid: all_samples ECE=0.140117, Brier=0.240694, log_loss=0.690266
- isotonic: all_samples ECE=0.123918, Brier=0.237655, log_loss=0.703742
