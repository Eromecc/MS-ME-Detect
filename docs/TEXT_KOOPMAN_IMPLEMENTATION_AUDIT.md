# Text-Koopman Implementation Audit

Two Text-Koopman variants are present:

1. Projection-based strict Text-Koopman:
   - hidden-state cache
   - train-only projection to 128/256
   - learned lifting
   - per-document local DMD
   - spectral-only classifier

2. Strict mathematical Text-Koopman:
   - Qwen hidden-state token trajectory
   - direct `hidden_size -> observable_dim` lifting
   - no PCA/random projection
   - no pooled-z classifier
   - no global shared K
   - per-document truncated exact-DMD `K_i`
   - spectral-only classifier

The strict mathematical variant is the most theory-aligned implementation.  Its
small-validation audit passed with:

```text
hidden_size=1536
observable_dim=3072
uses_projection=false
uses_label_classifier=false
uses_global_K_parameter=false
strict_math_failed_due_to_memory=false
```

It validates the intended mathematical pipeline, but it is not selected as the
main detector because it does not beat transition-state profiling on external
AUROC/AUPRC.
