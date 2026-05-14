# Analysis Outputs

This folder stores feature-analysis experiments used for the paper draft.

## Directory layout

- `cross_dataset/`
  - `multi_dataset_domain_separability/`
    - Cross-dataset analysis across multiple underwater datasets.
    - Main outputs:
      - `tsne_degradation_features.png`
      - `tsne_water_domain_features.png`
      - `domain_separability_metrics.csv`
      - `domain_separability_summary.csv`

- `within_dataset/`
  - `lsui_natural_clustering/`
    - Within-dataset clustering analysis on LSUI.
    - Main outputs:
      - `tsne_degradation_features.png`
      - `tsne_water_domain_features.png`
      - `within_dataset_cluster_metrics.csv`
      - `cluster_assignments.csv`

- `prior_coupling/`
  - `uieb_attention_heatmap/`
    - Prior-coupling attention analysis for the physics-coupled model.
    - Main outputs:
      - `prior_attention_mean_heatmap.png`
      - `prior_attention_std_heatmap.png`
      - `prior_attention_mean.csv`
      - `prior_attention_std.csv`
      - `prior_attention_summary.txt`

- `legacy/`
  - `cross_dataset_v1/`
    - Earlier cross-dataset experiment kept for reference only.

## Notes

- `sample_paths.txt` records which images were sampled.
- `degradation_features.npy` and `water_domain_features.npy` store extracted features.
- `labels.npy` appears only in cross-dataset analysis because it uses dataset labels.
- `prior_attention_matrices.npy` stores per-image `4x4` prior-attention matrices.
