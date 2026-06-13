# Released Results

This directory contains compact summary artifacts from the standalone DSRNet
deployment used as an external comparison model.

Included here:

- full-resolution tiled evaluation summary on the ERRNet-processed benchmarks
- `max_long_edge=512` evaluation summary on the same benchmark suite
- per-dataset `summary.txt` and `per_image_metrics.csv` exports for both
  evaluation modes
- the saved training loss log from the local fine-tuning run
- the lightweight checkpoint `DSRNet/weights/dsrnet_s_epoch14.pt`

Large generated image dumps and datasets are intentionally excluded from this
GitHub export.
