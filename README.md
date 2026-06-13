# DIP_pj

This repository is a GitHub-friendly export of the local `dip_pj` workspace.
It now contains:

- code for the `ERRNet/` and `DSRNet/` subprojects
- released experiment summaries and qualitative comparison outputs
- a small set of directly tracked lightweight checkpoints for convenience

Large datasets, heavyweight checkpoints, and full raw result dumps are still
kept external. See [ASSET_MANIFEST.md](./ASSET_MANIFEST.md) for the exact
download-and-placement instructions.

Included subprojects:

- `ERRNet/`: ERRNet-based reflection removal experiments, ablations, RDNet integration, and RDNet-refiner code.
- `DSRNet/`: DSRNet-related training and evaluation code used in the same project workspace.

Important repository policy:

- benchmark-fair RDNet refiner results are included
- the non-fair `all_paired` / mixed-test-training RDNet refiner run is intentionally excluded

See each subproject's own `README.md` and released-results notes for setup,
usage, and result interpretation details.
