# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-27

### Added
- Pick formats `cryolo-cbox` (crYOLO CBOX, STAR and legacy, with confidences), `eman-box`
  (EMAN `.box`), `cryosparc` (`.cs` picker output) and `xmipp-pos` (Xmipp/Scipion `.pos`,
  read and write). They share a per-micrograph base (`formats/per_micrograph.py`), onto
  which `relion-coords` was refactored.
- `--flip-y` for `localize` and `convert-picks`, for pickers with the opposite y axis.
- `parapick unstack`: writes one 2D `.mrc` per tilt image, because most SPA pickers (crYOLO,
  Topaz in 2D mode) need one micrograph per file. It also writes matching geometry, so picks
  on the unstacked images localise directly.
- Image I/O through [vitrio](https://pypi.org/project/vitrio/) (MRC, TIFF, EER, DM), behind a
  batch `ImageIO` interface that returns pending results. It is synchronous for now and ready
  for asynchronous implementations. `mrcfile` is no longer a runtime dependency.
- The RELION 5 reader supports the stacked layout (`rlnTomoTiltSeriesName`, with or without
  per-tilt `rlnMicrographName`). `TiltSeries.images` separates where each tilt's pixels are
  from how the tilt is identified.
- First implementation of ParaPick: 3D particle localization from 2D tilt-series picks via a
  projected Gaussian mixture fitted by EM (capped Sinkhorn E-step, Dirichlet MAP pruning),
  written in JAX (CPU/GPU).
- Robustness components (each can be disabled): uniform background term, deterministic
  annealing, lifted initialisation, duplicate merging.
- Format-neutral data model, with format plug-ins: RELION 5 (tomograms, picks, particles,
  optimisation set), per-micrograph RELION coordinates, Topaz. Third-party formats register
  through the `parapick.formats` entry-point group.
- `parapick` CLI with the sub-commands `localize`, `convert-picks`, `project`, `filter`,
  `evaluate`, `simulate` and `formats`, plus a Python API.

[Unreleased]: https://github.com/oierlauzi/parapick/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/oierlauzi/parapick/releases/tag/v0.1.0
