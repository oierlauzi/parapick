# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- First implementation of ParaPick: 3D particle localization from 2D tilt-series picks via a
  projected Gaussian mixture fitted by EM (capped Sinkhorn E-step, Dirichlet MAP pruning),
  written in JAX (CPU/GPU).
- Robustness extensions (each can be disabled): uniform background term, deterministic
  annealing, lifted initialisation, duplicate merging.
- Format-neutral data model, with format plug-ins: RELION 5 (tomograms, picks, particles,
  optimisation set), per-micrograph RELION coordinates, Topaz. Third-party formats register
  through the `parapick.formats` entry-point group.
- `parapick` CLI with the sub-commands `localize`, `convert-picks`, `project`, `filter`,
  `evaluate`, `simulate` and `formats`, plus a Python API.
