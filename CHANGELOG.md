# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `match-template` (optional, `pip install "parapick[leopard]"`, Python ≥ 3.12): 2D template
  matching of every tilt image with [Leopard-EM](https://github.com/Lucaslab-Berkeley/Leopard-EM),
  with the parameters of each tilt inferred from the acquisition. The low-pass follows the
  dose received before the tilt (Grant & Grigorieff, 2015), the angular and defocus steps the
  resolution left, and the defocus range the tilt of the sample. The dose and acquisition
  order come from `rlnMicrographPreExposure`, the CTF from `rlnDefocusU/V/Angle`, and the
  angle of the sample (pre-tilt included) and direction of its defocus gradient from the
  geometry. Writes `schedule.csv`, one Leopard-EM configuration and result per tilt, and the
  peaks as `picks.star` for `localize`. `--max-tilts` searches only the tilts with the highest
  predicted SNR; runs resume. `--binning N` searches the tilts binned by Fourier cropping
  (the template is binned too if given at the native pixel size; picks stay in native
  pixels). `--sampling fast` keeps the steps proportional to the resolution of each tilt,
  uncapped, with the fewest defocus planes covering the range. Peaks are kept down to one
  resolution element of each tilt apart, not the particle size, since particles overlap in
  projection. `--score mip` ranks peaks by the best correlation normalised over the image
  instead of Leopard-EM's z-score, whose normalisation over the orientations removes the
  signal of particles that correlate at every orientation (low-dose, binned tilts);
  `--peaks-per-tilt N` keeps the N best peaks of each tilt. Changing only the picking re-reads
  the results of the searched tilts, without searching again. The command is only
  registered when Leopard-EM is installed.
  In Python: `parapick.matching` (the schedule needs only numpy and scipy). Leopard-EM
  (Giammar et al., 2026) is credited in the README, `CITATION.cff`, the command's help and
  its log.
- Pick assignments: `localize --write-assignments` writes `assignments.star`, which lists the
  2D picks each particle explains (at most one per tilt), with their input row
  (`ppkPickIndex`), responsibility and residual. `--assignment-threshold` (default 0.5) sets
  the minimum responsibility. In Python: `LocalizationResult.assignments(selection)` and
  `parapick.gmm.assign_detections`.
- Particles carry a stable per-tilt-series id (`Particles.ids`, `ppkParticleId` in
  `particles.star`) that selection does not renumber.

### Changed
- `unstack` runs as an asynchronous pipeline on vitrio: several batches are read and written
  at once, and each file is finished with `close_async` as soon as it is written. New
  `--in-flight` option (`in_flight=` in Python, replacing `max_pending_writes=`); `--batch-size`
  now defaults to 4. Files left incomplete by a failed run are deleted.
- `VitrioImageIO` reads and writes asynchronously. The `ImageIO` interface gains `finish(path)`,
  which a file written with `write_batch` needs before it is complete.
- Stacks flagged as volumes are read section by section, no longer whole.
- Requires vitrio ≥ 0.2.0.

### Fixed
- Piping CLI output into a command that closes early (e.g. `parapick formats | head -1`) no
  longer logs `[Errno 32] Broken pipe`.

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
