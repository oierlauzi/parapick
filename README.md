# ParaPick

[![CI](https://github.com/oierlauzi/parapick/actions/workflows/ci.yml/badge.svg)](https://github.com/oierlauzi/parapick/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tomogram-free particle localization for cryo-ET.** ParaPick finds the 3D coordinates of
particles directly from 2D picks made independently on each image of a tilt-series. You can
pick with any single-particle picker (Topaz, crYOLO, RELION, ...). ParaPick fuses the picks
from all tilts into a consistent set of 3D positions, with a confidence score for each one.
No tomogram needs to be reconstructed.

## How it works

Each particle is a point in 3D whose projection should show up as a pick on every tilt image.
ParaPick starts with more candidates than there are particles, and repeats two steps:

1. **Match:** each pick is matched to the candidates that project close to it. A candidate
   takes at most one pick per tilt, and picks that match no candidate are set aside as false
   positives.
2. **Move:** each candidate moves to the 3D position that best agrees with its picks.

Candidates that too few picks support are dropped, so the number of particles is found
automatically. A particle's score is the fraction of tilts whose picks support it.

The method is a Gaussian mixture fitted by expectation-maximisation.
[`docs/algorithm.md`](docs/algorithm.md) gives the details and how they map to the code.

## Features

- **Any 2D picker:** reads picks from RELION, crYOLO, Topaz, cryoSPARC, EMAN2 and
  Xmipp/Scipion. `unstack` splits tilt-series stacks into one micrograph per tilt for pickers
  that need it.
- **RELION 5 in and out:** the geometry comes from `tomograms.star`, and the particles feed
  straight into subtomogram averaging.
- **Many tilt-series per run,** on CPU or GPU (JAX).
- **Command line and Python API,** with file formats as plug-ins.

## Installation

```bash
pip install parapick                # CPU
pip install "parapick[cuda12]"      # NVIDIA GPU (CUDA 12); use [cuda13] for CUDA 13
```

To install from source, use [uv](https://docs.astral.sh/uv/): run `uv sync` in a clone.
Python ≥ 3.11 is required.

## Quick start (command line)

```bash
# 1. Pick each tilt image with your favourite 2D picker, then gather the picks into one
#    STAR file. Matching them to the tilts of tomograms.star adds tilt-series names.
parapick convert-picks topaz_picks.txt --tomograms tomograms.star -o picks.star

# 2. Localise the particles in 3D (all tilt-series at once).
parapick localize --tomograms tomograms.star --picks picks.star -o localize/

# 3. Keep the confident ones (score = fraction of tilts supporting the particle).
parapick filter localize/particles.star --min-score 0.3 --tomograms tomograms.star \
    -o localize/particles_selected.star
```

`localize/` then contains:
- `particles.star`: RELION 5 `rlnCenteredCoordinate{X,Y,Z}Angst`, with the score in
  `rlnAutopickFigureOfMerit`.
- `optimisation_set.star`: ready for RELION 5 subtomogram extraction.
- `localize_report.json`: per tilt-series diagnostics, such as iterations, number of
  particles, background fraction and per-tilt pick σ.

Run `parapick --help` and `parapick <command> --help` for all options.

### Picking on stacked tilt-series

Most single-particle pickers need one micrograph per file. crYOLO rejects MRC files with more
than one section, and Topaz (2D) treats each file as one micrograph. Tilt-series, however,
often come as one stack per tilt-series (IMOD, AreTomo, Warp, RELION's IMOD import).
`unstack` writes each tilt as its own 2D `.mrc` file, together with a copy of the geometry
that points at them:

```bash
parapick unstack --tomograms tomograms.star -o unstacked/
# pick on unstacked/micrographs/*.mrc (files named <TS>_<NNN>.mrc), e.g. with Topaz or crYOLO
parapick convert-picks topaz_picks.txt --tomograms unstacked/tomograms.star -o picks.star
parapick localize --tomograms unstacked/tomograms.star --picks picks.star -o localize/
```

The images keep their native sampling, so pick coordinates need no conversion. The new
`tomograms.star` has the same geometry and RELION metadata, so it can replace the original
downstream. `--link` symlinks tilts that are already single `.mrc` files instead of copying
them.

| Command         | Purpose |
|-----------------|---------|
| `localize`      | **Main task:** 3D coordinates from 2D picks on the tilt images. |
| `unstack`       | Write each tilt image as its own 2D `.mrc` file (for SPA pickers), plus matching geometry. |
| `convert-picks` | Convert picks between formats, resolving them against the tilt-series. |
| `project`       | Project 3D particles onto every tilt image, e.g. ground truth to train a 2D picker, or QC. |
| `filter`        | Select particles by score, support or rank. |
| `evaluate`      | Precision, recall, F1 and RMSE against a reference set. |
| `simulate`      | Generate a synthetic dataset (geometry, ground truth, noisy picks). |
| `formats`       | List the available file formats, including installed plug-ins. |

Common options: `--device {auto,cpu,gpu}`, `--x64`, `-v`/`-q`.

## Quick start (Python)

```python
import parapick as pp

tomograms = pp.read_tomograms("tomograms.star")  # RELION 5 tilt-series geometry
picks = pp.read_picks("picks.star")  # 2D picks (any registered format)

result = pp.localize(tomograms, picks, config=pp.LocalizeConfig(alpha=0.5))
particles = result.particles(pp.Selection(min_score=0.3))
pp.write_particles(particles, "particles.star", tomograms_path="tomograms.star")

for ts in result.results:
    print(ts.name, ts.fit.n_components, ts.fit.converged, ts.fit.sigmas.mean())
```

The packages are layered and can be used on their own:

| Package               | Contents |
|-----------------------|----------|
| `parapick.tomography` | Format-neutral data model: `TiltSeries` (one affine projection per tilt), `PickTable`, `Particles`, pick-to-tilt matching. |
| `parapick.gmm`        | The JAX numerics: `EMFitter`, `LocalizeConfig`, initialisers, capped Sinkhorn, EM step. NumPy in, NumPy out. |
| `parapick.formats`    | Format plug-ins and their registry (RELION 5, RELION coordinates, Topaz). |
| `parapick.workflows`  | Use cases: `localize`, `project_particles`, `evaluate`, `simulate`, ... |
| `parapick.cli`        | The thin `argparse` layer. |

## Inputs and conventions

- **Geometry:** RELION 5 `tomograms.star` and the per tilt-series STAR files
  (`rlnTomoXTilt`, `rlnTomoYTilt`, `rlnTomoZRot`, `rlnTomo{X,Y}ShiftAngst`). ParaPick uses
  exactly RELION's projection model, including its centring conventions. The tilt-image size
  is read from the header of the first tilt image, from the `ppkTiltImageSize{X,Y}`
  labels, or from `--tilt-image-size`.
- **Picks:** pixel coordinates on the tilt images, as written by single-particle pickers.
  Read directly from RELION (STAR, autopick), crYOLO (STAR, CBOX, EMAN box), Topaz,
  cryoSPARC (`.cs`), EMAN2 (`.box`) and Xmipp/Scipion (`.pos`). `--flip-y` handles pickers
  whose y axis points the other way. The canonical ParaPick picks file is a single STAR
  table with `rlnTomoName`, `rlnMicrographName`, `rlnCoordinateX/Y` and, optionally,
  `rlnAutopickFigureOfMerit` and `rlnMicrographPixelSize` (for binned picks). Picks are
  matched to tilts by micrograph path, file name or stem.
- **Particles:** RELION 5 centred Ångström coordinates. Legacy pixel coordinates can be read
  with the geometry.

See [`docs/conventions.md`](docs/conventions.md) for the frames, every STAR label (including
the ParaPick-specific `ppk*` labels), and how to write a format plug-in. See
[`docs/algorithm.md`](docs/algorithm.md) for how the paper's equations map to the code.

## Development

```bash
uv sync                        # create the environment (dev tools included)
uv run pre-commit install      # ruff + mypy on every commit
uv run pytest --cov            # tests (CPU, double precision)
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

CI runs lint, type-checks, tests on Linux and macOS for Python 3.11–3.13, and builds the
package. Tagging `vX.Y.Z` publishes to PyPI through trusted publishing.

## License

MIT. See [LICENSE](LICENSE).
