# ParaPick

[![CI](https://github.com/oierlauzi/parapick/actions/workflows/ci.yml/badge.svg)](https://github.com/oierlauzi/parapick/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tomogram-free particle localization for cryo-ET.** ParaPick finds the 3D coordinates of
particles directly from 2D picks made independently on each image of a tilt-series. You can
pick with any single-particle picker (Topaz, crYOLO, RELION, ...). ParaPick fuses the picks
from all tilts into a consistent set of 3D positions, with a confidence score for each one.
No tomogram needs to be reconstructed.

## How it works

ParaPick builds a 3D model of the particle positions that has to explain the 2D picks of
every tilt: projected through the tilt-series geometry, each particle should land on a pick in
every tilt image. The model is fitted by repeating two steps:

1. **Match:** each pick is matched to the particles that project close to it. A particle
   takes at most one pick per tilt, and picks that match no particle are set aside as false
   positives.
2. **Move:** each particle moves to the 3D position that best agrees with its picks.

Particles that too few picks support are dropped, so their number is found automatically. A
particle's score is the fraction of tilts whose picks support it.

The method is a Gaussian mixture fitted by expectation-maximisation.
[`docs/algorithm.md`](docs/algorithm.md) gives the details and how they map to the code.

## Features

- **Input:** the tilt-series geometry in RELION 5 format (`tomograms.star`), and 2D picks in
  the formats of RELION, crYOLO, Topaz, cryoSPARC, EMAN2 or Xmipp/Scipion.
- **Output:** RELION 5 particles, which feed straight into subtomogram averaging.
- **Stacked tilt-series:** `unstack` splits each stack into one micrograph per tilt, for
  pickers that need one image per file.
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
| `match-template`| *Optional:* pick every tilt image by 2D template matching with Leopard-EM (see below). |
| `formats`       | List the available file formats, including installed plug-ins. |

Common options: `--device {auto,cpu,gpu}`, `--x64`, `-v`/`-q`.

### Picking by 2D template matching (optional)

With [Leopard-EM](https://github.com/Lucaslab-Berkeley/Leopard-EM) installed
(`pip install "parapick[leopard]"`, Python ≥ 3.12), `match-template` searches every tilt image
for a template and writes the peaks as picks for `localize`:

```bash
parapick match-template --tomograms tomograms.star --template ribosome_1.7A.mrc \
    --particle-diameter 250 -o match/
parapick localize --tomograms match/tomograms.star --picks match/picks.star -o localize/
```

2DTM is tuned for single, untilted, high-dose images, so the parameters of each tilt are
derived from its acquisition (`parapick.matching.schedule`):

- **low-pass** where the dose received before the tilt has destroyed more signal than noise
  (radiation damage model of Grant & Grigorieff, 2015), plus a high-pass well below the
  particle size;
- **angular and defocus steps** coarser as the remaining resolution drops (never finer than
  Leopard-EM's defaults);
- **defocus range** covering the tilted plane across the field of view and the thickness of
  the sample.

Everything is inferred from `tomograms.star`: the dose and acquisition order from
`rlnMicrographPreExposure`, the CTF of each tilt from `rlnDefocusU/V/Angle`, the angle of the
sample (pre-tilt included) and the direction of its defocus gradient from the geometry, and
the thickness from the tomogram size. Only the particle diameter is required; `--thickness`
and `--dose-per-tilt` override the inferred values. The template must be at the pixel size of
the tilt-series. `match/schedule.csv` lists the parameters and predicted SNR of every tilt;
`--max-tilts N` searches only the N best tilts of each tilt-series (usually the first
acquired), and `--schedule-only` writes the Leopard-EM configurations without searching. Runs
resume tilt by tilt. The derived parameters are a principled starting point, not validated
settings.

The template matching itself is done by Leopard-EM, developed by the Lucas lab (UC Berkeley,
BSD-3-Clause licence); ParaPick only derives its parameters for tilt-series. If you use
`match-template`, please cite Leopard-EM, and Grant & Grigorieff for the damage model:

- Giammar MD, Dickerson JL, Hall LN, Lucas BA (2026). Leopard-EM: an extensible 2D
  template-matching package to accelerate in situ structural biology. *Acta Cryst. D* 82,
  30–42. https://doi.org/10.1107/S2059798325009982
- Grant T, Grigorieff N (2015). Measuring the optimal exposure for single particle cryo-EM
  using a 2.6 Å reconstruction of rotavirus VP6. *eLife* 4, e06980.
  https://doi.org/10.7554/eLife.06980

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
| `parapick.matching`   | Per-tilt 2D template matching parameters (numpy/scipy) and the optional Leopard-EM runner. |
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
