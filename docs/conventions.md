# Conventions, file formats and plug-ins

## Coordinate frames

ParaPick's numerics only see a **format-neutral** model (`parapick.tomography`):

- **Specimen frame:** 3D coordinates in Å, centred as defined by the source convention. For
  RELION 5 this is the frame of `rlnCenteredCoordinate{X,Y,Z}Angst`.
- **Image frame:** 2D coordinates in Å on a tilt image, `pixel_index × pixel_size`, from the
  first pixel. A pick in pixels converts to this frame by multiplying by the pixel size of
  the image it was picked on. Picks made on binned images need only their pixel size, not
  the binned image dimensions.
- Each tilt is an affine orthographic projection: `y = P_i X + t_i`.

Readers translate their conventions into `(P_i, t_i)`. Writers translate them back.

## RELION 5 projection model

From `Tomogram::setProjectionMatrix` (RELION 5):

```
y_px = floor(img/2) + shift_Å/p + [Rz(ZRot) · Ry(YTilt) · Rx(XTilt) · (X_px − floor(tomo/2))]_xy
```

RELION's particle coordinates are relative to `tomo/2`, with no `floor`
(`ParticleSet::getPosition`). With `X_c` in centred Å and `o = (tomo/2 − floor(tomo/2))·p`:

```
P_i = [Rz Ry Rx]_xy        t_i = floor(img/2)·p + shift_Å + P_i·o
```

`tests/unit/test_geometry.py` checks this against a literal transcription of RELION's 4×4
matrix composition, including odd tomogram sizes. Deformation grids are ignored, with a
warning.

## Files

### Tilt-series geometry: `tomograms.star` (read and write)

- `data_global`: `rlnTomoName`, `rlnTomoTiltSeriesStarFile`, `rlnTomoTiltSeriesPixelSize`.
  Optional: `rlnTomoSize{X,Y,Z}` (bin-1 pixels), `ppkTiltImageSize{X,Y}`.
- `data_<TomoName>` in each tilt-series STAR file: `rlnTomoYTilt`. Optional:
  `rlnMicrographName`, `rlnTomoXTilt`, `rlnTomoZRot`, `rlnTomo{X,Y}ShiftAngst`.

Like RELION 5 (`TomogramSet::loadTomogram`), both image layouts are supported:

- **A (stack):** the global `rlnTomoTiltSeriesName` names a stack, e.g. `TS_01.st:mrc`. Tilt
  `i` is its section `i`, in the row order of the tilt-series table. RELION gives this layout
  precedence.
- **B (one file per tilt):** each row's `rlnMicrographName` is a 2D image, as written by
  RELION's motion correction.

Tilts are identified, and picks matched, by `rlnMicrographName` when it is present.
Otherwise the identifier is the section reference `N@stack` (1-based). Image references use
RELION's syntax: `N@path` selects a section and a `:mrc` type suffix forces the MRC reader.
When writing, `rlnTomoTiltSeriesName` is kept only if the tilt images are exactly the
sections of one stack, in order. A stale stack name would otherwise take precedence over
new per-tilt images.

### Unstacked datasets (`parapick unstack`)

`parapick unstack --tomograms tomograms.star -o OUT` writes:

- `OUT/micrographs/<TS>_<NNN>.mrc`: one float32 MRC2014 image per tilt (1-based tilt order),
  at native sampling. Stems are unique across tilt-series, so pickers that report images by
  stem (Topaz) match unambiguously. The pixel size is not recorded in the headers (they say
  1 Å/px). The pickers ignore it: crYOLO and Topaz read only the pixel data and shape, and
  RELION and cryoSPARC take it from the user or the STAR optics table. ParaPick takes it from
  the geometry.
- `OUT/tomograms.star` and `OUT/tilt_series/<TS>.star`: the same geometry and metadata, with
  `rlnMicrographName` pointing at the new files (layout B).

Stacks are read section by section, as MRC2014 image stacks (space group 0, as IMOD writes
them). Stacks flagged as volumes (space group 1) are read once, whole. Reading and writing go
through vitrio, in batches of `--batch-size` images (bounding memory) arranged as a pipeline:
the next batch is requested before the current one is written. The bundled image I/O
(`parapick.formats.vitrio_images.VitrioImageIO`) is synchronous for now. An asynchronous
`ImageIO` (e.g. on vitrio's `read_batch_async` / `write_batch_async`) can overlap the steps
without changes to the workflow.

Pick on `OUT/micrographs/` and pass `--tomograms OUT/tomograms.star` to `convert-picks` and
`localize`.

Paths are resolved as RELION does (relative to the project directory). ParaPick also tries
the directory of the STAR file and its parents. The tilt-image size is looked up in this
order:
1. `--tilt-image-size`;
2. the `ppkTiltImageSize{X,Y}` labels;
3. the header of the first tilt image (header only).

### 2D picks: ParaPick picks STAR (`relion5` format, read and write)

One `data_picks` table for any number of tilt-series:

| Label | Required | Meaning |
|---|---|---|
| `rlnMicrographName` | yes | tilt image, matched to the tilt-series STAR rows |
| `rlnCoordinateX`, `rlnCoordinateY` | yes | pixels (x = column, y = row), RELION SPA convention |
| `rlnTomoName` | no | restricts matching to that tilt-series (otherwise searched in all) |
| `rlnAutopickFigureOfMerit` | no | picker score (`--min-pick-score` thresholds it) |
| `rlnMicrographPixelSize` | no | pixel size of the picked images (binned picks) |

**Matching.** A pick is matched to its tilt by trying, in order: the exact micrograph path,
the file name, then the stem (the file name without known image extensions such as
`.mrc`). A RELION `N@` stack prefix is ignored. Unmatched picks are reported and dropped.
Ambiguous matches are an error.

### Other pick formats

- `relion-coords`: per-micrograph RELION coordinate STAR files, given as an autopick index
  (`rlnMicrographName` + `rlnMicrographCoordinates`), a directory or a single file.
- `topaz`: TSV with `image_name x_coord y_coord [score]`.

`parapick convert-picks INPUT --tomograms tomograms.star -o picks.star` turns either one into
a self-describing ParaPick picks file.

### 3D particles (`relion5`, read and write)

- `particles.star` (`data_particles`): `rlnTomoName`, `rlnCenteredCoordinate{X,Y,Z}Angst`,
  `rlnAutopickFigureOfMerit` (score `s_k`) and `ppkEffectiveCount` (`n_k`).
- `optimisation_set.star`: `rlnTomoParticlesFile` and `rlnTomoTomogramsFile`. It can be
  passed directly to RELION 5 subtomogram extraction.
- Legacy `rlnCoordinate{X,Y,Z}` (bin-1 pixels from the corner) are read when the geometry is
  provided.

### ParaPick-specific labels

RELION keeps unknown labels when it reads and writes STAR files, so these are safe to carry
along:

| Label | Where | Meaning |
|---|---|---|
| `ppkTiltImageSizeX`, `ppkTiltImageSizeY` | `tomograms.star` | tilt-image size (avoids reading image headers) |
| `ppkEffectiveCount` | `particles.star` | effective number of supporting detections `n_k` |

Per-tilt diagnostics (`σ_i`, number of picks), iterations, convergence, background fraction
and log-likelihood are written to `localize_report.json`.

## Writing a format plug-in

A format is any object with `name`, `description`, `suffixes` and `sniff(path)`, plus one
or more role methods (see `parapick.formats.base`):

```python
# my_package/imod.py
from pathlib import Path
from parapick.formats import TomogramsReadOptions
from parapick.tomography import TiltSeries, TomogramSet


class ImodFormat:
    name = "imod"
    description = "IMOD .xf/.tlt alignments"
    suffixes = (".tlt",)

    def sniff(self, path: Path) -> bool:
        return path.suffix == ".tlt" and path.with_suffix(".xf").exists()

    def read_tomograms(self, path: Path, options: TomogramsReadOptions) -> TomogramSet:
        # Build P_i (2x3) and t_i (Å) for each tilt, then:
        # return TomogramSet((TiltSeries(...), ...))
        raise NotImplementedError
```

Register it in your package's `pyproject.toml`:

```toml
[project.entry-points."parapick.formats"]
imod = "my_package.imod:ImodFormat"
```

It then appears in `parapick formats` and in every `--*-format` option, and it takes part
in auto-detection. Nothing in ParaPick needs to change. In Python, you can also call
`parapick.formats.default_registry().register(ImodFormat)`.

| Role | Method |
|---|---|
| `TomogramsReader` / `TomogramsWriter` | `read_tomograms(path, options)` / `write_tomograms(tomograms, path)` |
| `PicksReader` / `PicksWriter` | `read_picks(path)` / `write_picks(picks, path)` |
| `ParticlesReader` / `ParticlesWriter` | `read_particles(path, tomograms)` / `write_particles(particles, path, context)` |
