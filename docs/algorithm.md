# Algorithm

ParaPick fits a 3D Gaussian mixture model (GMM) to 2D detections spread over the tilt
images, by expectation-maximisation (EM). This page maps the equations of the paper (§2)
to the code. It then describes the implementation choices and the optional extensions.

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| `i = 1..N` | tilt images | axis 0 of `Observations.detections` |
| `j = 1..M_i` | detections on tilt `i` (padded to `M`) | axis 1, with `Observations.mask` |
| `k = 1..K` | mixture components (candidate particles) | `EMState.means` rows |
| `P_i ∈ R^{2×3}` | projection of tilt `i` | `TiltSeries.projections`, `Observations.projections` |
| `ŷ_ij` | detection with the tilt's translation removed | `Observations.detections` |
| `μ_k`, `π_k`, `σ_i²` | centres, weights, per-tilt variances | `EMState.means`, `.log_weights`, `.sigma2` |

`TiltSeries` stores, for each tilt, a general affine map `y = P_i X + t_i` from the specimen
frame to the tilt image (both in Å). The paper writes `ŷ_ij = R_iᵀ(y_ij − δ_i)` with `P_i`
the first two rows of the tilt rotation. That is the same model: ParaPick folds the in-plane
rotation into `P_i` and uses `ŷ_ij = y_ij − t_i`. Squared distances are unchanged because
`R_i` is orthogonal (`workflows/localize.py::localize_tilt_series`).

## Equations ↔ code

| Paper | What | Where |
|---|---|---|
| (1) | `d²_ijk = ‖ŷ_ij − P_i μ_k‖²` | `gmm/em.py::tilt_statistics` |
| (4)–(7) | capped Sinkhorn: `r_ijk = u_ij v_ik K_ijk`, `Σ_k r = 1`, `Σ_j r ≤ 1` | `gmm/sinkhorn.py::capped_sinkhorn` |
| (8) | `μ_k = (Σ_i n_ik/σ_i² PᵢᵀPᵢ + εI)⁻¹ Σ_i Pᵢᵀ Σ_j r_ijk ŷ_ij/σ_i²` | `gmm/em.py::em_step` (batched 3×3 solves) |
| (9) | `σ_i² = Σ_jk r_ijk d²_ijk / (2 M_i)` | `gmm/em.py::em_step` |
| (11)–(13) | Dirichlet MAP weights `π_k ∝ max(0, n_k + α − 1)`, pruning | `gmm/em.py::em_step` |
| `s_k = n_k / N` | confidence score | `gmm/fitter.py::EMFitter.fit` |

`tests/helpers/numpy_em.py` is a literal NumPy transcription of these equations, structured
like the original proof of concept. `tests/unit/test_gmm.py` checks that one JAX EM step
matches it to numerical precision.

## Implementation notes

- **Static shapes.** Picks are padded to `(N_pad, M_pad)` with masks. `M` is rounded up to
  a power of two and `N` to a multiple of 8. The initial `K` is rounded to a coarse grid, so
  similar tilt-series reuse one compiled XLA program. Pruning masks components
  (`log π_k = −∞`) instead of removing them.
- **Memory.** The E-step runs with `lax.scan` over tilts, accumulating sufficient statistics
  (`n_ik`, `Σ_j r_ijk ŷ_ij`, `Σ r d²`). Memory is `O(M·K)`, not `O(N·M·K)`.
- **Sinkhorn in the linear domain.** Each kernel row is divided by its maximum, which only
  rescales `u` and leaves the responsibilities unchanged. The alternating projections are
  then two matrix-vector products each, about 3× faster than log-domain iterations, with the
  same result. The last projection is a row projection, so `Σ_k r_ijk = 1` holds exactly.
- **Regularisation.** `ε = regularization × trace(A_k)/3`, so it is scale-free in Å.
- **Convergence.** Iteration stops when three conditions hold together: the mean
  displacement of the surviving components is below `tolerance` (Å), no component was pruned
  in that iteration, and annealing has finished. Otherwise it stops after `max_iterations`.
- **Initial K.** `K⁽⁰⁾ = overcomplete_factor × max_i M_i`. The cap of one detection per
  tilt means at least `max_i M_i` components are needed.

## Extensions beyond the paper

These are on by default. Each can be switched off; doing all of the following reproduces the
paper's model exactly: `background_fraction=None`, `annealing=0`, `merge_distance=0`, and
`UniformBoxInitializer`.

1. **Uniform background (outlier) term**, `background_fraction`. An extra column of the
   soft assignment has kernel `π_0 · 2πσ_i² / A_i`, which is the density `π_0 / A_i` of a
   uniform background over the image area `A_i`, relative to the components' unnormalised
   Gaussians. This column is *not* capped. `π_0` is re-estimated jointly with the weights.
   `σ_i²` is computed over the responsibility held by the components.
   *Why:* in the closed-world model, every false-positive pick must be explained by some
   component, which inflates `σ_i` and creates spurious particles. On simulations with
   ~15–30% false positives, F1 went from ~0.3–0.7 to ~0.95–0.99. With no false positives
   the results are essentially unchanged.
2. **Deterministic annealing**, `annealing`. The variances are bounded below by a floor
   that starts at `σ⁽⁰⁾²` and is multiplied by `annealing²` each iteration.
   *Why:* well-fitted particles quickly drive `σ_i` down to the pick noise. Components that
   are still misplaced in depth then freeze, collecting only near-zero-tilt detections.
   Annealing lets them settle coarse-to-fine.
3. **Lifted initialisation**, `LiftedPicksInitializer`. The picks of the least tilted image
   are back-projected along the viewing direction, at a random depth inside the box, so most
   initial components start on the ray of a real particle. The paper's uniform
   initialisation is available as `--init uniform`.
4. **Duplicate merging**, `merge_distance`. After the fit, components closer than
   `5 × median σ_i` are merged (support-weighted mean, summed support). They are duplicates
   of one particle that took its detections on different tilts. Set it to about the
   particle radius for best results.

## Confidence scores

`s_k = n_k / N` is the effective fraction of tilts that support particle `k`. If the 2D
picker has per-image recall `R`, a true particle has an expected score of about `R`.
Spurious components collect only occasional detections. Thresholding the score at `K/N`
corresponds to requiring `K` supporting views. Under a binomial model, the resulting 3D
recall is `P(X ≥ K)` with `X ~ Binom(N, R)` (paper, eq. 15).
