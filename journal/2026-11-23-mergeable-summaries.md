# 2026-11-23 — From Batched Mixture Reduction to Mergeable Summaries (design discussion)

## Context & Goal

This entry records a design discussion that followed the [Salmond tree](./2026-11-23-salmond-tree.md) experiment, and the direction it produced. **No new code or measurements.** The Salmond tree's code was discarded after this discussion; its numbers stay in its own entry.

The discussion went through four framings, each more general than the last:
1. use expensive mixture reducers offline, in batches;
2. replace the bbox resolve rule with an error estimate;
3. treat the whole problem as compression of the rendered density;
4. stop assuming a node is a Gaussian at all.

---

## 1. Batched mixture reduction (the starting idea)

**Idea.** Runnalls and Salmond (`rust/src/`) give very good mixtures but cost `O(N²)`–`O(N³)` per reduction, too slow per frame. Apply them offline instead, in bounded batches (for example 1024 inputs → 32 groups, not necessarily consecutive in x), recursively, so that nodes approximate their contents better and the client descends less.

**Analysis at the time:**
- **Cost:** with a fixed batch size the total is `O(N · batch)`, linear in N. It's affordable offline even at 10M points.
- **Exactness:** each reduced component is a moment-preserving merge of a subset, so the result is a partition. The tree stays exact as long as the points are permuted so every group is contiguous.
- **Uneven groups:** they break the implicit `j·F` child indexing, so fanout must be variable with explicit links.
- **Boundary cost:** a group chosen by similarity spans its whole batch in x. At a viewport edge, all groups of that batch straddle and must descend, against 1 of 32 with x-runs. Batches must therefore be x-windows, to keep that cost bounded.
- **Group points, not nodes:** regrouping x-run leaves can't help white noise, because every 32-point x-run is equally tall in y. Clustering has to start from points, which is also what the k-d tree does.

The [Salmond tree](./2026-11-23-salmond-tree.md) implemented this. At 1M points it cut wire 11–12× against the x-tree on white-noise overview and zoom-in, and 4–7× on bursty-data pans. It cost 11–47% more wire on the smooth random walk, and windowed white-noise views were 7–16% worse.

## 2. The resolve rule is a proxy, not an error measure

**The current rule** (`standard_decision` in `progressive-common/src/render.rs`, shared by every hierarchy strategy):
1. **x-containment (exactness):** a node whose `[min_x, max_x]` crosses a viewport edge always descends, down to raw points, which are clipped.
2. **Size on screen:** a fully inside node resolves as one Gaussian when its bbox diagonal, in kernel widths, is below `merge_threshold`:
   - `x_factor = screen_w / view_width / σ_px` and `y_factor = screen_h / y_span / σ_px`;
   - resolve when `dx² + dy² < t²`;
   - `t = 1` and σ = 16 px in the benchmarks, so the bbox must be about 16 px on screen.
3. **Off-screen exception:** with explicit `y_bounds`, a node at least 6σ off-screen (both its bbox and its mean against its own spread) resolves regardless of size.
4. **Greedy merging:** `push_group` merges neighbouring outputs under the same size test. This affects kernel count, not what's fetched.

**What's wrong with it.** The bbox measures size, not error:
- **One outlier** stretches the bbox and forces descent, even though it changes the rendered density by about `1/n` of the node's mass.
- **A wide but nearly Gaussian node** descends as eagerly as a bimodal one, which really is badly approximated.

For the Salmond tree this matters directly: Salmond's distance absorbs light, far components easily, which stretches bboxes. That probably explains part of its `y_zoom` and boundary costs.

## 3. Error-driven descent

**What to estimate.** On screen, a node's points render as `Σ wᵢ N(pᵢ, S)`, where `S = diag(σx², σy²)` is the kernel in data units for the current view. Resolving draws `M · N(μ, Σ + S)` instead. The quantity to estimate is the L1 difference divided by the node's mass `M`: the node's share of the frame's density error (what NMAE measures).

**The key fact.** Blurring with a Gaussian kernel adds `S` to the covariance and leaves every cumulant of order ≥ 3 unchanged. The drawn Gaussian matches mass, mean and covariance exactly, so all of the error comes from the third and fourth cumulants. An Edgeworth expansion gives

```
err(node, view) ≈ c₃·‖γ₃‖ + c₄·‖γ₄‖          (capped at 2)
γ₃ = κ₃ standardized by (Σ + S)^(3/2),   γ₄ = κ₄ standardized by (Σ + S)²
```

Here `c₃` and `c₄` are the L1 norms of the Hermite correction terms: constants computed once and then calibrated against measured error.

**How it behaves:**
- **Node much smaller than σ:** `S` dominates, the error goes to 0, and the node resolves, matching today's behaviour.
- **One far outlier:** its weight `1/n` keeps κ₃ and κ₄ small, so the node resolves where the bbox rule would descend.
- **Two y-bands in one node:** γ₄ ≈ −2, so it descends until the bands are separate nodes.
- **Uniform block (white-noise x-runs):** γ₄ ≈ −1.2 while the block is wide on screen, so it descends; zooming out lets `S` take over and it resolves.
- **Anisotropic views:** handled by standardizing with the full `Σ + S`, so `y_zoom` needs no special case.

**Rule and guarantee.** Resolve when the node is x-contained and `err ≤ ε`. Each node's error is weighted by its mass, so the frame's relative error is at most about ε. ε becomes an accuracy target (for example 1%) instead of a pixel size. `push_group` uses the same test.

**Cost:** 9 more numbers per node (4 third and 5 fourth central moments): 36 B as `f32`, about 18 B quantized. They merge exactly (Pébay's parallel update formulas), so `Node::combined` stays exact and associative.

**Caveats:**
- Edgeworth is an approximation, weakest for nodes with few points relative to σ, so it needs an estimate-versus-measured calibration test.
- `PackedNode` is shared by every strategy, so the first test should change only the strategies being compared.

**What it enables:** budgeted refinement. Instead of a yes/no rule, the client refines the nodes with the largest mass × error first, until a byte or time budget runs out.

## 4. Reframing: compression of the rendered density

The goal is better stated as compression: replace N points with N' ≪ N kernels so that the **rendered** density changes as little as possible.

```
D(x)  = Σᵢ wᵢ · N(x; pᵢ, S)
D'(x) = Σⱼ Wⱼ · N(x; μⱼ, Σⱼ + S)
```

Because `S` changes with the view, the same compression can be good at one zoom and bad at another.

**Consequences:**
- **A moment-matched node is already the best single Gaussian** for its points (the KL-optimal M-projection). After rendering, its mass, mean and covariance are exact; it loses only the cumulants of order ≥ 3. So "preserve the rendered moments" and "keep each drawn node's standardized higher cumulants small" are the same requirement.
- **A tree is a family of nested compressions.** Every cut is one compression. The client's job is to pick the best cut for the view: fewest kernels or bytes at error ≤ ε, or least error within a budget.
- **Keep compressing by partition plus moment matching.** Freely fitted kernels (EM, least squares) could lower the error somewhat. But they lose exact mass and moments over every x-range and don't merge exactly, so the only freedom left is which points go together.
- **Builder objective:** choose groups to minimize rendered error at the scale where they'll be drawn. Options are Runnalls' KL bound or the closed-form rendered L2 distance (a sum of Gaussian overlaps), each computed with that level's kernel `S_ℓ`, instead of a single overview scale for every level.
- **Measurement:** compare strategies on rate against distortion (bytes and kernels against true NMAE), not on bytes at a fixed bbox rule. The harness already records `kernel_count` and per-step NMAE.

## 5. Generalization: a node is a mergeable summary, not necessarily a Gaussian

The Gaussian was chosen because it was a known good method, not because the problem requires it. What the tree actually needs from a node is an object that can:
- **merge** into a parent object of the same kind;
- **estimate its own rendered error** for a given view;
- **emit** something the renderer can draw.

What it stores internally is free.

### The `Summary` trait (sketch)

```rust
trait Summary {
    fn from_points(points: &[Point]) -> Self;
    fn merge(children: &[Self]) -> Self;             // closed under aggregation
    fn error(&self, view: &View) -> f64;             // estimated rendered error at this view's kernel S
    fn emit(&self, view: &View, out: &mut Vec<f32>); // kernels for the GPU (today: Gaussians)
    fn encode(&self) -> Vec<u8>;                     // bounded size, for static byte ranges
}
```

The project's convention is `dyn` over generics, so the real version would probably be an enum or a per-strategy byte format behind a `dyn` trait. The sketch only fixes the capabilities.

### Three independent axes

A strategy becomes a choice along three axes that earlier experiments mixed together:
1. **Grouping:** which points go together (x-runs, k-d splits, grid cells, Salmond or Runnalls clustering).
2. **Summary:** what a node stores and how it renders (today always one moment-matched Gaussian).
3. **Cut policy:** how the client chooses nodes (bbox rule, error ≤ ε, or largest error first under a budget).

For example, the ripmap and sparse-grid "planes" are really a different summary (a y-histogram of moment cells) attached to an x-tree grouping.

### Rules every summary must follow

- **Exact merge, or a carried error.** If `merge` is approximate (for example, reducing 2K kernels to K), the node stores its inherited error plus the reduction error, so `error()` stays honest as errors accumulate up the tree.
- **Mass, mean and covariance stay exact,** keeping today's guarantees and the conformance suite. Everything beyond is judged by rendered error.
- **View-dependent error without children:** `error()` receives the view's `S` (and the on-screen part of the node) and must not need the children. That's what makes it usable for deciding whether to descend.
- **Bounded byte size,** so each node is still a fixed range in a static object.
- **Output stays as kernels** for now: if `emit` produces Gaussians, the renderer and the `Frame` contract don't change. Other primitives (textures, segments) would be a separate change.

### Candidate summaries

| Summary | Merge | Error estimate | Good at |
|---|---|---|---|
| **Moments up to order 4** (one Gaussian + skewness/kurtosis) | exact | Edgeworth terms against `Σ + S` | cheap upgrade of today's node; tells you *when* one Gaussian is wrong |
| **Small Gaussian mixture, K ≤ 4–8** | 2K → K by Runnalls (approximate, error stored) | stored reduction error + per-component Edgeworth | bands, multimodal nodes, outliers kept as their own component |
| **Dyadic histogram / Haar coefficients** on a global grid | exact (coefficients add) | **exact in L2** (Parseval: sum of dropped coefficients squared); blurring by `S` damps fine scales by a known factor | white noise and dense 2D regions; what ripmap and grid do implicitly, made explicit |
| **Fourier samples** of the point distribution | exact (translation is a phase shift) | rendering multiplies by `exp(-½ωᵀSω)`, so truncation error is bounded directly by the view | clean theory, but needs a conversion step to output Gaussians |
| **Segment kernels** (a Gaussian smeared along a line, closed form with erf) | approximate (refit), error stored | residual spread around the segment | smooth time series: one primitive per long stretch of random walk |

Haar is the most principled, with exact merges and an exact error bound. Segment kernels target the one case where every current strategy is weak, smooth curves, but they're the most work.

## Findings

- The resolve rule, not just the grouping, limits every hierarchy strategy: it measures size, not error.
- "Which points go together", "what a node stores" and "how the client cuts" are independent design axes. Testing them separately should make future comparisons easier to interpret.
- Mass, mean and covariance after rendering are already exact for any moment-matched node. All remaining error lives in higher-order shape, which is exactly what a richer summary or an error estimate has to capture.

## Next Hypotheses

1. **Offline summary benchmark (next step).** In memory, no network or byte layout:
   - build each candidate summary on the same x-tree grouping;
   - for a set of views, choose the cut with the fewest bytes that meets error ≤ ε, using each summary's own `error()`;
   - measure the true error against the ground-truth rendered density;
   - report rate against distortion (bytes and kernels against true NMAE) and estimated against true error, since a trustworthy `error()` is the foundation.

   Start with moments up to order 4, a small mixture with K ≤ 4, and Haar. Then swap groupings (Salmond, k-d) in the same benchmark.
2. **Calibrate `c₃` and `c₄`** for the Edgeworth estimate from that benchmark.
3. **Promote the winner** to a real on-disk layout and client, compared with an x-tree baseline under the same cut policy.
4. **Builder cost at each level's scale** (Runnalls KL or rendered L2 with `S_ℓ`), evaluated on the same rate-distortion curves.
