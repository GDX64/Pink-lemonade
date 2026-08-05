# Response to Reviewers

We thank the reviewers for their careful reading and for the constructive
comments, which have helped us improve the manuscript considerably. Below we
address each point raised, indicating the corresponding changes in the revised
version.

## Reviewer 1

### R1.1 — Quantitative comparison with alternative aggregation/simplification methods
> The paper did not provide a direct quantitative comparison with other spatial aggregation techniques or alternative simplification methods, leaving this only as proposed future work. The work would be strengthened by including such a comparison rather than leaving it as future work.

*Response:* We agree that this comparison strengthens the contribution, and we
have added it to the revised version rather than leaving it as future work. We
now compare our method against two mixture-reduction approaches already cited in
the paper, Runnalls and Salmond (clustered), reporting both runtime and
approximation error. The results make the trade-off explicit: our method is
substantially simpler and faster, at the cost of a bounded increase in
approximation error. We hope the new performance and error charts convey this
trade-off clearly.

### R1.2 — Automatic transition from heatmap to scatter plot at high zoom
> Although the limitation of the heatmap pattern disappearing at high zoom levels is described appropriately, it would be helpful to include a brief discussion or illustration of how the automated transition from the heatmap to standard scatter plots could be triggered in the system to avoid an "empty" visualization.

*Response:* We thank the reviewer for this suggestion. The revised text now
discusses how the system detects this regime and adapts to it. The renderer
applies a user-configurable opacity cut that suppresses very low densities, which
improves legibility in the aggregated regime (particularly when quantization is
enabled). Because that same cut is what produces an "empty" view once kernels no
longer overlap, it is now disabled automatically when the on-screen density falls
below one kernel per kernel bandwidth along the time axis — the point at which
the visualization has effectively degenerated into a scatter plot and every
kernel should remain visible. We have added a brief discussion of this criterion
to the text.

## Reviewer 2

### R2.1 — Motivation and contextualization in the Introduction
> The authors have a target application (visual exploration of massive time series / financial tick data). They could use it to motivate the contribution more effectively and contextualize it in the Introduction. The Introduction, as it is, reads very poorly as far as motivation and contextualization are concerned.

*Response:* We appreciate this observation and agree that the original
Introduction did not make the motivation sufficiently clear. We have rewritten
it around the target application, explaining why the visual exploration of
massive financial tick data is challenging and where conventional approaches fall
short, before positioning our contribution against them.

### R2.2 — Undefined symbols in the complexity expression
> What are the authors referring to in the O() expression (what are N, W, H)?

*Response:* We thank the reviewer for catching this omission. $N$ is the number
of points, and $W$ and $H$ are the width and height of the kernel footprint in
pixels. The symbols are now defined explicitly where the expression is
introduced.

### R2.3 — Structure and clarity of the Related Work section
> It would be more helpful to readers if the authors provided an overview of the rationale for their solution and then made the connections to previous work, explicitly stating where the new solution improves on prior approaches.

*Response:* We thank the reviewer for this suggestion. The section previously
opened by naming the two closest lines of prior work, which did not convey why
those particular lines were relevant. We have rewritten the opening paragraph to
lead with the rationale instead: it decomposes the problem into the three
requirements that an interactive weighted-KDE renderer must satisfy, identifies
which body of prior work addresses each one, and then states how our solution
differs from all three. The subsequent paragraphs, each of which already
contrasts our approach with the work it reviews, now follow from that framing.

### R2.4 — Explicit statement of the novel contributions
> Authors should explicitly state what is novel in the visual representation (the weighted KDE encoding of a second scalar variable) and what is novel in the rendering algorithm (the screen-space kernel reduction).

*Response:* We agree that these contributions deserved a more explicit
statement, and the revised manuscript now states each of them directly. On the
representation side, the novelty is the weighted KDE encoding of a second scalar
variable within the same view as the time series. On the algorithmic side, the
screen-space kernel reduction is a pragmatic strategy for the specific setting in
which points are ordered along one axis: the general reduction problem has been
addressed by the prior methods we discuss, which often achieve lower
approximation error but at a cost that is prohibitive for interactive rendering
at our target data scale. We now make this positioning explicit.

### R2.5 — Design justification for KDE as the encoding of the second scalar variable
> Authors never discuss why KDE is the appropriate encoding for the second scalar variable — why not just use a second synchronized chart? At least a brief justification is expected.

*Response:* This is a fair point, and we have added an explicit design
justification. Briefly, a synchronized secondary chart requires the reader to
mentally join two views along the time axis, which becomes unreliable at the
densities we target, whereas the KDE encoding places the second variable in the
same spatial position as the value it qualifies, preserving the local
correspondence at any zoom level and degrading gracefully as points aggregate.
This rationale is now stated in the revised Introduction, alongside the
discussion added for R2.1.

### R2.6 — Objective comparison with the state of the art and formal complexity analysis
> There is no objective comparison with any current state-of-the-art method for displaying large-scale time series data, nor a formal computational complexity analysis.

*Response:* We appreciate this concern. Identifying a single state-of-the-art
baseline is difficult here, as we are not aware of prior research or production
systems that encode this second variable as a continuous field over the time
series. The prevailing practice is either to display volume in a secondary
synchronized chart or to draw discrete 2D marks with area proportional to volume,
neither of which is directly comparable to a continuous-field representation. We
have therefore addressed this point in two ways. First, we now compare our
method quantitatively against established mixture-reduction methods on the
component of the pipeline where a fair comparison is well posed, reporting both
runtime and approximation error (see R1.1). Second, we have revised the text to
describe the prevailing alternatives explicitly and to state why a direct
head-to-head comparison against them is not meaningful.

Regarding the complexity analysis, we did not attempt a formal derivation. Our
merge is a single linear pass over the visible points, and the cost in the GPU rasterization depends on the number of surviving kernels and the screen resolution rather than on the dataset size; a formal
bound would therefore characterize the hardware more than the algorithm, and we
did not think it would inform the reader enough to justify the space it would
require within the six-page limit. Instead, we have added pseudocode for
the merging algorithm and expanded its accompanying description, so that its
cost per frame can be read directly from the procedure, and we have defined the
symbols in the cost expression given in the Introduction (see R2.2). We would
also note that the benchmark figure plots latency against the number of points
on logarithmic axes, and that the legend annotates each method with the scaling
exponent recovered by a least-squares fit of its curve. The scaling behaviour of
each method can therefore be read off the figure directly, which we believe
offers a measured counterpart to an asymptotic bound. We would of course be glad to include a formal analysis
should the reviewer consider it essential.

