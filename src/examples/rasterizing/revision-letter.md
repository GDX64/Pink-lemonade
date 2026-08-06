# Response to Reviewers

We thank the reviewers for their careful reading and for the constructive
comments, which have helped us improve the manuscript considerably. Below, we
address each point raised, indicating the corresponding changes in the revised
version. In the revised version, parts marked in yellow indicate modified text, while those marked in green indicate text we added. 

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

*Response:* We thank the reviewer for this suggestion, and we agree that the "empty" view is the practical failure mode to avoid. The mechanism responsible for it in our pipeline is the opacity cut applied during dynamic color mapping,
which suppresses very low densities and improves legibility in the aggregated
regime (particularly when quantization is enabled), but which also hides isolated
splats once kernels no longer overlap. In the current system, this cut is
user-configurable, so the degenerate view can be avoided by lowering or disabling the threshold; the paper describes the threshold in the Pass 3 discussion and
identifies the high-zoom degeneration explicitly under Limitations, noting that a plain line or scatter chart is the more appropriate representation at that scale.

Automating the switch is a natural next step, and we agree with the reviewer that it is worth stating how it would be triggered. The criterion we consider
appropriate is a density test on the current view — disabling the cut once the
on-screen density falls below roughly one merged kernel per kernel bandwidth
along the time axis, which is precisely the point at which the density field has
degenerated into a scatter plot, and every kernel should remain visible. We have
not implemented this automatic transition in the present version, and we prefer
not to describe it in the manuscript as though it were part of the system; we
would be glad to add it, together with an illustration of the transition, but the 6-page limit prevents us from doing that.

## Reviewer 2

### R2.1 — Motivation and contextualization in the Introduction
> The authors have a target application (visual exploration of massive time series / financial tick data). They could use it to motivate the contribution more effectively and contextualize it in the Introduction. The Introduction, as it is, reads very poorly as far as motivation and contextualization are concerned.

*Response:* We appreciate this observation and agree that the original Introduction did not make the motivation sufficiently clear. We have rewritten it around the target application, explaining why the visual exploration of massive financial tick data is challenging, and where conventional approaches fall
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
opened by naming the two closest lines of prior work, which did not convey why those particular lines were relevant. We have rewritten the opening paragraph to
lead with the rationale instead: it decomposes the problem into the three
requirements that an interactive weighted-KDE renderer must satisfy, identifies
which body of prior work addresses each one, and then states how our solution
differs from all three. The subsequent paragraphs, each of which already
contrasts our approach with the work it reviews, now follow from that framing.

### R2.4 — Explicit statement of the novel contributions
> Authors should explicitly state what is novel in the visual representation (the weighted KDE encoding of a second scalar variable) and what is novel in the rendering algorithm (the screen-space kernel reduction).

*Response:* We agree that these contributions deserved a more explicit
statement, and the revised manuscript now states each of them directly in the related work section to make it clear how our approach addresses shortcomings of prior work. On the
representation side, the novelty is the weighted KDE encoding of a second scalar
variable within the same view as the time series. On the algorithmic side, the
screen-space kernel reduction is a pragmatic strategy for the specific setting in which points are ordered along one axis: the general reduction problem has been addressed by the prior methods we discuss, which often achieve lower
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
This rationale is now stated in the revised Introduction, alongside the discussion added for R2.1.

### R2.6 — Objective comparison with the state of the art and formal complexity analysis
> There is no objective comparison with any current state-of-the-art method for displaying large-scale time series data, nor a formal computational complexity analysis.

*Response:* We appreciate this concern. The literature on visualization methods of large timeseries is vast, but we were focused on prior research or production
systems that encode a second variable as a continuous field over the time
series. The prevailing practice within production systems is either to display volume in a secondary
synchronized chart or to draw discrete 2D marks with area proportional to the value of the second variable,
neither of which is directly comparable to a continuous-field representation: the comparison would measure the difference between two visual encodings rather than between two methods for producing the same one. The revised Introduction states the limitations of the synchronized-chart practice, which is the alternative our target users employ today, as part of the motivation added for R2.1 and R2.5.

Rather than force such a comparison, we have addressed the reviewer's concern on the component of the pipeline where a fair comparison is well posed: we now compare our method quantitatively against established mixture-reduction methods
at matched output budget, reporting both runtime and approximation error (see
R1.1). We would be glad to expand the discussion of the prevailing alternatives in the text, but again, the 6-page limit constrains how far this can be developed.

Regarding the complexity analysis, we did not attempt a formal derivation. Our
merge is a single linear pass over the visible points, and the cost in the GPU rasterization depends on the number of surviving kernels and the screen resolution rather than on the dataset size; a formal
bound would therefore characterize the hardware more than the algorithm, and we
did not think it would inform the reader enough to justify the space it would
require within the 6-page limit. Instead, we have added pseudocode for
the merging algorithm and expanded its accompanying description, so that its
cost per frame can be read directly from the procedure, and we have defined the
symbols in the cost expression given in the Introduction (see R2.2). We would
also note that the benchmark figure plots latency against the number of points
on logarithmic axes, and that the legend annotates each method with the scaling
exponent recovered by a least-squares fit of its curve. The scaling behavior of
each method can therefore be read off the figure directly, which we believe
offers a measured counterpart to an asymptotic bound. In a future extended version of the paper, we could, of course, include a formal analysis as suggested.

