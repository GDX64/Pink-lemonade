# Response to Reviewers

We thank the reviewers for their careful reading and constructive comments.
Below we address each point raised.

## Reviewer 1

### R1.1 — Quantitative comparison with alternative aggregation/simplification methods
> The paper did not provide a direct quantitative comparison with other spatial aggregation techniques or alternative simplification methods, leaving this only as proposed future work. The work would be strengthened by including such a comparison rather than leaving it as future work.

*Response:* We have added comparisons with already cited works on mixtures approximation, we compared our method (that is considerably simpler than the others) with Runnalls and Salmond Clustered. We hope the tradeoffs between the methods are clear on the performance and error charts.

### R1.2 — Automatic transition from heatmap to scatter plot at high zoom
> Although the limitation of the heatmap pattern disappearing at high zoom levels is described appropriately, it would be helpful to include a brief discussion or illustration of how the automated transition from the heatmap to standard scatter plots could be triggered in the system to avoid an "empty" visualization.

*Response:* This is an interesting point. We have implemented a cut off threshold for small densities. The values that fall bellow this threshold are not rendered. We made this choice to improve the visual (specially when quantization is used), and it can be configured by the user.

## Reviewer 2

### R2.1 — Motivation and contextualization in the Introduction
> The authors have a target application (visual exploration of massive time series / financial tick data). They could use it to motivate the contribution more effectively and contextualize it in the Introduction. The Introduction, as it is, reads very poorly as far as motivation and contextualization are concerned.

*Response:* That is a very valid point. We have elaborated more on the introduction section on why we think our visualization improves on more orthodox approaches. 

### R2.2 — Undefined symbols in the complexity expression
> What are the authors referring to in the O() expression (what are N, W, H)?

*Response:* N is the number of points, W is the width of the kernel, and H is the height of the Kernel, thanks for pointing out.

### R2.3 — Structure and clarity of the Related Work section
> It would be more helpful to readers if the authors provided an overview of the rationale for their solution and then made the connections to previous work, explicitly stating where the new solution improves on prior approaches.

*Response:*

### R2.4 — Explicit statement of the novel contributions
> Authors should explicitly state what is novel in the visual representation (the weighted KDE encoding of a second scalar variable) and what is novel in the rendering algorithm (the screen-space kernel reduction).

*Response:* The screen space kernel reduction can be said to be pragmatic approach to this specific scenario where the points are ordered in one axis, the problem was already tackled by other discussed reduction methods, that will often yield better results but have prohibitive costs for our application.

### R2.5 — Design justification for KDE as the encoding of the second scalar variable
> Authors never discuss why KDE is the appropriate encoding for the second scalar variable — why not just use a second synchronized chart? At least a brief justification is expected.

*Response:* This point is similar to R2.3, we hope we have clarified the contributions in the introduction.

### R2.6 — Objective comparison with the state of the art and formal complexity analysis
> There is no objective comparison with any current state-of-the-art method for displaying large-scale time series data, nor a formal computational complexity analysis.

*Response:* It is hard to point out what is the current state of the art for this problem because this approach have not been taken by many researches or applications. What is usually done is encoding market volume in secondary charts or drawing 2D shapes with area proportional to the volume. Very different from our approach that produces a continuous field.

## Summary of Changes to the Manuscript

-
