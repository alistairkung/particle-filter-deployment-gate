# Route pre-decision regime changes through the reference regime

Use the same B ↔ A ↔ C graph for the 60-, 120- and 240-second half-lives. Each edge changes one physical mechanism. With $q=1-2^{-5/h}$, stay in the current regime with probability $1-q$ at every five-second pre-decision update. On leaving A (reference), choose B (elevated background) or C (slow publisher) with equal probability. On leaving B or C, return to A. The equal onset split expresses symmetry in the absence of incident-frequency evidence.

The stationary distribution in A/B/C order is $(1/2,1/4,1/4)$. It describes the long-run behaviour of this prior. Production prevalence, the initial regime prior and the evaluation mixture are separate quantities. Specify the initial prior independently.

A particle in B needs at least two updates to reach C, and conversely. Real disturbances could hand over within one interval. Routing through A can delay recognition of such a handover. Simultaneous elevated demand and slow publishing remain outside the primary state space. The filter receives neither true labels nor the intervention time. At the decision cut, each primary action forecast holds its sampled regime fixed for the full horizon.

The [scientific specification](../../particle_filter/SCIENTIFIC_SPEC.md#transition-destinations) records the matrix and deterministic verification cases.
