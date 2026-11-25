# Give the three starting regimes equal prior probability

Before the first five-second interval, assign each particle's regime prior probability $1/3$ for each of A, B and C. Apply this law once per history, identically for every root and for the 60-, 120- and 240-second half-lives. Subsequent updates propagate the current posterior through the transition and observation models.

Equal weights express a chosen symmetry over the three hypotheses and give the two disturbance regimes a combined probability of $2/3$. Other choices of symmetry could give different weights. These probabilities have no empirical interpretation as production prevalence. The graph's stationary distribution describes its long-run behaviour and supplies no initialisation information. [Transition graph](0007-regime-transition-destinations.md)

Keep the true reference-only start, assigned condition and future conditions private. Give all forecasting methods the same permitted information and allow each model class to express its own assumptions. Report possible dependence on the initial prior as a limitation. The uniform prior is fixed before protected evaluation. The [cold-start decision](0009-known-empty-start.md) specifies initial unfinished work and publisher memory.
