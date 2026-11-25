# Propagate unfinished DB work at FIFO admissions and use observed pending count for publisher memory

The particle state at a five-second boundary is $X_t=(\lambda_t^{BG}, W_t, P_t)$, comprising background intensity, unfinished DB work and publisher condition. This approximates the physical simulator by averaging over omitted details, including the publisher phase, residual acknowledgement time and ordering of queued visits.

Within each particle, propagate its own hypothetical FIFO trajectory. If $\delta_j$ seconds elapse since the previous admission and the new visit requires $S_j$ seconds of service, the work just after admission is $W_j=\max(0,W_{j-1}-\delta_j)+S_j$. Drain any residual work down to zero at unit rate between admissions, including the time from the last admission to the observation boundary. Pending rows give rise to DB work as the serial publisher submits each claim and mark. An acknowledgement occupies the publisher off the DB. A visit already queued is already included in $W$. Evaluator queue contents and publisher progress stay private.

## How the pending count enters the update

Let $H_t$ be the readings delivered by the start of an interval, $c_t$ the most recent measured pending count available in that history, and $a_t$ the known routing control. Keep the measurement time with the count so that an older reading remains distinguishable from a current one. Average over compatible publisher progress when predicting the next interval. A missing reading supplies no measured count of zero.

For the primary, immediately delivered observations, write $\pi_t(dx)=\Pr(X_t\in dx\mid H_t)$ and let $Y_{t+1}$ be the four measurements at the end of the interval. The approximate model factors their joint prediction as

$$
\Pr(X_t\in dx,X_{t+1}\in dx',Y_{t+1}\in dy\mid H_t,a_t)
= \pi_t(dx)\,K(dx'\mid x,c_t,a_t)\,g(dy\mid x,x',H_t,a_t).
$$

Here $K$ uses the count available at the start, with its measurement time when applicable. The joint likelihood $g$ describes the ending reading conditional on the transition and past history, averaging over any omitted within-interval path. The ending count enters the update once, as part of that joint reading. At the next step it can condition propagation as known history, with no additional likelihood factor for observing it again. Later deliveries are usable only from their availability time. Beyond the decision cut each forecast evolves its own pending count from hypothetical commits and completed marks.

## Evidence required for augmentation

On held-out development roots, compare predicted five-second mark counts and interval-end unfinished work with evaluator replay, including histories with similar starting counts and different publisher phases. Use a development-only diagnostic with access to phase and residual acknowledgement time to measure how much they improve physical predictions or hold and promote risk calibration. The deployable filter uses only permitted observations. If the compact model shows a repeatable, consequential error, document the evidence and the smallest inferable publisher-memory correction before changing the state.
