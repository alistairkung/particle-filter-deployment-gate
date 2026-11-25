# AIMS5701 feasibility pilot

**Status — ready for approval, 25 November 2026.** This document records the agreed scientific decisions. The finite warm-ups, queue-check tolerances, C1/C4 run budgets, descriptive intervals and seed conventions complete the contract as proposed execution defaults. Implementation starts after explicit approval of this document and the [implementation plan](IMPLEMENTATION_PLAN.md).

## Purpose

The project author's production services had a **99.99% availability target (four nines)** during the beneficiary-service migration. That target explains why resource regressions mattered during progressive live rollout. A rewritten service could return the correct result and still make requests on both routes wait longer for the shared database. The historical figure describes the target the services were held to. Achieved availability, the historical SLA calculation and rare-event availability modelling are outside the pilot's scope.

The historical migration programme used heterogeneous publisher implementations, including Ruby-based publishers, and some service rewrites coincided with event-driven redesigns. The pilot deliberately gives both routes one logical publisher with the same claim and mark costs. Holding this part of the experiment fixed isolates the change in synchronous DB work during live routing. The simulated architecture makes no claim to reconstruct a particular historical deployment.

Test whether changing Java's share of live, functionally equivalent writes can produce a measurable change in pooled write-SLO breach risk through contention for a shared DB resource. The pilot also tests whether an independent publisher slowdown can worsen outbox telemetry while writes remain healthy.

Historical shadow execution validated and logged requests without writes or downstream side effects. The experiment represents the subsequent live weighted-routing phase, with equivalent business effects on both routes. A route label alone should leave the complete physical trace unchanged. A configured synchronous work difference changes offered load by construction. Whether it produces consequential breach risk, whether background demand amplifies the increment, and whether outbox lag can also arise under healthy writes are falsifiable claims.

The amended rollout plan governs the shared-outbox model. The review's earlier Java-only event examples were superseded. This pilot uses the approved single routing step and 400 independent histories to estimate population-average effects. Historical brainstorming supplies motivation. The main research question and the later forecasting study retain their existing scope. [Rollout plan](https://github.com/alistairkung/msc-ai-learning/blob/1926d104a81388387f9314308fa3e4f0cab886d3/lesson_logs/aims5701/projects/particle_filter_rollout_risk_plan.md) · [Feasibility review](https://github.com/alistairkung/msc-ai-learning/blob/1926d104a81388387f9314308fa3e4f0cab886d3/lesson_logs/aims5701/projects/particle_filter_feasibility_review.md) · [Brainstorming history](https://github.com/alistairkung/msc-ai-learning/blob/1926d104a81388387f9314308fa3e4f0cab886d3/lesson_logs/aims5701/research_brainstorming_2026_11_09.md)

## Approved validation and controls

The isolated queue check is a prerequisite. The five scientific controls follow the amended research plan. All five use the same simulator and experiment runner, with configuration selecting the workload and intervention. [Committed experimental design](https://github.com/alistairkung/msc-ai-learning/blob/1926d104a81388387f9314308fa3e4f0cab886d3/lesson_logs/aims5701/projects/particle_filter_rollout_risk_plan.md)

| Check | Comparison and evidence |
| --- | --- |
| Prerequisite — isolated queue correctness | Compare an isolated single-server queue with its analytical stationary behaviour using a declared warm-up, independent seeds and a predeclared tolerance. A failure blocks interpretation of the scientific controls. |
| C1 — equal-work routing placebo | Match synchronous service draws, visits, event generation and scheduling. With paired inputs, require identical physical write and event timings and breach indicators across shares. Exercise both 0% and 100% Java with a real outbox row for each eligible commit. |
| C2 — genuine migration effect | At reference background demand, apply a predeclared increase in Java synchronous DB work. Compare holding the current share with promotion. Report both absolute risks and the paired risk difference with uncertainty. |
| C3 — background contention interaction | Increase background demand at the decision point for newly arriving queries. Estimate how much this changes the paired promotion effect relative to C2, with uncertainty. |
| C4 — background-only overload | Configure independent background queries so that total offered DB work exceeds capacity even with all writes routed to Ruby. Report absolute breach risk at 0% Java, at the current share under hold and at the proposed share under promotion. Report the paired promote-minus-hold risk difference with uncertainty. Record offered work and DB queue growth over the horizon, including the trajectory for each share. |
| C5 — publisher-only slowdown | At fixed 25% Java and zero background queries, compare 2 ms versus 50 ms off-DB acknowledgements. Use the paired experiment and isolation checks below. |

For C4, hold and promote start from the same decision state. Absolute risk describes how often a branch breaches. The paired difference estimates how much setting a higher routing share changes that probability. Near the ceiling of a binary breach metric, promotion can increase waiting even when the estimated risk difference is small. Its sign and magnitude remain empirical outcomes.

### Queue prerequisite and physical validation

Use the same SimPy FIFO DB service path as the full model. Disconnect the application commit effects, publisher and background stream. With service rate 100/s, run Poisson arrivals at 50/s and 90/s under exponential service (M/M/1) and fixed 10 ms service (M/D/1). For each of the four cells use 32 independent seeds. Start empty, discard the first 10,000 arrivals, and score the next 50,000 arrivals through completion. Measure queue wait from arrival to service start. The stationary mean waits are respectively 10 and 90 ms for M/M/1 and 5 and 45 ms for M/D/1. At each arrival rate the higher service-time variance doubles expected waiting.

Report each seed mean, the equally weighted mean and a two-sided 95% Student-t interval across the 32 seed means. Pass when the entire interval in each cell lies within ±10% of its analytical mean. This is a predeclared engineering equivalence tolerance. Keep failed checks and distinguish disagreement from insufficient precision. The mixed workload with feedback from the publisher has its own empirical outcomes.

Use one FCFS queue with route-independent ordering by enqueue time and stable insertion order for ties. At write completion apply the business change and create exactly one event atomically. An idle publisher wakes on a committed row, takes the oldest unclaimed row, queues its claim, releases the DB at claim completion, awaits the sink, queues its mark and removes the row from pending at mark completion. Record commit, completed claim, acknowledgement and completed mark as distinct timestamps. No batching, polling, retries, drops, autoscaling or request cancellations. On every measured window and at the cut verify:

- Pending at end equals pending at start plus eligible commits minus completed marks. Count in-flight rows once. Verify unique event IDs, at most one mark per event and commit ≤ claim ≤ acknowledgement ≤ mark for each observed lifecycle prefix.
- The server provides at most one unit of concurrent service. Account for busy time by workload, including partial service. Preserve FIFO order, residual work, scheduled arrivals, outstanding writes, pending rows and publisher phase across deterministic prefix replay. Compare physical state and a same-policy continuation to an uninterrupted run.

### Starting conditions and control budgets

For the scientific controls, initialise empty and run a **60 s finite warm-up** before the decision. The history law includes that initialisation. No claim of stationary DB or publisher queues is required. Draw independent future inputs for each root and pair them across the interventions defined for that root.

- **C1.** Use 16 independent reference-demand roots with equal service draws at 25% Java in warm-up. Replay each into 0%, 25%, 60% and 100% Java. Require exact equality of route-blind physical traces, pooled timings, event lifecycles and scores, including through the drain. Check route assignment at both extremes for post-cut arrivals and one real event per eligible commit. Keep inherited route assignments at the cut.
- **C2/C3.** Use the 400 four-way roots below. Check the route-specific synchronous service transform, identical publication cost, expected load arithmetic and matched input identities. Report the signed interaction and its interval for every result. Either sign, including zero, is scientifically admissible.
- **C4.** Use 400 independent roots with the migration-condition service laws. For the 25% hold/60% promote pair warm up at 25% and reference background demand. For the all-Ruby absolute-risk baseline warm up at 0% under reference demand and stay at 0%. Match exogenous inputs by root across all three arms. Apply 12.5 background arrivals/s after the cut in all three. Check the offered overload at 0%, report queue and unfinished-work growth, and reconcile delayed publisher visits against the outbox. Distinguish work already queued at the DB from claim/mark work still owed by admitted writes. Require valid workload accounting and the three branch results. No prescribed sign of queue growth on an individual root or of the paired breach increment.
- **C5.** Use a separate set of 400 paired roots with the zero-background nominal warm-up. Apply both C5 criteria below.

## Approved routing intervention

For C2 and C3, warm up each independent root history once at **reference background demand and 25% Java**. From that root's decision point, run all four continuations for the full 120-second horizon.

| Background demand for new queries after the cut | Hold at 25% Java | Promote to 60% Java |
| --- | --- | --- |
| Reference | C2 hold | C2 promote |
| Higher | C3 hold | C3 promote |

All four branches inherit identical physical state at the cut, including DB queue order, the active operation and its residual work, outstanding writes and their routes, pending outbox records and commit times, and the publisher's in-flight event, phase and remaining work or acknowledgement time. Queued background queries keep their assigned work and an active query keeps its residual work. Only newly arriving queries in C3 follow the higher-demand law.

Apply the assigned routing probability immediately to new requests and keep it fixed through the horizon. Across all four branches, match future application arrivals, routing uniforms and underlying service draws by stable identity. Keep randomness independent across roots.

**Background query inputs.** Generate one reference Poisson arrival stream per root, independent of the application and routing draws. Continue that reference stream in all four branches with identical query identities, arrival times and service demands. At the cut, start a second, independent Poisson stream of extra arrivals. Only the two C3 branches receive it, and both receive exactly the same extra jobs. Each query makes one FIFO DB visit with an independent exponential service requirement. Draw its work once per identity and reuse it wherever that query appears. The mean service time is common to both streams. The higher total arrival rate is the reference rate plus the extra rate.

This rate-only intervention increases the frequency of otherwise equivalent queries. Poisson arrivals and exponential work are synthetic simplifying assumptions. The isolated analytical queue prerequisite validates the scheduling mechanics under its own test conditions. Production workload calibration would require independent measurements.

For each root, subtract the C2 promote-minus-hold breach indicator difference from its C3 counterpart. Average these within-root contrasts to estimate the interaction. It measures how an unexpected post-decision demand increase changes the effect of promotion from the same starting backlog. All four branches have exactly the same pre-decision outbox observations, so this comparison cannot establish advance warning of that increase.

Reserve promotion into established contention for a later forecasting experiment. There, elevated demand begins before the cut and can affect the outbox observations available at decision time. C4 remains a separate pilot control for all-Ruby overload and risk saturation.

This larger step tests whether the simulator exhibits a consequential routing effect. It is an experimental choice with no implied production rollout recommendation. Reserve 25% to 35% for the later forecasting experiment. Keep the 25% versus 60% comparison fixed regardless of pilot outcomes.

## Frozen numerical workload

| Component | Distribution and units |
| --- | --- |
| Shared DB | One non-preemptive FIFO server with capacity one DB work-second per elapsed second. |
| Application | One Poisson stream at 40 eligible writes/s, independent of background queries. Each request takes one synchronous DB visit and creates one event at commit. |
| Ruby synchronous visit | Exponential business work with mean 11 ms plus a fixed 1 ms atomic outbox insert in the same visit. Mean total 12 ms. |
| Java synchronous visit | In the migration condition, exponential business work with mean 15 ms plus the same fixed 1 ms insert. Mean total 16 ms. In C1, both routes use the exact Ruby draw for that request. Across migration forks, scale the same underlying business-work draw by the assigned route's mean. |
| Publisher | One FIFO worker. Each event requires a fixed 1 ms DB claim, a fixed nominal 2 ms sink acknowledgement with the DB released, and a fixed 1 ms DB mark after acknowledgement. Both routes owe 2 ms of publisher DB work per commit. C5 changes the off-DB delay alone. |
| Background | Each query takes one exponential DB visit with mean 40 ms. The reference rate is 3.5 queries/s. C3 adds 2.0 queries/s for a total of 5.5/s. The separate C4 overload rate is 12.5/s total. |

With Java share $r$ and background query rate $q$ per second, expected offered DB work per second is $0.040q + 40[(1-r)0.012 + r0.016 + 0.002]$ in the migration condition. The synchronous terms each include the insert once. The final term accounts for the claim and mark owed by each offered write, even when publication occurs later.

| Setting | Background work as % of DB capacity | Total at 0% Java | Total at 25% hold | Total at 60% promote |
| --- | ---: | ---: | ---: | ---: |
| C2 reference | 14% | — | 74.0% | 79.6% |
| C3 higher | 22% | — | 82.0% | 87.6% |
| C4 overload | 50% | 106.0% | 110.0% | 115.6% |

The routing step adds 5.6 percentage points of offered DB load in each setting. Observed utilisation is elapsed time serving DB work within the measurement interval divided by interval duration, bounded by 100%. Report both quantities, including service that crosses an interval boundary in proportion to its overlap.

## C5 publisher-only slowdown

Warm up each C5 root at 25% Java with 40 eligible writes/s, zero background queries, the migration-condition synchronous service laws and one FIFO publisher using fixed 2 ms sink acknowledgements. From that single physical state, fork two 120-second continuations. Keep the nominal branch at 2 ms. In the slow branch use 50 ms for acknowledgements initiated after the decision. An acknowledgement already in progress keeps its residual time in both branches. Both branches retain 25% routing, the same application arrivals, routing uniforms and per-request DB service demands, and fixed 1 ms DB claim and 1 ms DB mark visits. Disable retries and extra polling.

**Required DB-release test.** Use a deterministic trace with one claimed event undergoing a 50 ms acknowledgement, an otherwise idle DB and a short application write arriving during the wait. Require the write to acquire the DB and commit before the acknowledgement finishes. Verify that the publisher releases the DB at claim completion and requests it again for the mark only after acknowledgement. In a fork fixture with an acknowledgement already in flight, require its completion time to match across branches and the next acknowledgement to use the assigned branch delay.

**Observations.** At the cut and each window end, record pending rows (including an in-flight row until marked) and the oldest pending row's age since commit. In each window report the distribution of commit-to-mark delays for rows actually marked, the number marked, and throughput in marks/s. Distinguish an empty outbox from a window with no marks. Over the horizon report actual claim and mark visit counts and their DB busy time. Slower acknowledgements can postpone visits beyond the horizon. Preserve this change in realised relay work. Use the common scored cohort to compare write-latency distributions, deadline-miss counts and fractions, and absolute breach risks with their paired slow-minus-nominal difference. Compare DB waiting by workload in the same windows.

### C5 write-health evidence

Use **400 independent C5 root histories**, each supplying one matched nominal/slow pair. Score one binary horizon breach outcome per branch per root. Report the four paired outcome counts, branch breach frequencies and paired risk difference. Apply all thresholds to unrounded bounds after the fixed 400 pairs. These thresholds are synthetic criteria for interpreting this control.

Let $n=400$. For a count $x$ across roots, write $U_a(x)$ and $L_a(x)$ for one-sided Clopper–Pearson bounds with tail probability $a$. With $Q(q;s,t)$ denoting a beta quantile, use $U_a(x)=Q(1-a;x+1,n-x)$ for $x<n$ and $U_a(n)=1$. Use $L_a(x)=Q(a;x,n-x+1)$ for $x>0$ and $L_a(0)=0$.

- **Absolute health.** In each branch count all roots that breach, including roots in which both branches breach. Require $U_{0.05}(x)<0.05$ separately in the nominal and slow branches.
- **Incremental harm.** Let $s$ count pairs where only the slow branch breaches and $m$ count pairs where only the nominal branch breaches. The estimated slow-minus-nominal risk difference is $(s-m)/400$. Use the one-sided upper bound $U_{0.025}(s)-L_{0.025}(m)$ and require it to be strictly below $0.02$. Each discordant count is marginally binomial over roots. Bonferroni gives at least 95% coverage for their probability difference, allowing dependence within a pair. Agreements remain in the denominator.

Claim write health only when all three upper limits clear their respective margins. If an upper limit reaches or exceeds its margin and the data do not establish a violation, report the write-health result as **inconclusive**. To distinguish established material deterioration, use the corresponding paired lower bound $L_{0.025}(s)-U_{0.025}(m)$: a value at or above $0.02$ establishes material harm and C5 has failed to isolate publisher-specific sensor ambiguity. A branch with $L_{0.05}(x)\ge0.05$ fails the absolute-health requirement; absolute risk alone cannot attribute harm to the slowdown.

### C5 outbox deterioration and interpretation

Use **oldest pending age at 120 s** as the sole outbox deterioration criterion. Take the horizon snapshot after processing all events timestamped at 120 s and before the diagnostic drain. A row is pending from commit until it is marked processed. Count unclaimed rows, claimed rows, rows awaiting acknowledgement and acknowledged rows awaiting completion of the DB mark. For a nonempty outbox, oldest pending age is the snapshot time minus the earliest commit time among these rows. For an empty outbox, record age 0 s and an explicit empty flag.

For each of the 400 matched C5 histories, subtract nominal oldest pending age from slow oldest pending age. Count one success if this difference is **at least 5 s**, including equality. Let $k$ be the number of successes. Report $k/400$ and the one-sided 95% exact Clopper–Pearson lower bound $L_{0.05}(k)$ defined above. Require the unrounded lower bound to **exceed 0.50**. A boundary check for the fixed design is that 216 successes fail and 217 pass. The 5 s margin is a synthetic experimental threshold.

C5 establishes publisher-specific sensor ambiguity in this setting only when **both** write health and outbox deterioration meet their criteria. Report actual backlog, oldest pending age, throughput, completed commit-to-mark delays, DB claim and mark counts, and corresponding DB work regardless of the result. These diagnostics explain how the publisher behaved. They provide no alternative route to passing C5. Preserve the acknowledgement delays (2 ms and 50 ms), workload, thresholds and original results. The +10-point migration GO rule applies only to C2 and C3.

## Scored cohort and horizon boundary

Measure time in seconds relative to the decision. For an eligible write with arrival time $a$ and deadline $d=a+0.250$, include it in both the primary score and the application latency distribution exactly when $a>0$ and $d\le120$. Earlier and later writes still take part in the physical simulation if admitted. Assign a scored write once, by deadline, to $(0,30]$, $(30,60]$, $(60,90]$ or $(90,120]$. A commit at or before its deadline is on time. A later commit or a write still unfinished after all events at its deadline have been processed counts as a miss.

After processing all events timestamped at 120 s, freeze the horizon breach indicator, all window denominators and misses, DB busy time and utilisation, ordered queue and residual work, outbox telemetry and every other horizon measurement. Run the following diagnostic drain only if scored writes are outstanding.

1. Close both external arrival streams after 120 s. Preserve every admitted operation and the publisher state. Let queued work, active service, and internal commit, claim, acknowledgement and mark events proceed under the existing discipline.
2. Stop when the last scored write commits or at **240 s after the decision**, whichever comes first. This gives at most **120 s of additional simulated time**. Process completions at the cap before counting unresolved writes.
3. Use post-horizon commits solely to record arrival-to-commit latency for scored writes. Label the result as a drained continuation. For writes unfinished at the cap, report their count and right-censoring time (240 s minus arrival time). Retain them in the cohort. When censoring leaves a requested percentile indeterminate, report it as such with any supported lower bound.

### Required deterministic boundary tests

| Fixture (seconds after decision) | Expected observation |
| --- | --- |
| Eligible arrivals before, exactly at and just after 0; an ineligible arrival after 0 | Only the eligible post-decision arrival enters the cohort. All admitted work still competes for the DB. |
| Deadlines just before, exactly at and just after each of 30, 60, 90 and 120 | Assign the first two to the window ending there. Assign the third to the next window at internal boundaries and exclude it from scoring at 120. Each scored write contributes to exactly one denominator. |
| Arrival 119.750, commit 120.000; repeat with commit 120.001 | The first write is on time in window four. The second is a frozen miss in that window and has diagnostic latency 251 ms. |
| Arrival 119.700, commit 120.400, with queued work and a publisher operation in flight at 120 | Freeze one miss at the 119.950 deadline and retain 700 ms diagnostic latency. Verify that admitted work carries into the drain, both external arrival streams are closed, and the frozen horizon measurements match the snapshot taken before the drain. |
| Two scored writes outstanding at the cap, one scheduled to commit exactly at 240 and the other later | Record the first completion. Report one censored write with its latency lower bound; both remain in their original scoring denominators. |
| Empty window, 1 miss in 20 writes, 2 misses in 20 writes | Only the last window triggers a breach. Exactly 5% misses is below the strict breach condition. |

## Diagnostics for interpreting the primary endpoint

- For each branch, show the distribution of arrival-to-commit latency in the scored cohort, pooled over routes, with counts and p50, p95 and p99. Also break down latency by route with sample counts. Apply the drain and censoring rules above.
- Report misses, eligible denominators and deadline-miss fractions by branch and 30-second window, plus the branch-wide fraction. Each horizon still contributes one binary outcome to the primary risk estimate.
- Report DB waiting time by workload and the time series of queued operations and unfinished DB work. Show values at the cut and horizon end and their change. Include remaining service on the active visit in unfinished work. Distinguish visits still waiting at the horizon from visits whose wait has been fully observed.
- Align expected and realised offered work by workload with observed DB utilisation. For outbox telemetry, report pending count, oldest pending age and completed commit-to-mark delays. Report absolute horizon breach risks and paired routing increments with the prescribed uncertainty.

If hold and promote both breach almost always, report any saturation of the binary endpoint alongside the measured increment and its uncertainty. Latency tails, miss fractions and queue growth can reveal additional harm in that case. The primary endpoint and its +10-point GO criterion still govern the decision. Preserve the original workload, deadline, horizon and results. Any revised parameters belong to a separately identified, predeclared exploratory follow-up.

## Approved practical-effect threshold

The primary outcome is one binary indicator per continuation. It equals one if, in any of four consecutive 30-second windows, more than 5% of the eligible pooled writes whose deadlines fall in that window miss their 250 ms arrival-to-commit deadline. Count post-decision arrivals whose deadlines fall within the 120-second horizon. A write still unfinished at its deadline counts as a miss. An empty window cannot trigger a breach.

The smallest practically consequential promotion effect is **+10 percentage points** in the probability of that horizon outcome. This is a synthetic threshold chosen for the feasibility decision. Estimate the effect as the mean of the paired promote-minus-hold indicators. For example, a rise from 15% under hold to 25% under promotion means ten additional horizons with a latency breach per hundred simulated horizons on average. This effect measures neither a 10% loss of availability nor an acceptable production deployment tolerance.

Exactly two migration comparisons can qualify for GO. They use the approved routing intervention under reference background demand (C2) and higher background demand (C3). For each setting report absolute risk at each share, the paired promote-minus-hold risk difference and both directional bounds. Report the interaction between routing and background demand separately.

Simulator correctness is assessed through validation, replay, conservation and placebo checks. The practical-effect threshold applies only to the migration evidence for GO. C1, C4 and C5 retain their own scientific purposes. Report smaller, zero, negative and inconclusive effects with their uncertainty. Preserve the frozen workload parameters and original results. Any later parameter exploration must be labelled separately and cannot qualify the original pilot for GO.

## Migration comparisons and uncertainty

**Independent sampling unit.** Draw 400 independent root histories from the same predeclared history-generating law. Each root supplies one matched hold/promote pair at each background setting, giving four horizon outcomes per root and 1,600 scored continuations. Pair exogenous inputs within each routing comparison and retain the four outcomes together under one root ID. Use fresh randomness across roots. Any resampling for descriptive intervals or the background interaction must resample whole roots with their four outcomes together.

**Paired estimand.** At each setting, let each branch outcome be 1 for a pooled write-SLO breach within the horizon and 0 otherwise. The target is the mean promote-minus-hold difference over the root and future workload law. Count roots in all four paired outcome cells. Let $x_+$ count roots where only promote breaches and $x_-$ count roots where only hold breaches. For $n=400$, the estimated risk difference is $(x_+-x_-)/n$. Its population value is $p_+-p_-$, where each probability refers to that paired outcome. Use all roots in the denominator, including agreements.

**Simultaneous one-sided bounds.** Apply the exact binomial component functions defined under C5 with tail probability $a=0.0125$ and denominator 400. Within each setting compute both directional bounds on the paired risk difference:

$$
B_{\rm lower}=L_a(x_+)-U_a(x_-),\qquad
B_{\rm upper}=U_a(x_+)-L_a(x_-).
$$

Each discordant count is marginally binomial across roots. For either direction, the two component tail errors sum to at most 0.025 per comparison and 0.05 across the two settings by Bonferroni. This remains valid when the comparisons share roots or a discordant cell is empty. Report the lower family and upper family separately as simultaneous one-sided 95% bounds. The 95% coverage statement applies to each direction separately. [Exact binomial bound reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html)

For descriptive uncertainty, give two-sided 95% exact binomial intervals on branch risks. For the signed C3 interaction and C4 routing increment, give percentile intervals (2.5th and 97.5th percentiles of 10,000 resampled means). Sample independent root IDs with replacement and keep their complete branch vectors together. Label these as descriptive, including any degenerate bootstrap interval. The exact directional bounds govern C2/C3 decisions.

## Verdict and stopping conditions

Assess once after completing the fixed budgets. First verify queue validation, deterministic replay, DB and outbox invariants, exact C1 placebo, C2/C3 input and workload integrity, C4 overload accounting and measurements, and both C5 criteria. Keep a separate validation status, control-by-control findings and scientific verdict.

| Verdict | Decision rule using unrounded bounds |
| --- | --- |
| **GO** | Validation and required controls pass, and at least one eligible migration comparison has $B_{\rm lower}\ge0.10$. The evidence supports investigating forecasting in the specified simulator. |
| **NO-GO** | Validation and required controls pass, and **both** eligible comparisons have $B_{\rm upper}<0.10$. This is a valid negative result for the frozen workloads, 25%→60% intervention and binary write-SLO endpoint. |
| **MODIFY / inconclusive** | Every other case. Report whether uncertainty spans the margin, endpoint saturation limits interpretation, C5 fails to isolate the sensor, or a run or validation check prevents evaluation. An upper bound exactly at +0.10 keeps NO-GO unestablished. |

If a reproducible defect corrupts the comparison, record the failed validation, fix the implementation and repeat the affected checks and runs with the original inputs. If unresolved, report the invalid or incomplete evidence explicitly under MODIFY. For a correctly executed experiment preserve smaller, negative, saturated and inconclusive results. Publish latency distributions, DB waiting, deadline-miss rates, absolute breach probabilities, paired increments and uncertainty for valid runs under every verdict. These diagnostics can explain a negative binary result. They cannot reverse the decision rule.

Keep the workload, endpoint, routing step, effect margin and replication budget fixed. Any new parameters or additional histories belong to a separately identified and predeclared follow-up. End the pilot with the feasibility report. A GO decision does not authorise particle-filter implementation.

## Reproducibility and limits of inference

Use master seed **57012026**. Derive deterministic named substreams for the queue prerequisite, C1, C2/C3, C4, C5, bootstrap and development fixtures. Within a family key streams by root ID, phase and source (application arrivals, routing, business work, reference queries, extra queries and query work). Key or pre-generate service draws by stable request/query identity. Forks deliberately share those inputs. Record the generator algorithm, entropy or spawn keys, numeric time representation and tie ordering. Route-independent tie order and explicit inclusive timestamp handling must pass the boundary fixtures.

Save the frozen config, its digest, replay seed manifest, source revision and dirty-state information, dependency versions, counts of completed roots, and per-root branch records sufficient to recompute every bound and verdict. Save observations and evaluator-only state in distinct fields or tables. Hidden background demand, residual queue state and future shocks are available to the evaluator. A future forecaster must rely on as-of-time observations and permitted routing and arrival inputs.

The FIFO server represents exclusive service and head-of-line blocking. It supplies no estimates of Aurora scheduling, SQL plans, connection pools, locks, or production workload distributions. A 60 s prefix and 120 s horizon define a transient experiment. A valid NO-GO leaves other workload laws, smaller routing steps, other endpoints and the general value of particle filtering open. The relative predictive value and lead time of outbox and latency sensors require the later study. Real infrastructure, dashboards, production gates and estimators are outside the pilot.
