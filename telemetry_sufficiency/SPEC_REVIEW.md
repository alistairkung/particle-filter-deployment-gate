# AIMS5701 · Specification review

This record covers the pre-implementation review of the [scientific protocol](SCIENTIFIC_SPEC.md) and [implementation contract](IMPLEMENTATION_SPEC.md) against the frozen pilot at `bb20ebe`. Separate agents conducted the scientific and engineering reviews. The specifications remain subject to study-owner approval.

| Review finding | Resolution in the specification |
|---|---|
| A failed fit could be interpreted as weak evidence. A positive numerical gain can also have zero variance within strata. | Missing or nonfinite required scores and unrepaired fitting failures are validity failures. A genuine zero SE in an otherwise complete evaluation leaves the relevant claim unsupported. READY uses the full two-claim rule. |
| Recognising a configured intensity can improve pooled demand forecasts while revealing little about fluctuations among roots in one condition. | The confirmatory demand comparison holds publisher condition nominal. Descriptive comparisons also score demand and D against each condition’s own mean. The report distinguishes inference across conditions from inference within them. |
| Temporal improvement and snapshot equivalence could be confused. | Report paired history-versus-current differences on action, incremental, physical and class targets. Uncertain temporal improvement supplies no equivalence claim. A limitations verdict identifies a defensible follow-up question. |
| The frozen pilot’s comparison CLI writes a reproduction receipt into its first input directory. | Call the read-only comparison function, write its receipt under the new study and verify the original pilot tree against its frozen revision. |
| An audit starting from stored work sums or miss counts can reproduce a biased extractor’s answer. | Independently replay all protected histories and branches from named draws. Check applied inputs, rebuild as-of observations, physical targets and deadline counts before reconstructing scores. Corrupted evidence must fail audit. |
| The existing routed tape omits the raw uniforms and business-work variates needed to verify common-input pairing. | Bind raw draws to actual admissions for each branch. Compare expected route and work after applying each action’s threshold. |

The engineering reviewer rechecked the revised contracts and approved them for user review. The final text also pins the overlap statistic, empty calibration bins, evaluator condition encoding and a regression fixture for reference queries during publisher slowdown.

The scientific reviewer accepted development-only model selection, the stronger best-constant comparison, and the two one-sided 97.5% Bonferroni bounds as an **asymptotic** familywise procedure conditional on the model lock. The review establishes the design and evidence required for a verdict. Experimental validity and telemetry information remain to be tested after approval.
