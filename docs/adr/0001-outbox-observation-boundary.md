# Observe committed outbox inserts and completed publication marks

The telemetry sufficiency study assumes a passive feed of committed outbox inserts and completed publication marks. A commit makes a row visible, and the matching mark makes its completed publication delay measurable. This contract permits measuring rows that enter and leave between samples without adding monitoring work to the shared DB.

Every five seconds the observer reports pending count and oldest age after all events at the interval end, plus mark throughput and median completed commit-to-mark delay over the interval open on the left and closed on the right. All four measurements derive from the same lifecycle. The primary study assumes delivery at the interval end and records availability separately so delayed or missing delivery can be tested. Only delivered observations enter the decision history. The evaluator owns physical state, workload settings, identifiers and paired outcomes.
