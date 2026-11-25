# Estimate both action risks after an established operating history

Each independent history starts at 25% Java and 40 writes/s. Its first 30 seconds use reference conditions. For the final 30 seconds the assigned condition is reference (3.5 queries/s, 2 ms acknowledgement), established contention (5.5/s, 2 ms), or publisher slowdown (3.5/s, 50 ms). The condition persists through the ordinary evaluation horizon. A separate surprise scenario changes background demand at the decision, so the prefix can test leakage.

At the cut the evaluator records realised background demand over the preceding 30 seconds and total unfinished DB work at that instant. Each fresh, unselected root supplies one matched future for hold at 25% and promote to 60%, scored with the frozen 120-second write-SLO endpoint. This design gives both actions the same inherited physical state and future external draws.

Promotion-breach probability is the primary forecast-accuracy target. The study also assesses hold-breach probability and the signed incremental promotion risk. All four paired outcomes are retained because a hold-only breach contributes negatively to the increment, and a breach under both actions contributes zero.
