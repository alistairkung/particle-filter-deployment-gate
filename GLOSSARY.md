# Outbox telemetry and deployment risk

These terms describe what the outbox reveals before a change in Java traffic share and how to evaluate the consequences of that change.

## Physical quantities

**Background intensity** is the expected number of external background queries arriving per second under the current operating condition.

**Realised background demand** is the total DB service time requested by background queries arriving in an interval, divided by the interval duration. It counts each arriving query's full service requirement, including work outstanding at the interval end.

**Unfinished DB work** is the service time remaining on the active DB visit plus the service time of visits already waiting for the DB. A pending outbox row can also owe a future publisher visit that has yet to join the DB queue.

**Publisher condition** is the operating condition governing the publisher's off-DB acknowledgement wait. An acknowledgement already in progress keeps the duration assigned when it began.

**Operating regime** is the combination of background query intensity and publisher condition in force at a given time. Realised query demand and unfinished DB work can vary within a single regime.

## Observation history

**Pending outbox row** means an event whose insert has committed and whose publication mark has yet to complete. A claimed or acknowledged event can still be pending.

**Oldest pending age** is the elapsed time since the earliest commit among currently pending rows. A measured empty outbox has age zero.

**Publication throughput** is the number of completed publication marks in a measurement interval divided by its duration.

**Completed publication delay** is the elapsed time from a row's commit to completion of its publication mark. The interval median is undefined when there are no completed marks.

**Available observation history** comprises the measurements delivered by the decision time, with their measurement intervals and availability times. An empty outbox, an interval with no marks, and an undelivered measurement describe different events.

## Paired outcomes

**Action risk** is the conditional probability of a horizon write-SLO breach under a specified routing action, given the available observation history and the study's workload law.

**Incremental promotion risk** is the promotion action risk minus the hold action risk for the same history. Under matched futures it equals the probability of a promote-only breach minus the probability of a hold-only breach.

**Avoidable breach** is a paired outcome in which promoting breaches and holding avoids the breach on the matched future.

**Safe promotion opportunity** is a paired outcome in which promotion completes the horizon without a write-SLO breach. It includes both neither-breach and hold-only outcomes.
