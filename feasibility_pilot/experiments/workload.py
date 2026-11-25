"""Apply the pilot's route-blind placebo and assigned background workloads."""

from dataclasses import replace

from simulator.workload import admissions, primitive_tape


def inputs(c, family, root, phase, share, extra_rate=0):
    tape = primitive_tape(
        c,
        (family, root),
        phase,
        reference_rate=0 if family == "C5" else c.reference,
        extra_rate=extra_rate,
    )
    routed_config = replace(c, java_ms=c.ruby_ms) if family == "C1" else c
    return admissions(tape, share, routed_config)
