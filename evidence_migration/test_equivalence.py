"""Regression probes for silent changes which leave root and row counts intact."""

import hashlib
import io
import unittest

import numpy as np
import pyarrow as pa

from .pilot import _sample_groups, derived
from .source import exact


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        elapsed = np.array([125_000_000, 250_000_001], dtype=np.int64)
        java = np.array([False, True])
        censored = np.array([False, False])
        self.archive = io.BytesIO()
        np.savez(self.archive, hold_ns=elapsed, hold_java=java, hold_censored=censored)
        self.sample = pa.table(
            {
                "family": ["C23"] * 2,
                "root": pa.array([0, 0], type=pa.int32()),
                "arm": ["hold"] * 2,
                "sample_index": pa.array([0, 1], type=pa.int32()),
                "elapsed_ns": elapsed,
                "java": java,
                "censored": censored,
            }
        )
        checksum = hashlib.sha256(
            elapsed.tobytes() + java.tobytes() + censored.tobytes()
        ).hexdigest()
        self.info = {"hold": (2, 0, checksum)}

    def verify_samples(self, sample):
        self.archive.seek(0)
        with np.load(self.archive, allow_pickle=False) as source:
            list(_sample_groups(sample, "C23", 0, self.info, source))

    def test_detects_one_changed_tick_route_or_censor(self):
        self.verify_samples(self.sample)
        for field, values in (
            ("elapsed_ns", [125_000_000, 250_000_000]),
            ("java", [True, False]),
            ("censored", [False, True]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify_samples(
                    self.sample.set_column(
                        self.sample.schema.get_field_index(field), field, [values]
                    )
                )

    def test_duplicate_ordinal_or_swapped_root(self):
        for field, values in (("root", [1, 0]), ("sample_index", [0, 0])):
            replacement = pa.array(values, type=pa.int32())
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify_samples(
                    self.sample.set_column(
                        self.sample.schema.get_field_index(field), field, replacement
                    )
                )

    def test_stored_values_are_type_and_bit_sensitive(self):
        for actual, original in [
            (True, 1),
            (1, 1.0),
            (0.0, -0.0),
            (1.0, 1.0 + 1e-15),
            ([None, 1], [1, None]),
        ]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                exact(actual, original, "stored field")

    def test_derived_values_have_predeclared_tolerance(self):
        derived({"gain": 1 + 1e-12}, {"gain": 1.0}, "claim")
        with self.assertRaises(ValueError):
            derived({"gain": 1 + 1e-8}, {"gain": 1.0}, "claim")
