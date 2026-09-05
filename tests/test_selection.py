"""Tests for coherent 3D selection value objects."""

from __future__ import annotations

import unittest

from copc_dataset import Bounds3D, DimensionRange, PointCloudSelection
from tests.helpers import dataset


class Bounds3DTests(unittest.TestCase):
    """Verify closed 3D bound validation and intersection semantics."""

    def test_intersection_includes_shared_boundary(self) -> None:
        """Treat a shared coordinate boundary as an intersection."""
        west = Bounds3D(0, 0, 0, 10, 10, 10)
        east = Bounds3D(10, 2, 2, 20, 8, 8)

        self.assertTrue(west.intersects(east))
        self.assertEqual(
            west.intersection(east),
            Bounds3D(10, 2, 2, 10, 8, 8),
        )

    def test_rejects_reversed_bounds(self) -> None:
        """Reject a minimum coordinate greater than its maximum."""
        with self.assertRaisesRegex(ValueError, "min_x"):
            Bounds3D(2, 0, 0, 1, 1, 1)


class SelectionTests(unittest.TestCase):
    """Verify dataset-aware construction of point-cloud selections."""

    def test_defaults_coordinates_to_dataset_extent(self) -> None:
        """Resolve all omitted coordinate ranges to aggregate bounds."""
        selection = dataset().selection("all")

        self.assertEqual(selection.bounds, Bounds3D(0, 0, -10, 20, 10, 30))

    def test_resolves_one_sided_ranges_and_dimension_case(self) -> None:
        """Resolve open bounds and canonicalize schema dimension names."""
        selection = dataset().selection(
            "filtered",
            x=(5, None),
            y=None,
            z=(None, 15),
            filters={"classification": (2, 5)},
            columns=("x", "Y", "classification"),
        )

        self.assertEqual(selection.bounds, Bounds3D(5, 0, -10, 20, 10, 15))
        self.assertEqual(selection.filters, (DimensionRange("Classification", 2, 5),))
        self.assertEqual(selection.columns, ("X", "Y", "Classification"))

    def test_rejects_coordinate_complementary_filter(self) -> None:
        """Keep coordinate constraints exclusively in coherent 3D bounds."""
        with self.assertRaisesRegex(ValueError, "Coordinate dimension"):
            dataset().selection("bad", filters={"X": (1, 2)})

    def test_rejects_malformed_complementary_range(self) -> None:
        """Report malformed complementary ranges as public input errors."""
        with self.assertRaisesRegex(ValueError, "exactly two"):
            dataset().selection("bad", filters={"Classification": (1, 2, 3)})

    def test_direct_selection_rejects_duplicate_filters(self) -> None:
        """Reject duplicate filter dimensions regardless of case."""
        with self.assertRaisesRegex(ValueError, "only be filtered once"):
            PointCloudSelection(
                "bad",
                Bounds3D(0, 0, 0, 1, 1, 1),
                filters=(
                    DimensionRange("Classification", 1, 2),
                    DimensionRange("classification", 3, 4),
                ),
            )


if __name__ == "__main__":
    unittest.main()
