"""Tests for DuckDB/PDAL selection compilation and Arrow extraction."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa

from copc_dataset import (
    Bounds3D,
    CopcBatchReader,
    CopcDataset,
    DuckDBCopcReader,
    PointCloudSelection,
    open_copc_dataset,
)
from copc_dataset.duckdb_reader import (
    _canonicalize_selection,
    _compile_expression,
    _connect_pdal,
    _selection_schema,
)
from tests.helpers import dataset, dimensions, source


class ReaderCompilationTests(unittest.TestCase):
    """Verify deterministic translation from selections to reader contracts."""

    def test_compiles_coordinate_and_complementary_ranges(self) -> None:
        """Compile all mandatory coordinates and optional filter endpoints."""
        value = dataset()
        selection = value.selection(
            "part",
            x=(1, 2),
            y=(3, 4),
            z=(5, 6),
            filters={"Classification": (2, None)},
        )

        expression = _compile_expression(selection.bounds, selection.filters)

        self.assertIn("(X >= 1)", expression)
        self.assertIn("(Z <= 6)", expression)
        self.assertIn("(Classification >= 2)", expression)
        self.assertNotIn("Classification <=", expression)

    def test_builds_projected_schema_with_provenance(self) -> None:
        """Add optional source provenance after requested point dimensions."""
        value = dataset()
        selection = value.selection("part", columns=("X", "Classification"))

        schema = _selection_schema(value, selection, include_source=True)

        self.assertEqual(
            schema.names,
            ["X", "Classification", "__source_id", "__source_href"],
        )
        self.assertEqual(schema.field("Classification").type, pa.uint8())

    def test_validates_reader_options(self) -> None:
        """Reject invalid network, resolution, and batch controls."""
        with self.assertRaises(ValueError):
            DuckDBCopcReader(requests=0)
        with self.assertRaises(ValueError):
            DuckDBCopcReader(resolution=0)
        with self.assertRaises(ValueError):
            DuckDBCopcReader(batch_size=0)
        with self.assertRaises(ValueError):
            DuckDBCopcReader(requests=1.5)

    def test_preserves_large_integer_filter_literals(self) -> None:
        """Avoid changing 64-bit integer endpoints through float formatting."""
        value = dataset()
        selection = value.selection(
            "large",
            filters={"Classification": (9_007_199_254_740_993, None)},
        )

        expression = _compile_expression(selection.bounds, selection.filters)

        self.assertIn("9007199254740993", expression)

    def test_canonicalizes_direct_selection_names(self) -> None:
        """Apply the dataset's dimension spelling to direct selection objects."""
        value = dataset()
        direct = PointCloudSelection(
            "direct",
            Bounds3D(0, 0, -10, 1, 1, 1),
            columns=("x", "classification"),
        )

        canonical = _canonicalize_selection(value, direct)

        self.assertEqual(canonical.columns, ("X", "Classification"))

    def test_closes_underlying_generator_after_partial_read(self) -> None:
        """Release generator resources when a batch stream is closed early."""
        closed = []

        def batches():
            """Yield two batches and record deterministic generator cleanup."""
            try:
                yield pa.record_batch([pa.array([1.0])], names=["X"])
                yield pa.record_batch([pa.array([2.0])], names=["X"])
            finally:
                closed.append(True)

        reader = CopcBatchReader(pa.schema([("X", pa.float64())]), batches())
        next(reader)
        reader.close()

        self.assertTrue(reader.closed)
        self.assertEqual(closed, [True])

    def test_reads_reserved_point_dimension_without_provenance(self) -> None:
        """Preserve point fields whose names match disabled provenance fields."""
        point_dimensions = dimensions(("__source_id", pa.uint16()))
        value = CopcDataset(
            (
                source(
                    "reserved",
                    Bounds3D(0, 0, 0, 1, 1, 1),
                    point_dimensions=point_dimensions,
                ),
            )
        )
        batch = pa.record_batch(
            [
                pa.array([0.5], type=pa.float64()),
                pa.array([0.5], type=pa.float64()),
                pa.array([0.5], type=pa.float64()),
                pa.array([7], type=pa.uint16()),
            ],
            schema=value.schema,
        )
        arrow_reader = pa.RecordBatchReader.from_batches(value.schema, [batch])
        connection = Mock()
        connection.execute.return_value.to_arrow_reader.return_value = arrow_reader

        with patch("copc_dataset.duckdb_reader._connect_pdal", return_value=connection):
            table = DuckDBCopcReader(
                include_source=False, install_extension=False
            ).read(value, value.selection("reserved"))

        self.assertEqual(table.schema.names, ["X", "Y", "Z", "__source_id"])
        self.assertEqual(table["__source_id"].to_pylist(), [7])
        connection.close.assert_called_once_with()

    def test_closes_connection_when_pdal_setup_fails(self) -> None:
        """Release reader connections when extension loading raises an error."""
        connection = Mock()
        connection.execute.side_effect = RuntimeError("load failed")

        with (
            patch("copc_dataset.duckdb_reader.duckdb.connect", return_value=connection),
            self.assertRaisesRegex(RuntimeError, "load failed"),
        ):
            _connect_pdal(install_extension=False)

        connection.close.assert_called_once_with()


@unittest.skipUnless(
    os.environ.get("COPC_TEST_FILE"),
    "Set COPC_TEST_FILE to run local DuckDB/PDAL integration tests",
)
class LocalCopcIntegrationTests(unittest.TestCase):
    """Exercise a bounded read against an explicitly supplied local COPC."""

    def test_reads_center_subset_to_arrow(self) -> None:
        """Return bounded XYZ points with a stable projected Arrow schema."""
        path = Path(os.environ["COPC_TEST_FILE"])
        value = open_copc_dataset(path, install_extension=False)
        center_x = (value.bounds.min_x + value.bounds.max_x) / 2
        center_y = (value.bounds.min_y + value.bounds.max_y) / 2
        selection = value.selection(
            "center",
            x=(center_x - 5, center_x + 5),
            y=(center_y - 5, center_y + 5),
            filters={"Classification": (2, 2)},
            columns=("X", "Y", "Z", "Classification"),
        )

        table = DuckDBCopcReader(
            requests=1,
            include_source=True,
            install_extension=False,
        ).read(value, selection)

        self.assertEqual(
            table.schema.names,
            [
                "X",
                "Y",
                "Z",
                "Classification",
                "__source_id",
                "__source_href",
            ],
        )
        self.assertGreater(table.num_rows, 0)
        self.assertGreaterEqual(min(table["X"].to_pylist()), center_x - 5)
        self.assertLessEqual(max(table["X"].to_pylist()), center_x + 5)
        self.assertGreaterEqual(min(table["Y"].to_pylist()), center_y - 5)
        self.assertLessEqual(max(table["Y"].to_pylist()), center_y + 5)
        self.assertGreaterEqual(min(table["Z"].to_pylist()), value.bounds.min_z)
        self.assertLessEqual(max(table["Z"].to_pylist()), value.bounds.max_z)
        self.assertEqual(set(table["Classification"].to_pylist()), {2})
        self.assertEqual(len(set(table["__source_id"].to_pylist())), 1)


if __name__ == "__main__":
    unittest.main()
