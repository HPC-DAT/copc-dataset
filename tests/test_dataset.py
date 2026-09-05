"""Tests for metadata-only logical COPC datasets."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS

from copc_dataset import Bounds3D, CopcDataset, open_copc_dataset
from copc_dataset.catalog import _connect_pdal, discover_copc_files
from tests.helpers import dataset, dimensions, source


def stac_row(**overrides: object) -> dict[str, object]:
    """Return one minimal valid STAC point-cloud catalog row."""
    row: dict[str, object] = {
        "id": "tile",
        "assets": {"data": {"href": "assets/tile.copc.laz"}},
        "pc:count": 42,
        "pc:schemas": [
            {"name": name, "size": 8, "type": "floating"} for name in ("X", "Y", "Z")
        ],
        "proj:bbox": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "proj:wkt2": CRS.from_epsg(28992).to_wkt(),
    }
    row.update(overrides)
    return row


class DatasetTests(unittest.TestCase):
    """Verify aggregate metadata and strict collection compatibility."""

    def test_derives_aggregate_metadata(self) -> None:
        """Aggregate counts, bounds, schema, CRS, and source manifest."""
        value = dataset()

        self.assertEqual(value.source_count, 2)
        self.assertEqual(value.point_count, 200)
        self.assertEqual(value.bounds, Bounds3D(0, 0, -10, 20, 10, 30))
        self.assertEqual(value.dimensions, ("X", "Y", "Z", "Classification"))
        self.assertEqual(value.crs.to_epsg(), 28992)
        self.assertEqual(value.sources.num_rows, 2)
        self.assertEqual(value.attrs["purpose"], "test")

    def test_is_pickle_serializable(self) -> None:
        """Allow the metadata-only object to be transported to Dask workers."""
        restored = pickle.loads(pickle.dumps(dataset()))

        self.assertEqual(restored.point_count, 200)
        self.assertEqual(restored.dimensions, dataset().dimensions)

    def test_rejects_incompatible_schema(self) -> None:
        """Reject a source missing a dimension from the common schema."""
        first = source("one", Bounds3D(0, 0, 0, 1, 1, 1))
        second = source(
            "two",
            Bounds3D(1, 0, 0, 2, 1, 1),
            point_dimensions=dimensions(),
        )

        with self.assertRaisesRegex(ValueError, "incompatible schema"):
            CopcDataset((first, second))

    def test_rejects_different_crs(self) -> None:
        """Reject source metadata expressed in a different CRS."""
        first = source("one", Bounds3D(0, 0, 0, 1, 1, 1))
        second = source(
            "two",
            Bounds3D(1, 0, 0, 2, 1, 1),
            crs=CRS.from_epsg(4326).to_wkt(),
        )

        with self.assertRaisesRegex(ValueError, "CRS differs"):
            CopcDataset((first, second))

    def test_prunes_sources_in_three_dimensions(self) -> None:
        """Use all mandatory coordinate ranges during metadata pruning."""
        value = dataset()
        selection = value.selection(
            "east-low",
            x=(12, 14),
            y=(1, 2),
            z=(-9, -6),
        )

        self.assertEqual(value.candidate_sources(selection), ())

    def test_repr_contains_summary_without_points(self) -> None:
        """Display the principal xarray-like metadata fields."""
        text = repr(dataset())

        self.assertIn("<CopcDataset>", text)
        self.assertIn("Sources: 2", text)
        self.assertIn("Points: 200", text)
        self.assertIn("Classification: uint8", text)

    def test_normalizes_authority_crs_to_wkt(self) -> None:
        """Keep the aggregate CRS property valid for accepted CRS inputs."""
        value = CopcDataset(
            (source("authority", Bounds3D(0, 0, 0, 1, 1, 1), crs="EPSG:28992"),)
        )

        self.assertEqual(value.crs.to_epsg(), 28992)
        self.assertEqual(CRS.from_wkt(value.crs_wkt).to_epsg(), 28992)

    def test_source_manifest_uses_stable_floating_bounds(self) -> None:
        """Expose coordinate bounds with stable types regardless of input values."""
        manifest = dataset().sources

        for name in ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z"):
            self.assertEqual(manifest.schema.field(name).type, pa.float64())


class CatalogTests(unittest.TestCase):
    """Verify local discovery and STAC GeoParquet normalization."""

    def test_discovers_case_insensitive_copc_suffixes(self) -> None:
        """Discover lowercase and uppercase COPC names while ignoring LAS files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.copc.laz").touch()
            (root / "two.COPC.LAZ").touch()
            (root / "ignored.laz").touch()

            discovered = discover_copc_files(root)

        self.assertEqual(len(discovered), 2)
        self.assertTrue(discovered[0].endswith("one.copc.laz"))
        self.assertTrue(discovered[1].endswith("two.COPC.LAZ"))

    def test_opens_stac_geoparquet_without_asset_access(self) -> None:
        """Build dataset metadata solely from a compatible STAC catalog row."""
        crs_wkt = CRS.from_epsg(28992).to_wkt()
        row = {
            "id": "tile",
            "assets": {"data": {"href": "https://invalid.test/tile.copc.laz"}},
            "pc:count": 42,
            "pc:schemas": [
                {"name": "X", "size": 8, "type": "floating"},
                {"name": "Y", "size": 8, "type": "floating"},
                {"name": "Z", "size": 8, "type": "floating"},
            ],
            "bbox": {
                "xmin": 4.0,
                "ymin": 52.0,
                "zmin": -100.0,
                "xmax": 5.0,
                "ymax": 53.0,
                "zmax": 100.0,
            },
            "proj:bbox": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "proj:wkt2": crs_wkt,
            "pc:type": "lidar",
        }
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.parquet"
            pq.write_table(pa.Table.from_pylist([row]), catalog)

            value = open_copc_dataset(catalog)

        self.assertEqual(value.point_count, 42)
        self.assertEqual(value.dimensions, ("X", "Y", "Z"))
        self.assertEqual(value.bounds, Bounds3D(1, 2, 3, 4, 5, 6))
        self.assertEqual(value.source_records[0].properties["pc:type"], "lidar")

    def test_resolves_relative_stac_asset_against_catalog(self) -> None:
        """Make relative COPC assets independent of worker working directories."""
        crs_wkt = CRS.from_epsg(28992).to_wkt()
        row = {
            "id": "relative",
            "assets": {"data": {"href": "assets/tile.copc.laz"}},
            "pc:count": 1,
            "pc:schemas": [
                {"name": name, "size": 8, "type": "floating"}
                for name in ("X", "Y", "Z")
            ],
            "bbox": [4.0, 52.0, 0.0, 5.0, 53.0, 1.0],
            "proj:bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "proj:wkt2": crs_wkt,
        }
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.parquet"
            pq.write_table(pa.Table.from_pylist([row]), catalog)

            value = open_copc_dataset(catalog)

            self.assertEqual(
                value.source_records[0].href,
                str((Path(directory) / "assets/tile.copc.laz").resolve()),
            )

    def test_resolves_asset_against_relative_stac_self_link(self) -> None:
        """Resolve an asset relative to its item's catalog-relative self link."""
        row = stac_row(
            links=[{"rel": "self", "href": "items/tile.json"}],
            assets={"data": {"href": "data/tile.copc.laz"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.parquet"
            pq.write_table(pa.Table.from_pylist([row]), catalog)

            value = open_copc_dataset(catalog)

            self.assertEqual(
                value.source_records[0].href,
                str((Path(directory) / "items/data/tile.copc.laz").resolve()),
            )

    def test_rejects_invalid_stac_identity_and_count(self) -> None:
        """Reject null identities and fractional physical point counts."""
        invalid_rows = (
            (stac_row(id=None), "id must be a non-empty string"),
            (stac_row(**{"pc:count": 1.5}), "invalid pc:count"),
        )
        for row, message in invalid_rows:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                catalog = Path(directory) / "catalog.parquet"
                pq.write_table(pa.Table.from_pylist([row]), catalog)

                with self.assertRaisesRegex(ValueError, message):
                    open_copc_dataset(catalog)

    def test_closes_connection_when_pdal_setup_fails(self) -> None:
        """Release catalog connections when extension loading raises an error."""
        connection = Mock()
        connection.execute.side_effect = RuntimeError("load failed")

        with (
            patch("copc_dataset.catalog.duckdb.connect", return_value=connection),
            self.assertRaisesRegex(RuntimeError, "load failed"),
        ):
            _connect_pdal(install_extension=False)

        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
