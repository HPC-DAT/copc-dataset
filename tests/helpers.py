"""Shared metadata fixtures for COPC dataset unit tests."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from pyproj import CRS

from copc_dataset import Bounds3D, CopcDataset, CopcSource, PointDimension


def dimensions(
    *extra: tuple[str, pa.DataType],
) -> tuple[PointDimension, ...]:
    """Return a minimal XYZ schema with optional additional dimensions."""
    return (
        PointDimension("X", pa.float64()),
        PointDimension("Y", pa.float64()),
        PointDimension("Z", pa.float64()),
        *(PointDimension(name, data_type) for name, data_type in extra),
    )


def source(
    source_id: str,
    bounds: Bounds3D,
    *,
    href: str | None = None,
    point_count: int = 100,
    point_dimensions: tuple[PointDimension, ...] | None = None,
    crs: str | None = None,
    properties: dict[str, Any] | None = None,
) -> CopcSource:
    """Create one valid synthetic COPC metadata record."""
    return CopcSource(
        id=source_id,
        href=href or f"https://example.test/{source_id}.copc.laz",
        bounds=bounds,
        point_count=point_count,
        dimensions=point_dimensions or dimensions(("Classification", pa.uint8())),
        crs_wkt=crs or CRS.from_epsg(28992).to_wkt(),
        properties=properties or {},
    )


def dataset() -> CopcDataset:
    """Create a two-source logical dataset with adjacent X extents."""
    return CopcDataset(
        (
            source("west", Bounds3D(0, 0, -10, 10, 10, 20)),
            source("east", Bounds3D(10, 0, -5, 20, 10, 30)),
        ),
        attrs={"purpose": "test"},
    )
