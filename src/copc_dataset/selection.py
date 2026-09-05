"""Immutable value objects describing point-cloud subset selections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

RangeInput: TypeAlias = tuple[float | None, float | None]


def _validate_bound(value: float, name: str) -> None:
    """Validate that one concrete range bound is finite."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


@dataclass(frozen=True, slots=True)
class Bounds3D:
    """Closed, axis-aligned bounds in point-cloud coordinate space."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def __post_init__(self) -> None:
        """Validate finite and correctly ordered coordinate bounds."""
        pairs = (
            ("x", self.min_x, self.max_x),
            ("y", self.min_y, self.max_y),
            ("z", self.min_z, self.max_z),
        )
        for axis, minimum, maximum in pairs:
            _validate_bound(minimum, f"min_{axis}")
            _validate_bound(maximum, f"max_{axis}")
            if minimum > maximum:
                raise ValueError(
                    f"min_{axis} must not exceed max_{axis}: {minimum} > {maximum}"
                )

    @property
    def x(self) -> tuple[float, float]:
        """Return the closed X coordinate range."""
        return self.min_x, self.max_x

    @property
    def y(self) -> tuple[float, float]:
        """Return the closed Y coordinate range."""
        return self.min_y, self.max_y

    @property
    def z(self) -> tuple[float, float]:
        """Return the closed Z coordinate range."""
        return self.min_z, self.max_z

    def intersects(self, other: Bounds3D) -> bool:
        """Return whether these closed bounds intersect in all dimensions."""
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
            or self.max_z < other.min_z
            or self.min_z > other.max_z
        )

    def intersection(self, other: Bounds3D) -> Bounds3D | None:
        """Return the closed intersection or ``None`` when bounds are disjoint."""
        if not self.intersects(other):
            return None
        return Bounds3D(
            max(self.min_x, other.min_x),
            max(self.min_y, other.min_y),
            max(self.min_z, other.min_z),
            min(self.max_x, other.max_x),
            min(self.max_y, other.max_y),
            min(self.max_z, other.max_z),
        )

    def as_geojson_bbox(self) -> list[float]:
        """Return bounds in PDAL's six-value GeoJSON BBOX order."""
        return [
            self.min_x,
            self.min_y,
            self.min_z,
            self.max_x,
            self.max_y,
            self.max_z,
        ]


@dataclass(frozen=True, slots=True)
class DimensionRange:
    """An optional-sided, closed range on a non-coordinate dimension."""

    name: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        """Validate the dimension name and supplied range endpoints."""
        if not self.name:
            raise ValueError("Dimension range name must not be empty")
        if self.name.upper() in {"X", "Y", "Z"}:
            raise ValueError(
                f"Coordinate dimension {self.name!r} must be specified in bounds"
            )
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"Range for {self.name!r} must have at least one bound")
        if self.minimum is not None:
            _validate_bound(self.minimum, f"{self.name} minimum")
        if self.maximum is not None:
            _validate_bound(self.maximum, f"{self.name} maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"Minimum for {self.name!r} must not exceed its maximum")


@dataclass(frozen=True, slots=True)
class PointCloudSelection:
    """A named coherent 3D subset plus complementary dimension filters."""

    id: str
    bounds: Bounds3D
    filters: tuple[DimensionRange, ...] = ()
    columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate identity and uniqueness of filters and output columns."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Selection id must not be empty")
        filter_names = [item.name.casefold() for item in self.filters]
        if len(filter_names) != len(set(filter_names)):
            raise ValueError("A dimension may only be filtered once")
        if self.columns is not None:
            if not self.columns:
                raise ValueError("Selection columns must not be empty")
            column_names = [name.casefold() for name in self.columns]
            if len(column_names) != len(set(column_names)):
                raise ValueError("Selection columns must be unique")
