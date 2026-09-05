"""Metadata-only representation of a logical multi-file COPC dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa
from pyproj import CRS
from pyproj.exceptions import CRSError

from copc_dataset.catalog import (
    CopcSource,
    discover_copc_files,
    inspect_copc_sources,
    is_copc_href,
    load_stac_geoparquet,
    sources_to_arrow,
)
from copc_dataset.selection import (
    Bounds3D,
    DimensionRange,
    PointCloudSelection,
    RangeInput,
)


@dataclass(frozen=True, slots=True)
class CopcDataset:
    """A coherent COPC collection represented entirely by source metadata."""

    _sources: tuple[CopcSource, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    _bounds: Bounds3D = field(init=False, repr=False)
    _schema: pa.Schema = field(init=False, repr=False)
    _crs_wkt: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate source compatibility and derive aggregate dataset metadata."""
        if not self._sources:
            raise ValueError("A COPC dataset must contain at least one source")
        _validate_unique_sources(self._sources)
        schema = _validate_and_build_schema(self._sources)
        crs_wkt = _validate_and_get_crs(self._sources)
        bounds = Bounds3D(
            min(source.bounds.min_x for source in self._sources),
            min(source.bounds.min_y for source in self._sources),
            min(source.bounds.min_z for source in self._sources),
            max(source.bounds.max_x for source in self._sources),
            max(source.bounds.max_y for source in self._sources),
            max(source.bounds.max_z for source in self._sources),
        )
        object.__setattr__(self, "_bounds", bounds)
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_crs_wkt", crs_wkt)
        object.__setattr__(self, "attrs", dict(self.attrs))

    @classmethod
    def from_files(
        cls,
        source: str | Path | Sequence[str | Path],
        *,
        recursive: bool = False,
        install_extension: bool = True,
        attrs: Mapping[str, Any] | None = None,
    ) -> CopcDataset:
        """Open local COPCs by inspecting metadata without reading point data."""
        hrefs = discover_copc_files(source, recursive=recursive)
        sources = inspect_copc_sources(hrefs, install_extension=install_extension)
        return cls(sources, attrs or {})

    @classmethod
    def from_urls(
        cls,
        urls: Iterable[str],
        *,
        install_extension: bool = True,
        attrs: Mapping[str, Any] | None = None,
    ) -> CopcDataset:
        """Open remote COPCs by inspecting metadata without reading point data."""
        hrefs = tuple(str(url) for url in urls)
        invalid = [url for url in hrefs if not _is_url(url) or not is_copc_href(url)]
        if invalid:
            raise ValueError(f"Invalid COPC URLs: {invalid}")
        sources = inspect_copc_sources(hrefs, install_extension=install_extension)
        return cls(sources, attrs or {})

    @classmethod
    def from_geoparquet(
        cls,
        path: str | Path,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> CopcDataset:
        """Open a STAC GeoParquet catalog without contacting its COPC assets."""
        return cls(load_stac_geoparquet(path), attrs or {})

    @property
    def source_records(self) -> tuple[CopcSource, ...]:
        """Return immutable normalized source metadata records."""
        return self._sources

    @property
    def sources(self) -> pa.Table:
        """Return the source metadata manifest as an Arrow table."""
        return sources_to_arrow(self._sources)

    @property
    def source_count(self) -> int:
        """Return the number of COPC sources in the logical dataset."""
        return len(self._sources)

    @property
    def point_count(self) -> int:
        """Return the sum of physical point records reported by all sources."""
        return sum(source.point_count for source in self._sources)

    @property
    def bounds(self) -> Bounds3D:
        """Return aggregate closed 3D bounds covering every source."""
        return self._bounds

    @property
    def schema(self) -> pa.Schema:
        """Return the common Arrow schema of point dimensions."""
        return self._schema

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Return point dimension names in stable source order."""
        return tuple(self._schema.names)

    @property
    def crs(self) -> CRS:
        """Return the common dataset coordinate reference system."""
        return CRS.from_wkt(self._crs_wkt)

    @property
    def crs_wkt(self) -> str:
        """Return the common dataset CRS as WKT."""
        return self._crs_wkt

    def canonical_dimension(self, name: str) -> str:
        """Return the schema spelling of a case-insensitive dimension name."""
        matches = [
            candidate
            for candidate in self.dimensions
            if candidate.casefold() == name.casefold()
        ]
        if not matches:
            raise ValueError(
                f"Unknown point dimension {name!r}; available: {self.dimensions}"
            )
        return matches[0]

    def selection(
        self,
        id: str,
        *,
        x: RangeInput | None = None,
        y: RangeInput | None = None,
        z: RangeInput | None = None,
        filters: Mapping[str, RangeInput] | None = None,
        columns: Sequence[str] | None = None,
    ) -> PointCloudSelection:
        """Create a validated selection with concrete X, Y, and Z bounds."""
        x_range = _resolve_range(x, self.bounds.x, "x")
        y_range = _resolve_range(y, self.bounds.y, "y")
        z_range = _resolve_range(z, self.bounds.z, "z")

        dimension_filters = tuple(
            DimensionRange(
                self.canonical_dimension(name),
                *_validate_filter_range(range_value, name),
            )
            for name, range_value in (filters or {}).items()
        )
        canonical_columns = (
            tuple(self.canonical_dimension(name) for name in columns)
            if columns is not None
            else None
        )
        return PointCloudSelection(
            id=id,
            bounds=Bounds3D(
                x_range[0],
                y_range[0],
                z_range[0],
                x_range[1],
                y_range[1],
                z_range[1],
            ),
            filters=dimension_filters,
            columns=canonical_columns,
        )

    def candidate_sources(
        self, selection: PointCloudSelection
    ) -> tuple[CopcSource, ...]:
        """Return sources whose metadata bounds intersect a selection."""
        return tuple(
            source
            for source in self._sources
            if source.bounds.intersects(selection.bounds)
        )

    def __repr__(self) -> str:
        """Return an xarray-inspired metadata summary without loading points."""
        bounds = self.bounds
        dimensions = ", ".join(f"{field.name}: {field.type}" for field in self.schema)
        return (
            "<CopcDataset>\n"
            f"  Sources: {self.source_count:,}\n"
            f"  Points: {self.point_count:,}\n"
            f"  X: [{bounds.min_x}, {bounds.max_x}]\n"
            f"  Y: [{bounds.min_y}, {bounds.max_y}]\n"
            f"  Z: [{bounds.min_z}, {bounds.max_z}]\n"
            f"  Dimensions: {dimensions}\n"
            f"  CRS: {self.crs.name}"
        )


def open_copc_dataset(
    source: str | Path | Iterable[str | Path],
    *,
    recursive: bool = False,
    install_extension: bool = True,
    attrs: Mapping[str, Any] | None = None,
) -> CopcDataset:
    """Open COPCs or a STAC GeoParquet catalog as one metadata-only dataset."""
    if isinstance(source, (str, Path)):
        raw_source = str(source)
        if raw_source.casefold().endswith((".parquet", ".geoparquet")):
            return CopcDataset.from_geoparquet(source, attrs=attrs)
        if _is_url(raw_source):
            return CopcDataset.from_urls(
                [raw_source],
                install_extension=install_extension,
                attrs=attrs,
            )
        return CopcDataset.from_files(
            source,
            recursive=recursive,
            install_extension=install_extension,
            attrs=attrs,
        )

    values = tuple(source)
    if not values:
        raise ValueError("At least one COPC source is required")
    if all(_is_url(str(value)) for value in values):
        return CopcDataset.from_urls(
            (str(value) for value in values),
            install_extension=install_extension,
            attrs=attrs,
        )
    if any(_is_url(str(value)) for value in values):
        raise ValueError("Local COPC paths and remote URLs cannot be mixed")
    return CopcDataset.from_files(
        values,
        recursive=recursive,
        install_extension=install_extension,
        attrs=attrs,
    )


def _is_url(value: str) -> bool:
    """Return whether a string contains a non-file URL scheme."""
    scheme = urlsplit(value).scheme.casefold()
    return bool(scheme and scheme != "file")


def _resolve_range(
    requested: RangeInput | None,
    default: tuple[float, float],
    axis: str,
) -> tuple[float, float]:
    """Resolve an optional-sided coordinate range against dataset bounds."""
    if requested is None:
        return default
    if len(requested) != 2:
        raise ValueError(f"{axis} range must contain exactly two values")
    minimum = default[0] if requested[0] is None else requested[0]
    maximum = default[1] if requested[1] is None else requested[1]
    return float(minimum), float(maximum)


def _validate_filter_range(requested: RangeInput, dimension: str) -> RangeInput:
    """Require exactly two endpoints for a complementary dimension range."""
    if not isinstance(requested, (tuple, list)) or len(requested) != 2:
        raise ValueError(f"Range for {dimension!r} must contain exactly two values")
    return requested[0], requested[1]


def _validate_unique_sources(sources: Sequence[CopcSource]) -> None:
    """Reject duplicate source IDs or hrefs that make provenance ambiguous."""
    ids = [source.id for source in sources]
    hrefs = [source.href for source in sources]
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    duplicate_hrefs = sorted({value for value in hrefs if hrefs.count(value) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate COPC source ids: {duplicate_ids}")
    if duplicate_hrefs:
        raise ValueError(f"Duplicate COPC source hrefs: {duplicate_hrefs}")


def _validate_and_build_schema(sources: Sequence[CopcSource]) -> pa.Schema:
    """Require compatible source dimensions and return their Arrow schema."""
    first = sources[0]
    reference = {
        dimension.name.casefold(): dimension.data_type for dimension in first.dimensions
    }
    missing_coordinates = {"x", "y", "z"}.difference(reference)
    if missing_coordinates:
        raise ValueError(
            f"COPC source {first.id!r} lacks coordinate dimensions: "
            f"{sorted(missing_coordinates)}"
        )

    for source in sources[1:]:
        candidate = {
            dimension.name.casefold(): dimension.data_type
            for dimension in source.dimensions
        }
        if candidate != reference:
            missing = sorted(set(reference).difference(candidate))
            extra = sorted(set(candidate).difference(reference))
            incompatible = sorted(
                name
                for name in set(reference).intersection(candidate)
                if reference[name] != candidate[name]
            )
            raise ValueError(
                f"COPC source {source.id!r} has an incompatible schema; "
                f"missing={missing}, extra={extra}, incompatible={incompatible}"
            )
    return pa.schema(
        [
            pa.field(dimension.name, dimension.data_type, nullable=False)
            for dimension in first.dimensions
        ]
    )


def _validate_and_get_crs(sources: Sequence[CopcSource]) -> str:
    """Require a semantically common CRS and return the first source WKT."""
    try:
        reference = CRS.from_user_input(sources[0].crs_wkt)
    except CRSError as error:
        raise ValueError(f"COPC source {sources[0].id!r} has an invalid CRS") from error
    for source in sources[1:]:
        try:
            candidate = CRS.from_user_input(source.crs_wkt)
        except CRSError as error:
            raise ValueError(f"COPC source {source.id!r} has an invalid CRS") from error
        if not reference.equals(candidate):
            raise ValueError(
                f"COPC source {source.id!r} CRS differs from {sources[0].id!r}"
            )
    return reference.to_wkt()
