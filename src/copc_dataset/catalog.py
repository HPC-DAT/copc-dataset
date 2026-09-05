"""COPC source discovery and metadata catalog adapters."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from copc_dataset.selection import Bounds3D

_STAC_STRUCTURAL_COLUMNS = {
    "assets",
    "bbox",
    "geometry",
    "id",
    "links",
    "proj:bbox",
    "proj:geometry",
    "proj:wkt2",
}


@dataclass(frozen=True, slots=True)
class PointDimension:
    """The name and Arrow type of one point-cloud dimension."""

    name: str
    data_type: pa.DataType

    def __post_init__(self) -> None:
        """Validate that the dimension has a non-empty name and numeric type."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Point dimension name must not be empty")
        if not (
            pa.types.is_integer(self.data_type) or pa.types.is_floating(self.data_type)
        ):
            raise TypeError(
                f"Point dimension {self.name!r} must be numeric, got {self.data_type}"
            )


@dataclass(frozen=True, slots=True)
class CopcSource:
    """Metadata for one COPC source without any loaded point records."""

    id: str
    href: str
    bounds: Bounds3D
    point_count: int
    dimensions: tuple[PointDimension, ...]
    crs_wkt: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required source metadata."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("COPC source id must not be empty")
        if not isinstance(self.href, str) or not self.href:
            raise ValueError("COPC source href must not be empty")
        if (
            isinstance(self.point_count, bool)
            or not isinstance(self.point_count, int)
            or self.point_count < 0
        ):
            raise ValueError("COPC source point count must not be negative")
        if not self.dimensions:
            raise ValueError("COPC source must define point dimensions")
        names = [dimension.name.casefold() for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError(f"COPC source {self.id!r} has duplicate dimensions")
        if not self.crs_wkt:
            raise ValueError(f"COPC source {self.id!r} has no CRS")


def is_copc_href(value: str | Path) -> bool:
    """Return whether a path or URL has a case-insensitive COPC suffix."""
    path = urlsplit(str(value)).path
    return path.casefold().endswith(".copc.laz")


def force_copc_reader_url(value: str) -> str:
    """Force PDAL to infer ``readers.copc`` for an uppercase remote URL."""
    if value.endswith(".copc.laz"):
        return value
    parsed = urlsplit(value)
    fragment = f"{parsed.fragment}/.copc.laz" if parsed.fragment else "/.copc.laz"
    return urlunsplit(parsed._replace(fragment=fragment))


@contextmanager
def pdal_copc_href(value: str) -> Iterator[str]:
    """Yield a PDAL-compatible href while preserving the original source name."""
    if value.endswith(".copc.laz"):
        yield value
        return
    if urlsplit(value).scheme and not Path(value).exists():
        yield force_copc_reader_url(value)
        return

    source = Path(value).resolve()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        temporary = tempfile.TemporaryDirectory(
            prefix=".copc-dataset-", dir=source.parent
        )
        link = Path(temporary.name) / "source.copc.laz"
        os.link(source, link)
    except OSError as error:
        if temporary is not None:
            temporary.cleanup()
        raise RuntimeError(
            "PDAL requires local COPC filenames to end in lowercase "
            "'.copc.laz'. A temporary hard link could not be created on the "
            "source filesystem; use a lowercase filename or a writable source "
            "directory that supports hard links."
        ) from error
    try:
        yield str(link)
    finally:
        temporary.cleanup()


def source_id_from_href(href: str) -> str:
    """Derive a stable source identifier from a COPC path or URL."""
    name = Path(unquote(urlsplit(href).path)).name
    return name[: -len(".copc.laz")] if name.casefold().endswith(".copc.laz") else name


def discover_copc_files(
    source: str | Path | Sequence[str | Path], *, recursive: bool = False
) -> list[str]:
    """Discover local COPC files from paths, a directory, or a glob pattern."""
    if isinstance(source, (str, Path)):
        values: Sequence[str | Path] = [source]
    else:
        values = source

    discovered: list[Path] = []
    for value in values:
        raw_value = str(value)
        path = Path(value).expanduser()
        if any(character in raw_value for character in "*?["):
            discovered.extend(
                Path(item) for item in glob.glob(raw_value, recursive=recursive)
            )
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            discovered.extend(item for item in iterator if item.is_file())
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(f"COPC path does not exist: {value}")

    files = sorted({str(path.resolve()) for path in discovered if is_copc_href(path)})
    if not files:
        raise FileNotFoundError("No .copc.laz files were found")
    return files


def _connect_pdal(*, install_extension: bool) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with the PDAL extension loaded."""
    connection = duckdb.connect(database=":memory:")
    try:
        if install_extension:
            connection.execute("INSTALL pdal FROM community")
        connection.execute("LOAD pdal")
    except BaseException:
        connection.close()
        raise
    return connection


def inspect_copc_sources(
    hrefs: Iterable[str], *, install_extension: bool = True
) -> tuple[CopcSource, ...]:
    """Read only COPC metadata for a collection of local or remote sources."""
    href_list = [str(href) for href in hrefs]
    if not href_list:
        raise ValueError("At least one COPC source is required")
    duplicate_hrefs = sorted(
        href for href, count in Counter(href_list).items() if count > 1
    )
    if duplicate_hrefs:
        raise ValueError(f"Duplicate COPC source hrefs: {duplicate_hrefs}")
    invalid = [href for href in href_list if not is_copc_href(href)]
    if invalid:
        raise ValueError(f"Sources are not COPC files: {invalid}")

    connection = _connect_pdal(install_extension=install_extension)
    try:
        sources = [_inspect_copc_source(connection, href) for href in href_list]
    finally:
        connection.close()
    return _disambiguate_generated_ids(sources)


def _disambiguate_generated_ids(
    sources: Sequence[CopcSource],
) -> tuple[CopcSource, ...]:
    """Add a stable location hash when generated source basenames collide."""
    counts = Counter(source.id for source in sources)
    return tuple(
        replace(
            source,
            id=(
                f"{source.id}-{hashlib.sha256(source.href.encode()).hexdigest()[:8]}"
                if counts[source.id] > 1
                else source.id
            ),
        )
        for source in sources
    )


def _inspect_copc_source(
    connection: duckdb.DuckDBPyConnection, href: str
) -> CopcSource:
    """Inspect one COPC source and fail if PDAL cannot return its metadata."""
    query = """
        SELECT
            point_count,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            srs_wkt,
            dimensions
        FROM PDAL_Info(?)
    """
    with pdal_copc_href(href) as reader_href:
        row = connection.execute(query, [reader_href]).fetchone()
    if row is None:
        raise RuntimeError(f"PDAL could not inspect COPC source: {href}")

    point_count, min_x, min_y, min_z, max_x, max_y, max_z, crs_wkt, dimensions = row
    return CopcSource(
        id=source_id_from_href(href),
        href=href,
        bounds=Bounds3D(min_x, min_y, min_z, max_x, max_y, max_z),
        point_count=int(point_count),
        dimensions=tuple(
            PointDimension(item["name"], _pdal_type_to_arrow(item["type"]))
            for item in dimensions
        ),
        crs_wkt=crs_wkt,
        properties={},
    )


def load_stac_geoparquet(path: str | Path) -> tuple[CopcSource, ...]:
    """Load COPC source metadata from a STAC GeoParquet catalog."""
    catalog_path = Path(path).expanduser().resolve()
    table = pq.read_table(catalog_path)
    required = {"id", "assets", "pc:count", "pc:schemas", "proj:bbox", "proj:wkt2"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(
            f"STAC GeoParquet catalog is missing columns: {sorted(missing)}"
        )

    sources = tuple(
        _source_from_stac_row(row, catalog_path) for row in table.to_pylist()
    )
    if not sources:
        raise ValueError(f"STAC GeoParquet catalog contains no sources: {path}")
    return sources


def _source_from_stac_row(row: dict[str, Any], catalog_path: Path) -> CopcSource:
    """Normalize one STAC GeoParquet row into a ``CopcSource``."""
    source_id = row["id"]
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"Catalog source id must be a non-empty string: {source_id!r}")
    point_count = row["pc:count"]
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 0
    ):
        raise ValueError(
            f"Catalog source {source_id!r} has an invalid pc:count: {point_count!r}"
        )
    schemas = row["pc:schemas"]
    if not isinstance(schemas, list) or not schemas:
        raise ValueError(
            f"Catalog source {source_id!r} must define a non-empty pc:schemas list"
        )
    if not all(isinstance(item, dict) for item in schemas):
        raise ValueError(f"Catalog source {source_id!r} has malformed pc:schemas")

    href = _resolve_asset_href(
        _extract_asset_href(row["assets"]),
        row.get("links"),
        catalog_path,
    )
    crs_wkt = row.get("proj:wkt2")
    if not isinstance(crs_wkt, str) or not crs_wkt:
        raise ValueError(f"Catalog source {source_id!r} has no proj:wkt2 CRS")
    properties = {
        key: value for key, value in row.items() if key not in _STAC_STRUCTURAL_COLUMNS
    }
    return CopcSource(
        id=source_id,
        href=href,
        bounds=_extract_bounds(row["proj:bbox"]),
        point_count=point_count,
        dimensions=tuple(
            PointDimension(
                item["name"],
                _stac_type_to_arrow(item["type"], item["size"]),
            )
            for item in schemas
        ),
        crs_wkt=crs_wkt,
        properties=properties,
    )


def _extract_asset_href(assets: Any) -> str:
    """Extract one unambiguous COPC data asset from a STAC assets structure."""
    if not isinstance(assets, dict):
        raise TypeError("STAC assets must be a mapping")
    candidates = []
    for key, asset in assets.items():
        if not isinstance(asset, dict) or not asset.get("href"):
            continue
        href = str(asset["href"])
        if is_copc_href(href):
            roles = asset.get("roles") or []
            candidates.append((key, href, "data" in roles or key == "data"))
    preferred = [href for _, href, is_data in candidates if is_data]
    if len(preferred) == 1:
        return preferred[0]
    if not preferred and len(candidates) == 1:
        return candidates[0][1]
    if not candidates:
        raise ValueError("STAC source has no COPC asset")
    raise ValueError("STAC source has multiple ambiguous COPC assets")


def _resolve_asset_href(
    href: str,
    links: Any,
    catalog_path: Path,
) -> str:
    """Resolve a relative STAC asset against its item or catalog location."""
    if urlsplit(href).scheme or Path(href).is_absolute():
        return href
    if isinstance(links, list):
        self_links = [
            link.get("href")
            for link in links
            if isinstance(link, dict) and link.get("rel") == "self" and link.get("href")
        ]
        if len(self_links) == 1:
            self_href = str(self_links[0])
            if urlsplit(self_href).scheme:
                return urljoin(self_href, href)
            self_path = Path(self_href).expanduser()
            if not self_path.is_absolute():
                self_path = catalog_path.parent / self_path
            return str((self_path.resolve().parent / href).resolve())
    return str((catalog_path.parent / href).resolve())


def _extract_bounds(value: Any) -> Bounds3D:
    """Parse a six-value list or named struct into ``Bounds3D``."""
    if isinstance(value, dict):
        return Bounds3D(
            value["xmin"],
            value["ymin"],
            value["zmin"],
            value["xmax"],
            value["ymax"],
            value["zmax"],
        )
    if isinstance(value, (list, tuple)) and len(value) == 6:
        return Bounds3D(value[0], value[1], value[2], value[3], value[4], value[5])
    raise ValueError(f"Unsupported 3D bounding box: {value!r}")


def _stac_type_to_arrow(kind: Any, size: Any) -> pa.DataType:
    """Convert a STAC point-cloud schema type and byte width to Arrow."""
    if not isinstance(kind, str) or isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(
            f"Invalid STAC point dimension type: kind={kind!r}, size={size!r}"
        )
    aliases = {
        ("floating", 4): pa.float32(),
        ("floating", 8): pa.float64(),
        ("signed", 1): pa.int8(),
        ("signed", 2): pa.int16(),
        ("signed", 4): pa.int32(),
        ("signed", 8): pa.int64(),
        ("unsigned", 1): pa.uint8(),
        ("unsigned", 2): pa.uint16(),
        ("unsigned", 4): pa.uint32(),
        ("unsigned", 8): pa.uint64(),
    }
    try:
        return aliases[(kind.casefold(), size)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported STAC point dimension type: kind={kind!r}, size={size}"
        ) from error


def _pdal_type_to_arrow(kind: str) -> pa.DataType:
    """Convert a PDAL dimension interpretation name to an Arrow type."""
    aliases = {
        "double": pa.float64(),
        "float": pa.float32(),
        "int8_t": pa.int8(),
        "int16_t": pa.int16(),
        "int32_t": pa.int32(),
        "int64_t": pa.int64(),
        "uint8_t": pa.uint8(),
        "uint16_t": pa.uint16(),
        "uint32_t": pa.uint32(),
        "uint64_t": pa.uint64(),
    }
    try:
        return aliases[kind.casefold()]
    except KeyError as error:
        raise ValueError(f"Unsupported PDAL point dimension type: {kind!r}") from error


def sources_to_arrow(sources: Sequence[CopcSource]) -> pa.Table:
    """Convert normalized source metadata to a stable Arrow manifest table."""
    dimension_type = pa.list_(
        pa.struct([pa.field("name", pa.string()), pa.field("type", pa.string())])
    )
    return pa.table(
        {
            "id": pa.array([source.id for source in sources], type=pa.string()),
            "href": pa.array([source.href for source in sources], type=pa.string()),
            "min_x": pa.array(
                [source.bounds.min_x for source in sources], type=pa.float64()
            ),
            "min_y": pa.array(
                [source.bounds.min_y for source in sources], type=pa.float64()
            ),
            "min_z": pa.array(
                [source.bounds.min_z for source in sources], type=pa.float64()
            ),
            "max_x": pa.array(
                [source.bounds.max_x for source in sources], type=pa.float64()
            ),
            "max_y": pa.array(
                [source.bounds.max_y for source in sources], type=pa.float64()
            ),
            "max_z": pa.array(
                [source.bounds.max_z for source in sources], type=pa.float64()
            ),
            "point_count": pa.array(
                [source.point_count for source in sources], type=pa.uint64()
            ),
            "crs_wkt": pa.array(
                [source.crs_wkt for source in sources], type=pa.string()
            ),
            "dimensions": pa.array(
                [
                    [
                        {"name": dimension.name, "type": str(dimension.data_type)}
                        for dimension in source.dimensions
                    ]
                    for source in sources
                ],
                type=dimension_type,
            ),
            "properties": pa.array(
                [
                    json.dumps(
                        source.properties,
                        default=_json_default,
                        sort_keys=True,
                    )
                    for source in sources
                ],
                type=pa.string(),
            ),
        }
    )


def _json_default(value: Any) -> str:
    """Serialize date-like and otherwise non-JSON metadata values as strings."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
