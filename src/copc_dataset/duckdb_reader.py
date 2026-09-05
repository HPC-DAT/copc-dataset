"""DuckDB/PDAL extraction of logical COPC dataset selections to Arrow."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any, Self

import duckdb
import pyarrow as pa

from copc_dataset.catalog import CopcSource, pdal_copc_href
from copc_dataset.dataset import CopcDataset
from copc_dataset.selection import Bounds3D, DimensionRange, PointCloudSelection

_SOURCE_COLUMNS = ("__source_id", "__source_href")


class CopcBatchReader:
    """A closeable iterator over Arrow batches from one logical selection."""

    def __init__(self, schema: pa.Schema, batches: Iterator[pa.RecordBatch]) -> None:
        """Store the stable output schema and underlying closeable iterator."""
        self.schema = schema
        self._batches = batches
        self._closed = False

    def __iter__(self) -> CopcBatchReader:
        """Return this reader as its own batch iterator."""
        return self

    def __next__(self) -> pa.RecordBatch:
        """Return the next Arrow batch from the logical selection."""
        if self._closed:
            raise StopIteration
        try:
            return next(self._batches)
        except StopIteration:
            self.close()
            raise

    def __enter__(self) -> Self:
        """Enter a context that closes source resources on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close source resources when leaving a reader context."""
        self.close()

    @property
    def closed(self) -> bool:
        """Return whether this reader has released its source iterator."""
        return self._closed

    def read_all(self) -> pa.Table:
        """Consume all remaining batches and return one Arrow table."""
        try:
            return pa.Table.from_batches(list(self), schema=self.schema)
        finally:
            self.close()

    def close(self) -> None:
        """Close the batch generator and its DuckDB/PDAL resources."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._batches, "close", None)
        if close is not None:
            close()


class DuckDBCopcReader:
    """Read bounded selections from a logical COPC dataset through DuckDB."""

    def __init__(
        self,
        *,
        requests: int = 4,
        resolution: float | None = None,
        batch_size: int = 65_536,
        include_source: bool = False,
        install_extension: bool = True,
    ) -> None:
        """Configure COPC network concurrency, resolution, and Arrow batching."""
        if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
            raise ValueError("requests must be at least 1")
        if resolution is not None and (
            not math.isfinite(resolution) or resolution <= 0
        ):
            raise ValueError("resolution must be a finite positive number")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")
        self.requests = requests
        self.resolution = resolution
        self.batch_size = batch_size
        self.include_source = include_source
        self.install_extension = install_extension

    def scan(
        self,
        dataset: CopcDataset,
        selection: PointCloudSelection,
    ) -> CopcBatchReader:
        """Return one lazy Arrow batch reader spanning all matching COPC sources."""
        selection = _canonicalize_selection(dataset, selection)
        output_schema = _selection_schema(
            dataset,
            selection,
            include_source=self.include_source,
        )
        batches = self._iter_batches(dataset, selection, output_schema)
        return CopcBatchReader(output_schema, batches)

    def read(
        self,
        dataset: CopcDataset,
        selection: PointCloudSelection,
    ) -> pa.Table:
        """Materialize one dataset selection as an Arrow table."""
        reader = self.scan(dataset, selection)
        try:
            return reader.read_all()
        finally:
            reader.close()

    def _iter_batches(
        self,
        dataset: CopcDataset,
        selection: PointCloudSelection,
        output_schema: pa.Schema,
    ) -> Iterator[pa.RecordBatch]:
        """Yield normalized Arrow batches while retaining only one source view."""
        connection = _connect_pdal(install_extension=self.install_extension)
        try:
            coordinate_names = (
                dataset.canonical_dimension("X"),
                dataset.canonical_dimension("Y"),
                dataset.canonical_dimension("Z"),
            )
            for source in dataset.candidate_sources(selection):
                clipped_bounds = source.bounds.intersection(selection.bounds)
                if clipped_bounds is None:
                    continue
                yield from self._read_source_batches(
                    connection,
                    source,
                    clipped_bounds,
                    selection,
                    output_schema,
                    coordinate_names,
                )
        finally:
            connection.close()

    def _read_source_batches(
        self,
        connection: duckdb.DuckDBPyConnection,
        source: CopcSource,
        clipped_bounds: Bounds3D,
        selection: PointCloudSelection,
        output_schema: pa.Schema,
        coordinate_names: tuple[str, str, str],
    ) -> Iterator[pa.RecordBatch]:
        """Execute and yield Arrow batches for one source-level bounded read."""
        point_columns = selection.columns or tuple(
            output_schema.names[: -len(_SOURCE_COLUMNS)]
            if self.include_source
            else output_schema.names
        )
        projection = ", ".join(_quote_identifier(column) for column in point_columns)
        expression = _compile_expression(
            clipped_bounds,
            selection.filters,
            coordinate_names=coordinate_names,
        )
        pipeline = json.dumps(
            [{"type": "filters.expression", "expression": expression}]
        )
        bounds_json = json.dumps(clipped_bounds.as_geojson_bbox())

        options_sql = "MAP {'bounds': ?, 'requests': ?}"
        with pdal_copc_href(source.href) as reader_href:
            parameters: list[Any] = [
                reader_href,
                pipeline,
                bounds_json,
                str(self.requests),
            ]
            if self.resolution is not None:
                options_sql = "MAP {'bounds': ?, 'requests': ?, 'resolution': ?}"
                parameters.append(str(self.resolution))

            query = f"""
                SELECT {projection}
                FROM PDAL_Pipeline(?, ?, options => {options_sql})
            """
            arrow_reader = connection.execute(query, parameters).to_arrow_reader(
                self.batch_size
            )
            try:
                point_schema = pa.schema(
                    [output_schema.field(column) for column in point_columns]
                )
                for batch in arrow_reader:
                    normalized = batch.cast(point_schema)
                    if self.include_source:
                        normalized = _append_source_columns(normalized, source)
                    yield normalized.cast(output_schema)
            finally:
                arrow_reader.close()


def _connect_pdal(*, install_extension: bool) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with its PDAL extension loaded."""
    connection = duckdb.connect(database=":memory:")
    try:
        if install_extension:
            connection.execute("INSTALL pdal FROM community")
        connection.execute("LOAD pdal")
    except BaseException:
        connection.close()
        raise
    return connection


def _canonicalize_selection(
    dataset: CopcDataset, selection: PointCloudSelection
) -> PointCloudSelection:
    """Canonicalize a directly constructed selection against dataset names."""
    filters = tuple(
        DimensionRange(
            dataset.canonical_dimension(dimension.name),
            dimension.minimum,
            dimension.maximum,
        )
        for dimension in selection.filters
    )
    columns = (
        tuple(dataset.canonical_dimension(column) for column in selection.columns)
        if selection.columns is not None
        else None
    )
    return PointCloudSelection(selection.id, selection.bounds, filters, columns)


def _selection_schema(
    dataset: CopcDataset,
    selection: PointCloudSelection,
    *,
    include_source: bool,
) -> pa.Schema:
    """Build the stable Arrow schema returned for a selection."""
    columns = selection.columns or dataset.dimensions
    fields = [dataset.schema.field(column) for column in columns]
    if include_source:
        collisions = sorted(set(dataset.dimensions).intersection(_SOURCE_COLUMNS))
        if collisions:
            raise ValueError(
                f"Source provenance columns collide with point dimensions: {collisions}"
            )
        fields.extend(
            [
                pa.field("__source_id", pa.string(), nullable=False),
                pa.field("__source_href", pa.string(), nullable=False),
            ]
        )
    return pa.schema(fields)


def _compile_expression(
    bounds: Bounds3D,
    filters: tuple[DimensionRange, ...],
    *,
    coordinate_names: tuple[str, str, str] = ("X", "Y", "Z"),
) -> str:
    """Compile validated ranges into a PDAL ``filters.expression`` predicate."""
    x_name, y_name, z_name = coordinate_names
    for name in (*coordinate_names, *(item.name for item in filters)):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(
                f"Dimension name is not valid in a PDAL expression: {name!r}"
            )
    clauses = [
        f"{x_name} >= {_format_number(bounds.min_x)}",
        f"{x_name} <= {_format_number(bounds.max_x)}",
        f"{y_name} >= {_format_number(bounds.min_y)}",
        f"{y_name} <= {_format_number(bounds.max_y)}",
        f"{z_name} >= {_format_number(bounds.min_z)}",
        f"{z_name} <= {_format_number(bounds.max_z)}",
    ]
    for dimension in filters:
        if dimension.minimum is not None:
            clauses.append(f"{dimension.name} >= {_format_number(dimension.minimum)}")
        if dimension.maximum is not None:
            clauses.append(f"{dimension.name} <= {_format_number(dimension.maximum)}")
    return " && ".join(f"({clause})" for clause in clauses)


def _format_number(value: float) -> str:
    """Format a previously validated finite number for a PDAL expression."""
    if not math.isfinite(value):
        raise ValueError(f"PDAL expression values must be finite: {value!r}")
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return format(value, ".17g")


def _quote_identifier(name: str) -> str:
    """Quote a validated DuckDB identifier without treating it as SQL text."""
    return '"' + name.replace('"', '""') + '"'


def _append_source_columns(batch: pa.RecordBatch, source: CopcSource) -> pa.RecordBatch:
    """Append optional source provenance columns to an Arrow batch."""
    source_ids = pa.array([source.id] * batch.num_rows, type=pa.string())
    source_hrefs = pa.array([source.href] * batch.num_rows, type=pa.string())
    return batch.append_column("__source_id", source_ids).append_column(
        "__source_href", source_hrefs
    )
