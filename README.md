# COPC Dataset

This repository contains `copc_dataset`, a metadata-first Python package for
treating a collection of Cloud Optimized Point Cloud (COPC) files as one
logical point-cloud dataset.

Opening a dataset reads source metadata such as point counts, dimensions,
bounds, CRS, and properties. It does not load point records. Point data is read
only when a bounded selection is passed to the DuckDB/PDAL reader.

Application-specific partitioners can
inspect the dataset metadata and produce reusable `PointCloudSelection`
objects.

## Status

This is an initial implementation. The public API may still change before a
stable release.

Note: The current DuckDB PDAL extension materializes the points selected from one
COPC source before DuckDB exposes them as Arrow batches. The package processes
matching sources one at a time, but it cannot make that source-level operation
fully streaming.

## Installation

The compiled geospatial dependencies are managed through Conda in this
repository:

```bash
conda env create -f environment.yml
conda activate copc
python -m pip install -e .
```

For an existing environment:

```bash
conda env update -n copc -f environment.yml --prune
conda activate copc
python -m pip install -e .
```

DuckDB's PDAL community extension is installed automatically by default. This
requires network access on its first use. Set `install_extension=False` after
the extension has been installed and cached.

## Opening A Dataset

`open_copc_dataset` accepts a local COPC file, directory, glob pattern, URL
collection, or compatible STAC GeoParquet catalog.

```python
from copc_dataset import open_copc_dataset

local_dataset = open_copc_dataset("/data/copc/*.copc.laz")

remote_dataset = open_copc_dataset(
    [
        "https://example.test/tile-1.copc.laz",
        "https://example.test/tile-2.copc.laz",
    ]
)

catalog_dataset = open_copc_dataset("ahn5.parquet")
```

Opening files or URLs inspects their COPC metadata. Opening GeoParquet trusts
the catalog and does not contact the referenced COPC assets.

The dataset has an xarray-inspired metadata representation:

```python
print(catalog_dataset)
print(catalog_dataset.point_count)
print(catalog_dataset.bounds)
print(catalog_dataset.schema)
print(catalog_dataset.crs)
print(catalog_dataset.sources)
```

`point_count` is the sum of physical point records reported by all sources. It
is not a distinct count when source files overlap.

## Metadata Contract

All COPCs in one dataset must have:

- Numeric X, Y, and Z dimensions.
- The same dimension names and identical data types.
- Semantically equivalent coordinate reference systems.
- Unique source IDs and source locations.

The package fails while opening incompatible collections rather than silently
dropping dimensions or filling missing dimensions with null values.

`dataset.sources` is an Arrow table containing one row per source, including
the source location, 3D bounds, physical point count, CRS, dimensions, and
source properties. `dataset.source_records` exposes the same information as
frozen Python metadata records. Nested `attrs` and `properties` mappings should
be treated as read-only application metadata.

The initial GeoParquet adapter supports the following STAC columns (as used by the included AHN
catalog):

- `id`
- `assets`
- `proj:bbox`
- `pc:count`
- `pc:schemas`
- `proj:wkt2`

Core STAC `bbox` is not used for point selection because it normally contains
WGS84 coordinates. `proj:bbox` supplies bounds in the COPC asset CRS identified
by `proj:wkt2`. Relative asset paths are resolved while opening the catalog so
workers do not depend on their current working directory.

## Defining Selections

Every selection is a coherent, closed, axis-aligned 3D volume. X, Y, and Z are
mandatory in the resolved selection, but each omitted range defaults to the
complete dataset range.

```python
selection = catalog_dataset.selection(
    "study-area-001",
    x=(124_500.0, 124_750.0),
    y=(485_000.0, 485_250.0),
    z=None,
    columns=("X", "Y", "Z", "Intensity", "Classification"),
)
```

One-sided coordinate ranges use `None`:

```python
selection = catalog_dataset.selection(
    "upper-east",
    x=(125_000.0, None),
    y=None,
    z=(0.0, None),
)
```

Complementary numeric ranges can filter any non-coordinate point dimension:

```python
selection = catalog_dataset.selection(
    "classified",
    x=(124_500.0, 124_750.0),
    y=(485_000.0, 485_250.0),
    z=None,
    filters={
        "Classification": (2, 5),
        "Intensity": (100, None),
    },
    columns=("X", "Y", "Z", "Intensity", "Classification"),
)
```

Coordinate ranges determine spatial coherence and source pruning. Other
dimension filters complement them but do not replace them. All ranges are
inclusive.

The dataset metadata can be used by future partitioning implementations:

```python
sources = catalog_dataset.candidate_sources(selection)
```

No point records are read by `selection` or `candidate_sources`.

## Reading Arrow Data

Use `scan` to consume a closeable `CopcBatchReader` of Arrow record batches or
`read` to materialize one Arrow table:

```python
from copc_dataset import DuckDBCopcReader

reader = DuckDBCopcReader(
    requests=4,
    batch_size=65_536,
    include_source=False,
)

with reader.scan(catalog_dataset, selection) as batches:
    for batch in batches:
        process_batch(batch)

table = reader.read(catalog_dataset, selection)
```

The reader performs the following steps for each selection:

1. Prune COPC sources using their metadata bounds.
2. Intersect the requested 3D bounds with each source extent.
3. Pass the clipped bounds to `readers.copc` for hierarchy pruning.
4. Apply exact coordinate and complementary dimension predicates through a
   PDAL expression filter.
5. Project requested dimensions and expose them as Arrow batches.
6. Advance to the next source only after the current source is exhausted.

Set `include_source=True` to append `__source_id` and `__source_href` to every
point. These provenance columns are disabled by default because repeated
strings increase memory use.

COPC `resolution` and per-reader HTTP `requests` are reader options because
they affect execution rather than the logical identity of a selection.

## Dask Execution

`DaskCopcExecutor` submits one task per selection. The metadata-only dataset is
scattered to workers, while each worker creates its own DuckDB connection.

```python
from distributed import Client, LocalCluster
from copc_dataset import DaskCopcExecutor

with LocalCluster(n_workers=4, threads_per_worker=1) as cluster:
    with Client(cluster) as client:
        executor = DaskCopcExecutor(
            client,
            reader_options={
                "requests": 2,
                "install_extension": False,
            },
        )

        futures = executor.submit(catalog_dataset, selections)
        first_table = futures["study-area-001"].result()
```

The number of workers and each reader's `requests` value jointly determine
network concurrency.

For large selections, run downstream processing on the worker rather than
returning all points to the Dask client. A processor receives the Arrow stream,
selection, and metadata-only dataset:

```python
def count_points(batches, selection, dataset):
    """Count points in one selection without returning its point records."""
    return {
        "selection_id": selection.id,
        "point_count": sum(batch.num_rows for batch in batches),
        "dataset_crs": dataset.crs.to_string(),
    }


with LocalCluster(n_workers=4, threads_per_worker=1) as cluster:
    with Client(cluster) as client:
        executor = DaskCopcExecutor(
            client,
            reader_options={"install_extension": False},
        )
        futures = executor.submit(
            catalog_dataset,
            selections,
            processor=count_points,
        )
        results = {
            selection_id: future.result()
            for selection_id, future in futures.items()
        }
```

Processors should be importable callables so Dask can serialize them reliably.
The processor must consume the stream before returning.

## Module Relationships

The package modules have deliberately narrow responsibilities:

```text
catalog.py
  Discovers sources, reads COPC metadata, and normalizes STAC GeoParquet.
       |
       v
dataset.py
  Validates one logical dataset and exposes aggregate/source metadata.
       |
       v
selection.py
  Defines immutable 3D bounds and complementary dimension ranges.
       |
       v
duckdb_reader.py
  Converts dataset + selection into bounded PDAL reads and Arrow batches.
       |
       v
dask.py
  Executes independent selections and processors on Dask workers.
```

`selection.py` contains no I/O. `CopcDataset` contains no open connection or
point data. `DuckDBCopcReader` knows nothing about how selections were
partitioned. This allows a later Laserchicken/Laserfarm partitioner to depend
on the public metadata and selection types without becoming part of the reader.

## Repository Files

- `src/copc_dataset/`: installable reusable package.
- `tests/`: unit tests and optional local COPC integration test.
- `ahn5.parquet`: nine-source AHN STAC GeoParquet example catalog.
- `ahn5.vpc`: source virtual point-cloud catalog used to create GeoParquet.
- `aoi.json`: projected AHN example area of interest.
- `environment.yml`: reproducible Conda environment.

## Testing

Run unit tests with:

```bash
PYTHONPATH=src python -m pytest
```

The tests use synthetic metadata and do not require remote point data. To run
the optional bounded integration tests against a local fixture containing
classification-2 points near the center of its extent:

```bash
COPC_TEST_FILE=/path/to/example.copc.laz \
PYTHONPATH=src python -m pytest tests/test_duckdb_reader.py tests/test_dask.py
```

The PDAL extension must already be installed for this integration test.

## Current Limitations

- Only rectangular X/Y/Z selections are supported.
- Point dimensions must be compatible across every source.
- Overlapping source files are not deduplicated.
- Existing catalogs are trusted unless their COPCs are opened separately.
- No point ordering is guaranteed.
- Uppercase remote `.COPC.LAZ` URLs require an internal inference workaround.
- Uppercase local `.COPC.LAZ` files require a temporary lowercase hard link in
  the source directory; that directory must be writable and its filesystem must
  support hard links.
- Arrow batching does not remove the current DuckDB PDAL extension's
  source-level `PointView` materialization.

## AHN Catalog Preparation

The included AHN catalog was prepared from a list of COPC URLs:

```bash
pdal_wrench build_vpc --output=ahn5.vpc --input-file-list=selectie.txt
rustac translate -i json ahn5.vpc ahn5.parquet
```

The generic package does not require this exact preparation route; it can
inspect local files or URL collections directly.
