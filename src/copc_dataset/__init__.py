"""Public API for metadata-first access to collections of COPC files."""

from copc_dataset.catalog import CopcSource, PointDimension
from copc_dataset.dask import DaskCopcExecutor, SubsetProcessor
from copc_dataset.dataset import CopcDataset, open_copc_dataset
from copc_dataset.duckdb_reader import CopcBatchReader, DuckDBCopcReader
from copc_dataset.selection import Bounds3D, DimensionRange, PointCloudSelection

__all__ = [
    "Bounds3D",
    "CopcBatchReader",
    "CopcDataset",
    "CopcSource",
    "DaskCopcExecutor",
    "DimensionRange",
    "DuckDBCopcReader",
    "PointCloudSelection",
    "PointDimension",
    "SubsetProcessor",
    "open_copc_dataset",
]
