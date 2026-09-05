"""Dask execution of independent point-cloud selections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from distributed import Client, Future

from copc_dataset.dataset import CopcDataset
from copc_dataset.duckdb_reader import CopcBatchReader, DuckDBCopcReader
from copc_dataset.selection import PointCloudSelection

Result = TypeVar("Result")


class SubsetProcessor(Protocol[Result]):
    """Protocol for processing one selection's Arrow stream on a Dask worker."""

    def __call__(
        self,
        batches: CopcBatchReader,
        selection: PointCloudSelection,
        dataset: CopcDataset,
    ) -> Result:
        """Consume a selection stream and return a serializable result."""
        ...


class DaskCopcExecutor:
    """Submit independent COPC selections to a Dask distributed client."""

    def __init__(
        self,
        client: Client,
        *,
        reader_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the client and options used to construct readers on workers."""
        self.client = client
        self.reader_options = dict(reader_options or {})

    def submit(
        self,
        dataset: CopcDataset,
        selections: Sequence[PointCloudSelection],
        *,
        processor: SubsetProcessor[Any] | None = None,
        retries: int = 0,
    ) -> dict[str, Future]:
        """Submit selections and return futures keyed by stable selection ID."""
        selection_list = tuple(selections)
        ids = [selection.id for selection in selection_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Selection ids must be unique within one submission")
        if retries < 0:
            raise ValueError("retries must not be negative")

        dataset_future = self.client.scatter(dataset, broadcast=True)
        return {
            selection.id: self.client.submit(
                _execute_selection,
                dataset_future,
                selection,
                self.reader_options,
                processor,
                retries=retries,
                pure=False,
            )
            for selection in selection_list
        }


def _execute_selection(
    dataset: CopcDataset,
    selection: PointCloudSelection,
    reader_options: Mapping[str, Any],
    processor: Callable[[CopcBatchReader, PointCloudSelection, CopcDataset], Any]
    | None,
) -> Any:
    """Read and optionally process one selection entirely on a Dask worker."""
    reader = DuckDBCopcReader(**dict(reader_options))
    batches = reader.scan(dataset, selection)
    try:
        if processor is None:
            return batches.read_all()
        return processor(batches, selection, dataset)
    finally:
        batches.close()
