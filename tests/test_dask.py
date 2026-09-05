"""Tests for Dask submission of independent COPC selections."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock

from distributed import Client, LocalCluster

from copc_dataset import DaskCopcExecutor, open_copc_dataset
from tests.helpers import dataset


def count_batches(batches, selection, dataset_metadata):
    """Count a selection's Arrow batches inside a Dask worker."""
    return (
        selection.id,
        sum(batch.num_rows for batch in batches),
        dataset_metadata.source_count,
    )


class DaskExecutorTests(unittest.TestCase):
    """Verify submission contracts without requiring point-cloud I/O."""

    def test_submits_futures_by_selection_id(self) -> None:
        """Scatter metadata once and retain stable selection-to-future mapping."""
        client = Mock()
        client.scatter.return_value = "dataset-future"
        client.submit.side_effect = ["west-future", "east-future"]
        value = dataset()
        selections = (
            value.selection("west", x=(0, 5)),
            value.selection("east", x=(15, 20)),
        )

        futures = DaskCopcExecutor(
            client,
            reader_options={"requests": 2},
        ).submit(value, selections)

        self.assertEqual(
            futures,
            {"west": "west-future", "east": "east-future"},
        )
        client.scatter.assert_called_once_with(value, broadcast=True)
        self.assertEqual(client.submit.call_count, 2)

    def test_rejects_duplicate_selection_ids(self) -> None:
        """Prevent result futures from overwriting one another by key."""
        value = dataset()
        selections = (
            value.selection("same", x=(0, 5)),
            value.selection("same", x=(15, 20)),
        )

        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            DaskCopcExecutor(Mock()).submit(value, selections)


@unittest.skipUnless(
    os.environ.get("COPC_TEST_FILE"),
    "Set COPC_TEST_FILE to run local Dask integration tests",
)
class LocalDaskIntegrationTests(unittest.TestCase):
    """Exercise Arrow-table and worker-callback futures against a local COPC."""

    def test_executes_independent_selections(self) -> None:
        """Read two bounded subsets concurrently and retain their selection IDs."""
        value = open_copc_dataset(
            Path(os.environ["COPC_TEST_FILE"]),
            install_extension=False,
        )
        center_x = (value.bounds.min_x + value.bounds.max_x) / 2
        center_y = (value.bounds.min_y + value.bounds.max_y) / 2
        selections = (
            value.selection(
                "west",
                x=(center_x - 10, center_x),
                y=(center_y - 5, center_y + 5),
                columns=("X", "Y", "Z"),
            ),
            value.selection(
                "east",
                x=(center_x, center_x + 10),
                y=(center_y - 5, center_y + 5),
                columns=("X", "Y", "Z"),
            ),
        )
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        )
        client = Client(cluster)
        try:
            executor = DaskCopcExecutor(
                client,
                reader_options={
                    "requests": 1,
                    "install_extension": False,
                },
            )
            counted = client.gather(
                list(
                    executor.submit(
                        value,
                        selections,
                        processor=count_batches,
                    ).values()
                )
            )
            table = executor.submit(value, selections[:1])["west"].result()
        finally:
            client.close()
            cluster.close()

        self.assertEqual({item[0] for item in counted}, {"west", "east"})
        self.assertTrue(all(item[1] > 0 for item in counted))
        self.assertTrue(all(item[2] == 1 for item in counted))
        self.assertGreater(table.num_rows, 0)
        self.assertEqual(table.schema.names, ["X", "Y", "Z"])


if __name__ == "__main__":
    unittest.main()
