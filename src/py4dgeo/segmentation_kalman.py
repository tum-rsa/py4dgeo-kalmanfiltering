# ============================================================
# segmentation_kalman.py
# KALMAN-SEED INTEGRATION WITH PY4DGEO 4D-OBC REGION GROWING
# ============================================================

import numpy as np
import pandas as pd

from py4dgeo.segmentation import (
    RegionGrowingAlgorithm,
    RegionGrowingSeed,
)


# ============================================================
# SEED INTERVAL HELPERS
# ============================================================

def get_seed_start_epoch(
    row,
    use_support_interval=True
):
    """
    Return the seed start epoch used for region growing.

    If a validated spatial-support interval exists and
    use_support_interval=True, support_start_epoch is preferred.
    Otherwise start_epoch is used.
    """

    if (
        use_support_interval
        and "support_start_epoch" in row.index
        and not pd.isna(
            row["support_start_epoch"]
        )
    ):
        return int(
            row["support_start_epoch"]
        )

    return int(
        row["start_epoch"]
    )


def get_seed_end_epoch(
    row,
    use_support_interval=True
):
    """
    Return the seed end epoch used for region growing.

    If a validated spatial-support interval exists and
    use_support_interval=True, support_end_epoch is preferred.
    Otherwise end_epoch is used.
    """

    if (
        use_support_interval
        and "support_end_epoch" in row.index
        and not pd.isna(
            row["support_end_epoch"]
        )
    ):
        return int(
            row["support_end_epoch"]
        )

    return int(
        row["end_epoch"]
    )


# ============================================================
# PREPARE EXTERNAL SEEDS FOR PY4DGEO
# ============================================================

def prepare_seed_table_for_py4dgeo(
    seed_table,
    seed_mode="all",
    top_n=20,
    use_support_interval=True,
    ranking_columns=None
):
    """
    Prepare an externally generated seed table for py4dgeo
    region growing.

    Parameters
    ----------
    seed_table : pandas.DataFrame
        Must contain:
            corepoint_index_python
            start_epoch
            end_epoch

    seed_mode : {"all", "top_n"}
        Whether all candidate seeds or only the highest-ranked
        seeds are passed to region growing.

    top_n : int
        Number of seeds retained when seed_mode="top_n".

    use_support_interval : bool
        Prefer support_start_epoch/support_end_epoch when those
        columns are present.

    ranking_columns : list[str] or None
        Columns used for ranking.

        If None:
            event_magnitude is used when available,
            followed by duration_epochs when available.

    Returns
    -------
    pandas.DataFrame
        Prepared seed table containing py4dgeo-compatible
        temporal intervals.
    """

    if (
        seed_table is None
        or len(seed_table) == 0
    ):
        raise ValueError(
            "Seed table is empty."
        )

    required_columns = [
        "corepoint_index_python",
        "start_epoch",
        "end_epoch",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in seed_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from seed table: "
            + ", ".join(missing_columns)
        )

    prepared = seed_table.copy()

    prepared[
        "py4dgeo_seed_start_epoch"
    ] = prepared.apply(
        lambda row: get_seed_start_epoch(
            row,
            use_support_interval=(
                use_support_interval
            ),
        ),
        axis=1,
    )

    prepared[
        "py4dgeo_seed_end_epoch"
    ] = prepared.apply(
        lambda row: get_seed_end_epoch(
            row,
            use_support_interval=(
                use_support_interval
            ),
        ),
        axis=1,
    )

    prepared[
        "py4dgeo_seed_duration_epochs"
    ] = (
        prepared[
            "py4dgeo_seed_end_epoch"
        ]
        - prepared[
            "py4dgeo_seed_start_epoch"
        ]
        + 1
    )

    prepared = prepared[
        prepared[
            "py4dgeo_seed_duration_epochs"
        ] > 0
    ].copy()

    if len(prepared) == 0:
        raise ValueError(
            "No valid seed intervals remain after "
            "preparing the py4dgeo seed intervals."
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    if ranking_columns is None:

        ranking_columns = []

        if (
            "event_magnitude"
            in prepared.columns
        ):
            ranking_columns.append(
                "event_magnitude"
            )

        if (
            "duration_epochs"
            in prepared.columns
        ):
            ranking_columns.append(
                "duration_epochs"
            )

        if len(ranking_columns) == 0:
            ranking_columns = [
                "py4dgeo_seed_duration_epochs"
            ]

    missing_ranking_columns = [
        column
        for column in ranking_columns
        if column not in prepared.columns
    ]

    if missing_ranking_columns:
        raise ValueError(
            "Ranking columns missing from seed table: "
            + ", ".join(
                missing_ranking_columns
            )
        )

    prepared = prepared.sort_values(
        ranking_columns,
        ascending=[
            False
        ] * len(ranking_columns),
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Seed selection
    # --------------------------------------------------------

    if seed_mode == "all":

        selected = prepared.copy()

    elif seed_mode == "top_n":

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        selected = prepared.head(
            int(top_n)
        ).copy()

    else:

        raise ValueError(
            "seed_mode must be "
            "'all' or 'top_n'."
        )

    selected[
        "py4dgeo_custom_seed_rank"
    ] = np.arange(
        1,
        len(selected) + 1,
        dtype=int,
    )

    return selected


# ============================================================
# KALMAN-SEED REGION GROWING
# ============================================================

class KalmanSeedRegionGrowing(
    RegionGrowingAlgorithm
):
    """
    py4dgeo 4D-OBC region growing using externally generated
    Kalman-based seeds.

    The existing py4dgeo region-growing implementation is
    retained. Only seed generation is replaced.

    find_seedpoints() converts the prepared KF-Mag, KF-Rate,
    KF-Mag-NMS or KF-Rate-NMS seed table into
    RegionGrowingSeed objects.
    """

    def __init__(
        self,
        seed_table,
        seed_mode="all",
        top_n=20,
        use_support_interval=True,
        method_label="kalman_seed_region_growing",
        ranking_columns=None,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.original_seed_table = (
            seed_table.copy()
        )

        self.seed_mode = (
            seed_mode
        )

        self.top_n = (
            top_n
        )

        self.use_support_interval = (
            use_support_interval
        )

        self.method_label = (
            method_label
        )

        self.ranking_columns = (
            ranking_columns
        )

        self.prepared_seed_table = (
            prepare_seed_table_for_py4dgeo(
                seed_table=(
                    self.original_seed_table
                ),
                seed_mode=(
                    self.seed_mode
                ),
                top_n=(
                    self.top_n
                ),
                use_support_interval=(
                    self.use_support_interval
                ),
                ranking_columns=(
                    self.ranking_columns
                ),
            )
        )

    def find_seedpoints(self):
        """
        Return the externally defined seeds in py4dgeo's
        RegionGrowingSeed representation.
        """

        seeds = []

        for _, row in (
            self.prepared_seed_table.iterrows()
        ):

            corepoint_index = int(
                row[
                    "corepoint_index_python"
                ]
            )

            start_epoch = int(
                row[
                    "py4dgeo_seed_start_epoch"
                ]
            )

            end_epoch = int(
                row[
                    "py4dgeo_seed_end_epoch"
                ]
            )

            seeds.append(
                RegionGrowingSeed(
                    corepoint_index,
                    start_epoch,
                    end_epoch,
                )
            )

        return seeds


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_kalman_region_growing(
    seed_table,
    method_label,
    seed_mode="all",
    top_n=20,
    use_support_interval=True,
    ranking_columns=None,
    window_width=14,
    minperiod=2,
    height_threshold=0.05,
    neighborhood_radius=1.0,
    min_segments=10,
    thresholds=None,
):
    """
    Create a py4dgeo region-growing algorithm using externally
    generated Kalman-based seeds.

    The spatial region-growing implementation itself remains
    the existing py4dgeo RegionGrowingAlgorithm.
    """

    if thresholds is None:
        thresholds = [
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]

    return KalmanSeedRegionGrowing(
        seed_table=seed_table,
        seed_mode=seed_mode,
        top_n=top_n,
        use_support_interval=(
            use_support_interval
        ),
        method_label=method_label,
        ranking_columns=(
            ranking_columns
        ),
        window_width=window_width,
        minperiod=minperiod,
        height_threshold=(
            height_threshold
        ),
        neighborhood_radius=(
            neighborhood_radius
        ),
        min_segments=min_segments,
        thresholds=thresholds,
    )