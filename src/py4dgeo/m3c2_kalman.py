# ============================================================
# m3c2_kalman.py
# INTEGRATED KALMAN-BASED 4D-OBC SEED DETECTION FOR PY4DGEO
# ============================================================

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from filterpy.kalman import KalmanFilter
from scipy.spatial import cKDTree

from py4dgeo.segmentation import (
    RegionGrowingAlgorithm,
    RegionGrowingSeed,
)

from py4dgeo.util import Py4DGeoError


# ============================================================
# MODULE-LEVEL IN-MEMORY CACHE
# ============================================================

# Keeps a valid Kalman solution available to other algorithm
# instances during the same Python/Jupyter session.
#
# Persistent disk caching is handled separately below.
_MEMORY_KALMAN_CACHE = {}

_KALMAN_CACHE_VERSION = "kalman-rts-v1"


# ============================================================
# TIME HANDLING
# ============================================================

def _timedeltas_to_days(timedeltas):
    times_days = np.asarray(
        [
            td.total_seconds() / (60.0 * 60.0 * 24.0)
            for td in timedeltas
        ],
        dtype=float,
    )

    if times_days.size == 0:
        raise Py4DGeoError(
            "SpatiotemporalAnalysis contains no timedeltas."
        )

    dt_series = np.diff(
        times_days,
        prepend=times_days[0],
    )

    if dt_series.size > 1:
        dt_series[0] = dt_series[1]
    else:
        dt_series[0] = 1.0

    if np.any(dt_series <= 0):
        raise Py4DGeoError(
            "Kalman filtering requires positive temporal "
            "step lengths."
        )

    return dt_series


# ============================================================
# DATA / CONFIGURATION FINGERPRINTING
# ============================================================

def _update_hash_with_array(hasher, array):
    """
    Add an array to a SHA-256 hash without changing its values.

    The array shape, dtype and numerical contents are included.
    This ensures that a cache generated for one M3C2 dataset
    cannot silently be reused for another dataset.
    """

    array = np.asarray(array)

    hasher.update(
        str(array.shape).encode("utf-8")
    )
    hasher.update(
        str(array.dtype).encode("utf-8")
    )

    contiguous = np.ascontiguousarray(array)

    byte_view = memoryview(
        contiguous
    ).cast("B")

    chunk_size = 16 * 1024 * 1024

    for start in range(
        0,
        len(byte_view),
        chunk_size,
    ):
        hasher.update(
            byte_view[
                start:
                start + chunk_size
            ]
        )


def _build_kalman_cache_key(
    distances,
    lodetection,
    dt_series,
    process_sigma,
    min_sigma_obs,
):
    """
    Build a fingerprint for the inputs that determine the
    Kalman solution.

    Detection mode, seed thresholds, spatial support and NMS
    parameters are deliberately excluded because they operate
    after the Kalman states have already been calculated.
    """

    hasher = hashlib.sha256()

    hasher.update(
        _KALMAN_CACHE_VERSION.encode("utf-8")
    )

    hasher.update(
        repr(float(process_sigma)).encode("utf-8")
    )

    hasher.update(
        repr(float(min_sigma_obs)).encode("utf-8")
    )

    _update_hash_with_array(
        hasher,
        distances,
    )

    _update_hash_with_array(
        hasher,
        lodetection,
    )

    _update_hash_with_array(
        hasher,
        dt_series,
    )

    return hasher.hexdigest()


# ============================================================
# PERSISTENT CACHE HELPERS
# ============================================================

def _cache_directory(
    cache_root,
    cache_key,
):
    cache_root = Path(
        cache_root
    )

    return (
        cache_root
        / cache_key
    )


def _load_kalman_cache(
    cache_root,
    cache_key,
    expected_shape,
):
    """
    Load a validated Kalman cache.

    Returns
    -------
    dict or None
        None means that no valid cache is available.
    """

    cache_dir = _cache_directory(
        cache_root,
        cache_key,
    )

    metadata_path = (
        cache_dir
        / "metadata.json"
    )

    if not metadata_path.exists():
        return None

    required_files = {
        "change":
            cache_dir / "change.npy",
        "rate":
            cache_dir / "rate.npy",
        "sigma_change":
            cache_dir / "sigma_change.npy",
        "sigma_rate":
            cache_dir / "sigma_rate.npy",
        "dt_days":
            cache_dir / "dt_days.npy",
    }

    if not all(
        path.exists()
        for path in required_files.values()
    ):
        return None

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if (
        metadata.get("cache_version")
        != _KALMAN_CACHE_VERSION
    ):
        return None

    if (
        metadata.get("cache_key")
        != cache_key
    ):
        return None

    if (
        tuple(
            metadata.get(
                "distances_shape",
                (),
            )
        )
        != tuple(expected_shape)
    ):
        return None

    try:
        result = {
            "change": np.load(
                required_files["change"],
                mmap_mode="r",
            ),
            "rate": np.load(
                required_files["rate"],
                mmap_mode="r",
            ),
            "sigma_change": np.load(
                required_files[
                    "sigma_change"
                ],
                mmap_mode="r",
            ),
            "sigma_rate": np.load(
                required_files[
                    "sigma_rate"
                ],
                mmap_mode="r",
            ),
            "dt_days": np.load(
                required_files["dt_days"],
                mmap_mode="r",
            ),
        }

    except (
        OSError,
        ValueError,
    ):
        return None

    for key in (
        "change",
        "rate",
        "sigma_change",
        "sigma_rate",
    ):
        if (
            result[key].shape
            != tuple(expected_shape)
        ):
            return None

    return result


def _write_kalman_cache(
    cache_root,
    cache_key,
    change,
    rate,
    sigma_change,
    sigma_rate,
    dt_days,
    process_sigma,
    min_sigma_obs,
):
    """
    Write Kalman states to a dataset/configuration-specific
    cache directory.

    Metadata is written last. Therefore an interrupted write
    will not be considered a valid cache on the next run.
    """

    cache_root = Path(
        cache_root
    )

    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = _cache_directory(
        cache_root,
        cache_key,
    )

    temporary_dir = (
        cache_root
        / f"{cache_key}.tmp"
    )

    if temporary_dir.exists():
        shutil.rmtree(
            temporary_dir
        )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        np.save(
            temporary_dir / "change.npy",
            np.asarray(change),
        )

        np.save(
            temporary_dir / "rate.npy",
            np.asarray(rate),
        )

        np.save(
            temporary_dir / "sigma_change.npy",
            np.asarray(sigma_change),
        )

        np.save(
            temporary_dir / "sigma_rate.npy",
            np.asarray(sigma_rate),
        )

        np.save(
            temporary_dir / "dt_days.npy",
            np.asarray(
                dt_days,
                dtype=float,
            ),
        )

        metadata = {
            "cache_version":
                _KALMAN_CACHE_VERSION,

            "cache_key":
                cache_key,

            "distances_shape":
                list(
                    np.asarray(change).shape
                ),

            "process_sigma":
                float(process_sigma),

            "min_sigma_obs":
                float(min_sigma_obs),
        }

        with (
            temporary_dir / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        if cache_dir.exists():
            shutil.rmtree(
                cache_dir
            )

        temporary_dir.rename(
            cache_dir
        )

    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(
                temporary_dir
            )

        raise


# ============================================================
# KALMAN FILTER + RTS SMOOTHER
# ============================================================

def _kalman_smooth_series(
    observations,
    observation_sigma,
    dt_series,
    process_sigma,
):
    observations = np.asarray(
        observations,
        dtype=float,
    )

    observation_sigma = np.asarray(
        observation_sigma,
        dtype=float,
    )

    dt_series = np.asarray(
        dt_series,
        dtype=float,
    )

    n_epochs = observations.size

    if (
        observation_sigma.shape
        != observations.shape
    ):
        raise ValueError(
            "observation_sigma must have the same "
            "shape as observations."
        )

    if (
        dt_series.shape
        != observations.shape
    ):
        raise ValueError(
            "dt_series must have the same length "
            "as observations."
        )

    valid = np.isfinite(
        observations
    )

    if np.sum(valid) < 2:
        empty = np.full(
            n_epochs,
            np.nan,
            dtype=float,
        )

        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )

    first_valid = int(
        np.where(valid)[0][0]
    )

    filtered_states = np.full(
        (n_epochs, 2),
        np.nan,
        dtype=float,
    )

    filtered_covariances = np.full(
        (n_epochs, 2, 2),
        np.nan,
        dtype=float,
    )

    kf = KalmanFilter(
        dim_x=2,
        dim_z=1,
    )

    kf.x = np.asarray(
        [
            observations[first_valid],
            0.0,
        ],
        dtype=float,
    )

    kf.P = np.eye(
        2,
        dtype=float,
    )

    kf.H = np.asarray(
        [[1.0, 0.0]],
        dtype=float,
    )

    # --------------------------------------------------------
    # Forward Kalman filter
    # --------------------------------------------------------

    for epoch_index in range(
        n_epochs
    ):
        dt = float(
            dt_series[
                epoch_index
            ]
        )

        kf.F = np.asarray(
            [
                [1.0, dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        q = (
            float(process_sigma)
            ** 2
        )

        kf.Q = np.asarray(
            [
                [
                    0.25 * dt**4,
                    0.5 * dt**3,
                ],
                [
                    0.5 * dt**3,
                    dt**2,
                ],
            ],
            dtype=float,
        ) * q

        kf.predict()

        if np.isfinite(
            observations[
                epoch_index
            ]
        ):
            sigma = max(
                float(
                    observation_sigma[
                        epoch_index
                    ]
                ),
                np.finfo(float).eps,
            )

            kf.R = np.asarray(
                [[sigma**2]],
                dtype=float,
            )

            kf.update(
                np.asarray(
                    [
                        observations[
                            epoch_index
                        ]
                    ],
                    dtype=float,
                )
            )

        filtered_states[
            epoch_index
        ] = kf.x

        filtered_covariances[
            epoch_index
        ] = kf.P

    # --------------------------------------------------------
    # Rauch-Tung-Striebel smoother
    # --------------------------------------------------------

    smoothed_states = (
        filtered_states.copy()
    )

    smoothed_covariances = (
        filtered_covariances.copy()
    )

    for epoch_index in range(
        n_epochs - 2,
        -1,
        -1,
    ):
        dt = float(
            dt_series[
                epoch_index + 1
            ]
        )

        transition = np.asarray(
            [
                [1.0, dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        q = (
            float(process_sigma)
            ** 2
        )

        process_covariance = (
            np.asarray(
                [
                    [
                        0.25 * dt**4,
                        0.5 * dt**3,
                    ],
                    [
                        0.5 * dt**3,
                        dt**2,
                    ],
                ],
                dtype=float,
            )
            * q
        )

        predicted_covariance = (
            transition
            @ filtered_covariances[
                epoch_index
            ]
            @ transition.T
            + process_covariance
        )

        try:
            smoother_gain = (
                filtered_covariances[
                    epoch_index
                ]
                @ transition.T
                @ np.linalg.inv(
                    predicted_covariance
                )
            )

        except np.linalg.LinAlgError:
            continue

        smoothed_states[
            epoch_index
        ] = (
            filtered_states[
                epoch_index
            ]
            + smoother_gain
            @ (
                smoothed_states[
                    epoch_index + 1
                ]
                - transition
                @ filtered_states[
                    epoch_index
                ]
            )
        )

        smoothed_covariances[
            epoch_index
        ] = (
            filtered_covariances[
                epoch_index
            ]
            + smoother_gain
            @ (
                smoothed_covariances[
                    epoch_index + 1
                ]
                - predicted_covariance
            )
            @ smoother_gain.T
        )

    change = (
        smoothed_states[:, 0]
    )

    rate = (
        smoothed_states[:, 1]
    )

    sigma_change = np.sqrt(
        np.clip(
            smoothed_covariances[
                :,
                0,
                0,
            ],
            0,
            None,
        )
    )

    sigma_rate = np.sqrt(
        np.clip(
            smoothed_covariances[
                :,
                1,
                1,
            ],
            0,
            None,
        )
    )

    return (
        change,
        rate,
        sigma_change,
        sigma_rate,
    )


# ============================================================
# TEMPORAL SEED HELPERS
# ============================================================

def _extract_true_intervals(
    boolean_series
):
    boolean_series = np.asarray(
        boolean_series,
        dtype=bool,
    )

    active = np.where(
        boolean_series
    )[0]

    if active.size == 0:
        return []

    intervals = []

    start = int(
        active[0]
    )

    previous = int(
        active[0]
    )

    for epoch in active[1:]:
        epoch = int(epoch)

        if epoch == previous + 1:
            previous = epoch

        else:
            intervals.append(
                (
                    start,
                    previous,
                )
            )

            start = epoch
            previous = epoch

    intervals.append(
        (
            start,
            previous,
        )
    )

    return intervals


def _find_formation_start(
    change,
    change_sigma,
    raw_start_epoch,
    direction,
    earliest_allowed_epoch,
    baseline_tolerance,
    baseline_sigma_factor,
    max_extension_epochs,
    min_extension_slope,
):
    raw_start_epoch = int(
        raw_start_epoch
    )

    earliest_allowed_epoch = int(
        earliest_allowed_epoch
    )

    if raw_start_epoch <= 0:
        return raw_start_epoch

    search_start = max(
        earliest_allowed_epoch,
        raw_start_epoch
        - int(
            max_extension_epochs
        ),
    )

    candidate_epochs = np.arange(
        search_start,
        raw_start_epoch + 1,
        dtype=int,
    )

    signal = change[
        candidate_epochs
    ]

    sigma = change_sigma[
        candidate_epochs
    ]

    baseline_threshold = np.maximum(
        float(
            baseline_tolerance
        ),
        float(
            baseline_sigma_factor
        )
        * sigma,
    )

    if direction == "positive":
        same_sign = (
            signal
            >= -float(
                baseline_tolerance
            )
        )

    elif direction == "negative":
        same_sign = (
            signal
            <= float(
                baseline_tolerance
            )
        )

    else:
        return raw_start_epoch

    near_baseline = (
        np.abs(signal)
        <= baseline_threshold
    )

    candidates = (
        candidate_epochs[
            same_sign
            & near_baseline
        ]
    )

    if candidates.size:
        extended_start = int(
            candidates[-1]
        )

    else:
        finite = np.isfinite(
            signal
        )

        if not np.any(finite):
            return raw_start_epoch

        finite_epochs = (
            candidate_epochs[
                finite
            ]
        )

        finite_signal = (
            signal[
                finite
            ]
        )

        extended_start = int(
            finite_epochs[
                np.argmin(
                    np.abs(
                        finite_signal
                    )
                )
            ]
        )

    if (
        extended_start
        < raw_start_epoch - 1
    ):
        slopes = np.diff(
            change[
                extended_start:
                raw_start_epoch + 1
            ]
        )

        slopes = slopes[
            np.isfinite(slopes)
        ]

        if slopes.size == 0:
            return raw_start_epoch

        if direction == "positive":
            consistent_fraction = (
                np.mean(
                    slopes
                    >= -float(
                        min_extension_slope
                    )
                )
            )

        else:
            consistent_fraction = (
                np.mean(
                    slopes
                    <= float(
                        min_extension_slope
                    )
                )
            )

        if (
            consistent_fraction
            < 0.5
        ):
            return raw_start_epoch

    return int(
        max(
            earliest_allowed_epoch,
            min(
                extended_start,
                raw_start_epoch,
            ),
        )
    )


# ============================================================
# KALMAN-BASED REGION GROWING
# ============================================================

class KalmanRegionGrowingAlgorithm(
    RegionGrowingAlgorithm
):
    """
    Kalman-enabled 4D-OBC algorithm.

    Kalman filtering changes seed detection and provides the
    Kalman-smoothed change signal used by inherited py4dgeo
    region growing.

    A common Kalman solution can be reused by KF-Mag,
    KF-Rate and their NMS variants through a validated cache.
    """

    def __init__(
        self,
        detection_mode="magnitude",
        kalman_cache_path=None,
        process_sigma=0.01,
        min_sigma_obs=0.005,
        z_threshold=1.96,
        min_seed_duration=10,
        min_seed_magnitude=0.05,
        seed_direction="both",
        extend_magnitude_to_formation_start=True,
        formation_baseline_tolerance=0.02,
        formation_baseline_sigma_factor=1.0,
        max_formation_extension_epochs=45,
        min_extension_slope=0.001,
        use_spatial_support=False,
        spatial_support_radius=1.5,
        min_neighbour_support=3,
        min_support_fraction=0.30,
        support_time_tolerance=5,
        neighbour_magnitude_ratio=0.5,
        use_nms=False,
        nms_radius=3.0,
        nms_min_support_count=3,
        nms_min_support_fraction=0.30,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )

        if detection_mode not in (
            "magnitude",
            "rate",
        ):
            raise ValueError(
                "detection_mode must be "
                "'magnitude' or 'rate'."
            )

        if seed_direction not in (
            "positive",
            "negative",
            "both",
        ):
            raise ValueError(
                "seed_direction must be "
                "'positive', 'negative', or 'both'."
            )

        self.detection_mode = (
            detection_mode
        )

        self.kalman_cache_path = (
            None
            if kalman_cache_path is None
            else Path(
                kalman_cache_path
            )
        )

        self.process_sigma = float(
            process_sigma
        )

        self.min_sigma_obs = float(
            min_sigma_obs
        )

        self.z_threshold = float(
            z_threshold
        )

        self.min_seed_duration = int(
            min_seed_duration
        )

        self.min_seed_magnitude = float(
            min_seed_magnitude
        )

        self.seed_direction = (
            seed_direction
        )

        self.extend_magnitude_to_formation_start = bool(
            extend_magnitude_to_formation_start
        )

        self.formation_baseline_tolerance = float(
            formation_baseline_tolerance
        )

        self.formation_baseline_sigma_factor = float(
            formation_baseline_sigma_factor
        )

        self.max_formation_extension_epochs = int(
            max_formation_extension_epochs
        )

        self.min_extension_slope = float(
            min_extension_slope
        )

        self.use_spatial_support = bool(
            use_spatial_support
        )

        self.spatial_support_radius = float(
            spatial_support_radius
        )

        self.min_neighbour_support = int(
            min_neighbour_support
        )

        self.min_support_fraction = float(
            min_support_fraction
        )

        self.support_time_tolerance = int(
            support_time_tolerance
        )

        self.neighbour_magnitude_ratio = float(
            neighbour_magnitude_ratio
        )

        self.use_nms = bool(
            use_nms
        )

        self.nms_radius = float(
            nms_radius
        )

        self.nms_min_support_count = int(
            nms_min_support_count
        )

        self.nms_min_support_fraction = float(
            nms_min_support_fraction
        )

        self.kalman_change = None
        self.kalman_rate = None
        self.kalman_sigma_change = None
        self.kalman_sigma_rate = None
        self.kalman_observation_sigma = None
        self.kalman_dt_days = None
        self.kalman_cache_key = None

        self.seed_table = None


    # --------------------------------------------------------
    # Inherited py4dgeo seed-candidate interface
    # --------------------------------------------------------

    def _selected_seed_candidates(
        self
    ):
        n_corepoints = (
            self.analysis.distances.shape[
                0
            ]
        )

        if self.seed_candidates is None:
            return np.arange(
                n_corepoints,
                dtype=int,
            )

        indices = np.asarray(
            list(
                self.seed_candidates
            ),
            dtype=int,
        )

        if indices.ndim != 1:
            raise Py4DGeoError(
                "seed_candidates must be "
                "one-dimensional."
            )

        if (
            np.any(indices < 0)
            or
            np.any(
                indices
                >= n_corepoints
            )
        ):
            raise Py4DGeoError(
                "seed_candidates contains "
                "an invalid corepoint index."
            )

        return indices


    # --------------------------------------------------------
    # Kalman filtering / cache reuse
    # --------------------------------------------------------

    def _run_kalman_filter(
        self
    ):
        distances = np.asarray(
            self.analysis.distances,
            dtype=float,
        )

        if distances.ndim != 2:
            raise Py4DGeoError(
                "analysis.distances must "
                "have shape "
                "(n_corepoints, n_epochs)."
            )

        uncertainties = (
            self.analysis.uncertainties
        )

        if uncertainties is None:
            raise Py4DGeoError(
                "SpatiotemporalAnalysis "
                "contains no M3C2 "
                "uncertainties."
            )

        if (
            uncertainties.dtype.names
            is None
            or
            "lodetection"
            not in uncertainties.dtype.names
        ):
            raise Py4DGeoError(
                "analysis.uncertainties "
                "does not contain "
                "'lodetection'."
            )

        lodetection = np.asarray(
            uncertainties[
                "lodetection"
            ],
            dtype=float,
        )

        dt_series = (
            _timedeltas_to_days(
                self.analysis.timedeltas
            )
        )

        if (
            len(dt_series)
            != distances.shape[1]
        ):
            raise Py4DGeoError(
                "Number of analysis "
                "timedeltas does not match "
                "the number of distance "
                "epochs."
            )

        # ----------------------------------------------------
        # Cache fingerprint
        # ----------------------------------------------------

        cache_key = (
            _build_kalman_cache_key(
                distances=distances,
                lodetection=lodetection,
                dt_series=dt_series,
                process_sigma=(
                    self.process_sigma
                ),
                min_sigma_obs=(
                    self.min_sigma_obs
                ),
            )
        )

        self.kalman_cache_key = (
            cache_key
        )

        # ----------------------------------------------------
        # 1. Reuse same-session in-memory cache
        # ----------------------------------------------------

        if (
            self.kalman_cache_path
            is not None
            and
            cache_key
            in _MEMORY_KALMAN_CACHE
        ):
            cached = (
                _MEMORY_KALMAN_CACHE[
                    cache_key
                ]
            )

            self.kalman_change = (
                cached["change"]
            )

            self.kalman_rate = (
                cached["rate"]
            )

            self.kalman_sigma_change = (
                cached[
                    "sigma_change"
                ]
            )

            self.kalman_sigma_rate = (
                cached[
                    "sigma_rate"
                ]
            )

            self.kalman_dt_days = (
                cached["dt_days"]
            )

            self.kalman_observation_sigma = (
                np.maximum(
                    lodetection / 1.96,
                    self.min_sigma_obs,
                )
            )

            try:
                self.kalman_observation_sigma = (
                    np.broadcast_to(
                        self.kalman_observation_sigma,
                        distances.shape,
                    )
                )

            except ValueError as exc:
                raise Py4DGeoError(
                    "M3C2 LoDetection values "
                    "cannot be broadcast to "
                    "analysis.distances."
                ) from exc

            print(
                f"{self.name}: reusing "
                "Kalman states from "
                "memory cache."
            )

            self.analysis.smoothed_distances = (
                self.kalman_change
            )

            return

        # ----------------------------------------------------
        # 2. Reuse validated persistent disk cache
        # ----------------------------------------------------

        if (
            self.kalman_cache_path
            is not None
        ):
            cached = (
                _load_kalman_cache(
                    cache_root=(
                        self.kalman_cache_path
                    ),
                    cache_key=(
                        cache_key
                    ),
                    expected_shape=(
                        distances.shape
                    ),
                )
            )

            if cached is not None:
                self.kalman_change = (
                    cached["change"]
                )

                self.kalman_rate = (
                    cached["rate"]
                )

                self.kalman_sigma_change = (
                    cached[
                        "sigma_change"
                    ]
                )

                self.kalman_sigma_rate = (
                    cached[
                        "sigma_rate"
                    ]
                )

                self.kalman_dt_days = (
                    cached["dt_days"]
                )

                self.kalman_observation_sigma = (
                    np.maximum(
                        lodetection / 1.96,
                        self.min_sigma_obs,
                    )
                )

                try:
                    self.kalman_observation_sigma = (
                        np.broadcast_to(
                            self.kalman_observation_sigma,
                            distances.shape,
                        )
                    )

                except ValueError as exc:
                    raise Py4DGeoError(
                        "M3C2 LoDetection values "
                        "cannot be broadcast to "
                        "analysis.distances."
                    ) from exc

                _MEMORY_KALMAN_CACHE[
                    cache_key
                ] = cached

                print(
                    f"{self.name}: reusing "
                    "validated Kalman states "
                    "from disk cache."
                )

                self.analysis.smoothed_distances = (
                    self.kalman_change
                )

                return

        # ----------------------------------------------------
        # 3. No valid cache -> calculate Kalman states
        # ----------------------------------------------------

        observation_sigma = np.maximum(
            lodetection / 1.96,
            self.min_sigma_obs,
        )

        try:
            observation_sigma = (
                np.broadcast_to(
                    observation_sigma,
                    distances.shape,
                )
                .copy()
            )

        except ValueError as exc:
            raise Py4DGeoError(
                "M3C2 LoDetection values "
                "cannot be broadcast to "
                "analysis.distances."
            ) from exc

        shape = (
            distances.shape
        )

        change = np.full(
            shape,
            np.nan,
            dtype=float,
        )

        rate = np.full(
            shape,
            np.nan,
            dtype=float,
        )

        sigma_change = np.full(
            shape,
            np.nan,
            dtype=float,
        )

        sigma_rate = np.full(
            shape,
            np.nan,
            dtype=float,
        )

        # IMPORTANT:
        # Kalman filtering is deliberately performed for
        # ALL corepoints.
        #
        # seed_candidates limits only which corepoints may
        # GENERATE seeds. Region growing and spatial support
        # can access neighbouring corepoints outside that set.
        for cp_idx in range(
            distances.shape[0]
        ):
            (
                change[cp_idx],
                rate[cp_idx],
                sigma_change[cp_idx],
                sigma_rate[cp_idx],
            ) = _kalman_smooth_series(
                observations=(
                    distances[
                        cp_idx
                    ]
                ),
                observation_sigma=(
                    observation_sigma[
                        cp_idx
                    ]
                ),
                dt_series=(
                    dt_series
                ),
                process_sigma=(
                    self.process_sigma
                ),
            )

        self.kalman_change = (
            change
        )

        self.kalman_rate = (
            rate
        )

        self.kalman_sigma_change = (
            sigma_change
        )

        self.kalman_sigma_rate = (
            sigma_rate
        )

        self.kalman_observation_sigma = (
            observation_sigma
        )

        self.kalman_dt_days = (
            dt_series
        )

        # ----------------------------------------------------
        # 4. Store new solution in memory and on disk
        # ----------------------------------------------------

        if (
            self.kalman_cache_path
            is not None
        ):
            cached = {
                "change":
                    self.kalman_change,

                "rate":
                    self.kalman_rate,

                "sigma_change":
                    self.kalman_sigma_change,

                "sigma_rate":
                    self.kalman_sigma_rate,

                "dt_days":
                    self.kalman_dt_days,
            }

            _MEMORY_KALMAN_CACHE[
                cache_key
            ] = cached

            _write_kalman_cache(
                cache_root=(
                    self.kalman_cache_path
                ),
                cache_key=(
                    cache_key
                ),
                change=(
                    self.kalman_change
                ),
                rate=(
                    self.kalman_rate
                ),
                sigma_change=(
                    self.kalman_sigma_change
                ),
                sigma_rate=(
                    self.kalman_sigma_rate
                ),
                dt_days=(
                    self.kalman_dt_days
                ),
                process_sigma=(
                    self.process_sigma
                ),
                min_sigma_obs=(
                    self.min_sigma_obs
                ),
            )

            print(
                f"{self.name}: Kalman "
                "states stored in validated "
                "cache."
            )

        # Region growing must use Kalman-smoothed CHANGE.
        # Rate remains the seed-detection criterion for
        # KF-Rate.
        self.analysis.smoothed_distances = (
            self.kalman_change
        )


    # --------------------------------------------------------
    # Kalman seed detection
    # --------------------------------------------------------

    def _detect_seed_records(
        self
    ):
        timestamps = [
            (
                self.analysis
                .reference_epoch
                .timestamp
                + td
            )
            for td
            in self.analysis.timedeltas
        ]

        records = []

        selected_indices = (
            self._selected_seed_candidates()
        )

        for cp_idx in selected_indices:
            cp_idx = int(
                cp_idx
            )

            change = (
                self.kalman_change[
                    cp_idx
                ]
            )

            rate = (
                self.kalman_rate[
                    cp_idx
                ]
            )

            sigma_change = (
                self.kalman_sigma_change[
                    cp_idx
                ]
            )

            sigma_rate = (
                self.kalman_sigma_rate[
                    cp_idx
                ]
            )

            if np.all(
                np.isnan(change)
            ):
                continue

            if (
                self.detection_mode
                == "magnitude"
            ):
                signal = change
                signal_sigma = (
                    sigma_change
                )

            else:
                signal = rate
                signal_sigma = (
                    sigma_rate
                )

            lod = (
                self.z_threshold
                * signal_sigma
            )

            directional_masks = []

            if (
                self.seed_direction
                in (
                    "positive",
                    "both",
                )
            ):
                directional_masks.append(
                    (
                        "positive",
                        signal > lod,
                    )
                )

            if (
                self.seed_direction
                in (
                    "negative",
                    "both",
                )
            ):
                directional_masks.append(
                    (
                        "negative",
                        signal < -lod,
                    )
                )

            for (
                direction,
                activity_mask,
            ) in directional_masks:

                raw_intervals = (
                    _extract_true_intervals(
                        activity_mask
                    )
                )

                for (
                    interval_id,
                    (
                        raw_start,
                        raw_end,
                    ),
                ) in enumerate(
                    raw_intervals
                ):
                    raw_start = int(
                        raw_start
                    )

                    raw_end = int(
                        raw_end
                    )

                    if interval_id == 0:
                        earliest_allowed = 0

                    else:
                        earliest_allowed = (
                            int(
                                raw_intervals[
                                    interval_id - 1
                                ][1]
                            )
                            + 1
                        )

                    start_epoch = (
                        raw_start
                    )

                    if (
                        self.detection_mode
                        == "magnitude"
                        and
                        self.extend_magnitude_to_formation_start
                    ):
                        start_epoch = (
                            _find_formation_start(
                                change=(
                                    change
                                ),
                                change_sigma=(
                                    sigma_change
                                ),
                                raw_start_epoch=(
                                    raw_start
                                ),
                                direction=(
                                    direction
                                ),
                                earliest_allowed_epoch=(
                                    earliest_allowed
                                ),
                                baseline_tolerance=(
                                    self
                                    .formation_baseline_tolerance
                                ),
                                baseline_sigma_factor=(
                                    self
                                    .formation_baseline_sigma_factor
                                ),
                                max_extension_epochs=(
                                    self
                                    .max_formation_extension_epochs
                                ),
                                min_extension_slope=(
                                    self
                                    .min_extension_slope
                                ),
                            )
                        )

                    end_epoch = (
                        raw_end
                    )

                    duration = (
                        end_epoch
                        - start_epoch
                        + 1
                    )

                    if (
                        duration
                        < self.min_seed_duration
                    ):
                        continue

                    interval_change = (
                        change[
                            start_epoch:
                            end_epoch + 1
                        ]
                    )

                    if np.all(
                        np.isnan(
                            interval_change
                        )
                    ):
                        continue

                    if (
                        direction
                        == "positive"
                    ):
                        event_magnitude = (
                            float(
                                np.nanmax(
                                    interval_change
                                )
                            )
                        )

                    else:
                        event_magnitude = (
                            float(
                                abs(
                                    np.nanmin(
                                        interval_change
                                    )
                                )
                            )
                        )

                    if (
                        event_magnitude
                        < self.min_seed_magnitude
                    ):
                        continue

                    net_change = float(
                        change[
                            end_epoch
                        ]
                        - change[
                            start_epoch
                        ]
                    )

                    records.append(
                        {
                            "corepoint_index_python":
                                cp_idx,

                            "start_epoch":
                                int(
                                    start_epoch
                                ),

                            "end_epoch":
                                int(
                                    end_epoch
                                ),

                            "duration_epochs":
                                int(
                                    duration
                                ),

                            "raw_significant_start_epoch":
                                raw_start,

                            "raw_significant_end_epoch":
                                raw_end,

                            "direction":
                                direction,

                            "event_magnitude":
                                event_magnitude,

                            "net_change":
                                net_change,

                            "abs_net_change":
                                abs(
                                    net_change
                                ),

                            "start_time":
                                timestamps[
                                    start_epoch
                                ],

                            "end_time":
                                timestamps[
                                    end_epoch
                                ],
                        }
                    )

        seed_table = (
            pd.DataFrame(
                records
            )
        )

        if len(seed_table):
            seed_table = (
                seed_table
                .sort_values(
                    [
                        "event_magnitude",
                        "duration_epochs",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

        return seed_table


    # --------------------------------------------------------
    # Spatial support validation
    # --------------------------------------------------------

    def _apply_spatial_support(
        self,
        seed_table,
    ):
        """
        Validate Kalman-derived seeds using spatial neighbours.

        support_time_tolerance expands only the validation
        window. It does NOT change the seed interval passed
        to RegionGrowingSeed.
        """

        if (
            seed_table is None
            or
            len(seed_table) == 0
        ):
            return seed_table

        corepoints = np.asarray(
            self.analysis
            .corepoints
            .cloud,
            dtype=float,
        )

        tree = cKDTree(
            corepoints[:, :2]
        )

        n_epochs = (
            self.kalman_change
            .shape[1]
        )

        retained = []

        for _, seed in (
            seed_table.iterrows()
        ):
            cp_idx = int(
                seed[
                    "corepoint_index_python"
                ]
            )

            start_epoch = int(
                seed[
                    "start_epoch"
                ]
            )

            end_epoch = int(
                seed[
                    "end_epoch"
                ]
            )

            direction = (
                seed[
                    "direction"
                ]
            )

            event_magnitude = float(
                seed[
                    "event_magnitude"
                ]
            )

            neighbours = (
                tree.query_ball_point(
                    corepoints[
                        cp_idx,
                        :2,
                    ],
                    r=(
                        self
                        .spatial_support_radius
                    ),
                )
            )

            neighbours = [
                int(idx)
                for idx
                in neighbours
                if int(idx)
                != cp_idx
            ]

            if (
                len(neighbours)
                == 0
            ):
                continue

            validation_start_epoch = max(
                0,
                start_epoch
                - self
                .support_time_tolerance,
            )

            validation_end_epoch = min(
                n_epochs - 1,
                end_epoch
                + self
                .support_time_tolerance,
            )

            neighbour_support_count = 0

            for neighbour in neighbours:
                neighbour_change = (
                    self.kalman_change[
                        neighbour,
                        validation_start_epoch:
                        validation_end_epoch + 1,
                    ]
                )

                if np.all(
                    np.isnan(
                        neighbour_change
                    )
                ):
                    continue

                if (
                    direction
                    == "positive"
                ):
                    neighbour_value = float(
                        np.nanmax(
                            neighbour_change
                        )
                    )

                    supports_event = (
                        neighbour_value > 0
                        and
                        neighbour_value
                        >= (
                            self
                            .neighbour_magnitude_ratio
                            * event_magnitude
                        )
                    )

                elif (
                    direction
                    == "negative"
                ):
                    neighbour_min = float(
                        np.nanmin(
                            neighbour_change
                        )
                    )

                    supports_event = (
                        neighbour_min < 0
                        and
                        abs(
                            neighbour_min
                        )
                        >= (
                            self
                            .neighbour_magnitude_ratio
                            * event_magnitude
                        )
                    )

                else:
                    supports_event = False

                if supports_event:
                    neighbour_support_count += 1

            support_fraction = (
                neighbour_support_count
                / len(neighbours)
            )

            if (
                neighbour_support_count
                < self.min_neighbour_support
            ):
                continue

            if (
                support_fraction
                < self.min_support_fraction
            ):
                continue

            record = (
                seed.to_dict()
            )

            # Original seed interval remains unchanged.
            record[
                "support_start_epoch"
            ] = start_epoch

            record[
                "support_end_epoch"
            ] = end_epoch

            # Validation tolerance is stored separately.
            record[
                "validation_start_epoch"
            ] = (
                validation_start_epoch
            )

            record[
                "validation_end_epoch"
            ] = (
                validation_end_epoch
            )

            record[
                "validation_duration_epochs"
            ] = (
                validation_end_epoch
                - validation_start_epoch
                + 1
            )

            record[
                "n_neighbours"
            ] = int(
                len(neighbours)
            )

            record[
                "neighbour_support_count"
            ] = int(
                neighbour_support_count
            )

            record[
                "support_fraction"
            ] = float(
                support_fraction
            )

            record[
                "spatial_support_used"
            ] = True

            record[
                "spatial_support_radius"
            ] = float(
                self
                .spatial_support_radius
            )

            record[
                "support_time_tolerance"
            ] = int(
                self
                .support_time_tolerance
            )

            record[
                "neighbour_magnitude_ratio"
            ] = float(
                self
                .neighbour_magnitude_ratio
            )

            retained.append(
                record
            )

        validated_table = (
            pd.DataFrame(
                retained
            )
        )

        if (
            len(
                validated_table
            )
            > 0
        ):
            validated_table = (
                validated_table
                .sort_values(
                    [
                        "event_magnitude",
                        "support_fraction",
                        "duration_epochs",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            validated_table[
                "spatial_support_rank"
            ] = np.arange(
                1,
                len(
                    validated_table
                ) + 1,
                dtype=int,
            )

        return validated_table


    # --------------------------------------------------------
    # Non-maximum suppression
    # --------------------------------------------------------

    def _apply_nms(
        self,
        seed_table,
    ):
        if (
            seed_table is None
            or
            len(seed_table) == 0
        ):
            return seed_table

        table = (
            seed_table.copy()
        )

        if (
            "support_start_epoch"
            not in table.columns
        ):
            table[
                "support_start_epoch"
            ] = table[
                "start_epoch"
            ]

        if (
            "support_end_epoch"
            not in table.columns
        ):
            table[
                "support_end_epoch"
            ] = table[
                "end_epoch"
            ]

        table[
            "support_duration_epochs"
        ] = (
            table[
                "support_end_epoch"
            ]
            - table[
                "support_start_epoch"
            ]
            + 1
        )

        if (
            "support_fraction"
            not in table.columns
        ):
            table[
                "support_fraction"
            ] = 0.0

        table = (
            table
            .sort_values(
                [
                    "abs_net_change",
                    "support_fraction",
                    "support_duration_epochs",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        corepoints = np.asarray(
            self.analysis
            .corepoints
            .cloud,
            dtype=float,
        )

        tree = cKDTree(
            corepoints[:, :2]
        )

        suppressed = set()
        retained = []

        for (
            row_index,
            seed,
        ) in table.iterrows():

            if (
                row_index
                in suppressed
            ):
                continue

            cp_idx = int(
                seed[
                    "corepoint_index_python"
                ]
            )

            start_epoch = int(
                seed[
                    "support_start_epoch"
                ]
            )

            end_epoch = int(
                seed[
                    "support_end_epoch"
                ]
            )

            direction = (
                seed[
                    "direction"
                ]
            )

            neighbours = (
                tree.query_ball_point(
                    corepoints[
                        cp_idx,
                        :2,
                    ],
                    r=(
                        self.nms_radius
                    ),
                )
            )

            neighbours = [
                int(idx)
                for idx
                in neighbours
                if int(idx)
                != cp_idx
            ]

            neighbour_rows = (
                table[
                    table[
                        "corepoint_index_python"
                    ].isin(
                        neighbours
                    )
                ]
            )

            support_count = 0

            for _, other in (
                neighbour_rows
                .iterrows()
            ):
                if (
                    other[
                        "direction"
                    ]
                    != direction
                ):
                    continue

                overlap = max(
                    0,
                    min(
                        end_epoch,
                        int(
                            other[
                                "support_end_epoch"
                            ]
                        ),
                    )
                    - max(
                        start_epoch,
                        int(
                            other[
                                "support_start_epoch"
                            ]
                        ),
                    )
                    + 1,
                )

                if overlap > 0:
                    support_count += 1

            support_fraction = (
                support_count
                / max(
                    len(
                        neighbours
                    ),
                    1,
                )
            )

            if (
                support_count
                < self
                .nms_min_support_count
            ):
                continue

            if (
                support_fraction
                < self
                .nms_min_support_fraction
            ):
                continue

            record = (
                seed.to_dict()
            )

            record[
                "nms_support_count"
            ] = int(
                support_count
            )

            record[
                "nms_support_fraction"
            ] = float(
                support_fraction
            )

            retained.append(
                record
            )

            for (
                other_index,
                other,
            ) in table.iterrows():

                if (
                    other_index
                    <= row_index
                    or
                    other_index
                    in suppressed
                ):
                    continue

                other_cp = int(
                    other[
                        "corepoint_index_python"
                    ]
                )

                if (
                    other_cp
                    not in neighbours
                ):
                    continue

                if (
                    other[
                        "direction"
                    ]
                    != direction
                ):
                    continue

                overlap = max(
                    0,
                    min(
                        end_epoch,
                        int(
                            other[
                                "support_end_epoch"
                            ]
                        ),
                    )
                    - max(
                        start_epoch,
                        int(
                            other[
                                "support_start_epoch"
                            ]
                        ),
                    )
                    + 1,
                )

                if overlap > 0:
                    suppressed.add(
                        other_index
                    )

        nms_table = (
            pd.DataFrame(
                retained
            )
        )

        if len(nms_table):
            nms_table = (
                nms_table
                .sort_values(
                    [
                        "abs_net_change",
                        "nms_support_fraction",
                        "support_duration_epochs",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            nms_table[
                "nms_rank"
            ] = np.arange(
                1,
                len(
                    nms_table
                ) + 1,
                dtype=int,
            )

        return nms_table


    # --------------------------------------------------------
    # py4dgeo seed interface
    # --------------------------------------------------------

    def find_seedpoints(
        self
    ):
        self._run_kalman_filter()

        seed_table = (
            self._detect_seed_records()
        )

        if (
            self.use_spatial_support
            and
            len(seed_table)
        ):
            seed_table = (
                self._apply_spatial_support(
                    seed_table
                )
            )

        if (
            self.use_nms
            and
            len(seed_table)
        ):
            seed_table = (
                self._apply_nms(
                    seed_table
                )
            )

        self.seed_table = (
            seed_table.copy()
        )

        seeds = []

        for _, row in (
            seed_table.iterrows()
        ):
            if (
                self.use_spatial_support
                and
                "support_start_epoch"
                in row.index
            ):
                start_epoch = int(
                    row[
                        "support_start_epoch"
                    ]
                )

                end_epoch = int(
                    row[
                        "support_end_epoch"
                    ]
                )

            else:
                start_epoch = int(
                    row[
                        "start_epoch"
                    ]
                )

                end_epoch = int(
                    row[
                        "end_epoch"
                    ]
                )

            seeds.append(
                RegionGrowingSeed(
                    int(
                        row[
                            "corepoint_index_python"
                        ]
                    ),
                    start_epoch,
                    end_epoch,
                )
            )

        print(
            f"{self.name}: "
            f"{len(seeds)} seeds "
            f"from "
            f"{len(self._selected_seed_candidates())} "
            f"seed-candidate corepoints."
        )

        return seeds


    # --------------------------------------------------------
    # Protect original analysis smoothing
    # --------------------------------------------------------

    def run(
        self,
        analysis,
        force=False,
    ):
        """
        Temporarily replace analysis.smoothed_distances with
        the Kalman-smoothed change signal while inherited
        py4dgeo region growing is running.

        Restore the pre-existing temporal smoothing afterward.
        """

        previous_smoothed_distances = (
            None
            if (
                analysis
                .smoothed_distances
                is None
            )
            else np.asarray(
                analysis
                .smoothed_distances,
                dtype=float,
            ).copy()
        )

        try:
            return super().run(
                analysis,
                force=force,
            )

        finally:
            analysis.smoothed_distances = (
                previous_smoothed_distances
            )


    # --------------------------------------------------------
    # Method name
    # --------------------------------------------------------

    @property
    def name(
        self
    ):
        method = (
            "KF-Mag"
            if (
                self.detection_mode
                == "magnitude"
            )
            else
            "KF-Rate"
        )

        if self.use_nms:
            method += "-NMS"

        return method