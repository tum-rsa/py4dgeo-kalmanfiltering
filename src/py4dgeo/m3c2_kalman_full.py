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

# ============================================================
# FULL-DOMAIN CHECKPOINTED KALMAN + RTS PROCESSING (STAGE 2)
# ============================================================

def run_full_domain_kalman_rts(
    analysis,
    output_dir,
    prefix="full106_kalman_rts_v1",
    process_sigma=0.01,
    min_sigma_obs=0.005,
    chunk_size=256,
    resume=True,
    verbose=True,
):
    """Compute/cache the original module's Kalman+RTS states for every corepoint.

    Inputs: analysis.distances or analysis.distances_for_compute, M3C2
    uncertainties['lodetection'], and analysis.timedeltas. Outputs are four
    (corepoint, epoch) float32 NPY arrays. Progress resumes at completed
    chunk boundaries. A new prefix is required for different inputs/settings.

    LoDetection / 1.96 follows the source module's approximation and must be
    checked against the actual definition of LoDetection in the dataset.
    """
    import os
    from numpy.lib.format import open_memmap

    if hasattr(analysis, "distances_for_compute"):
        distances = np.asarray(analysis.distances_for_compute)
    else:
        distances = np.asarray(analysis.distances)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("Expected (corepoints, epochs) M3C2 distances")

    uncertainties = analysis.uncertainties
    if uncertainties is None or uncertainties.dtype.names is None or "lodetection" not in uncertainties.dtype.names:
        raise ValueError("analysis.uncertainties['lodetection'] is required")
    lod = np.asarray(uncertainties["lodetection"], dtype=float)
    try:
        obs_sigma = np.broadcast_to(np.maximum(lod / 1.96, min_sigma_obs), distances.shape)
    except ValueError as exc:
        raise ValueError(f"LoDetection shape {lod.shape} cannot broadcast to {distances.shape}") from exc

    dt = _timedeltas_to_days(analysis.timedeltas)
    if len(dt) != distances.shape[1]:
        raise ValueError("Number of timedeltas does not match epoch count")
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("Invalid temporal spacing")
    if process_sigma < 0 or min_sigma_obs <= 0 or chunk_size < 1:
        raise ValueError("Invalid Kalman parameters")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ("change", "rate", "sigma_change", "sigma_rate")
    paths = {name: out / f"{prefix}_{name}.npy" for name in names}
    state_path = out / f"{prefix}_progress.json"
    meta_path = out / f"{prefix}_metadata.json"
    config = {
        "version": "original-m3c2-kalman-rts-v1-checkpointed",
        "shape": list(distances.shape),
        "process_sigma": float(process_sigma),
        "min_sigma_obs": float(min_sigma_obs),
        "chunk_size": int(chunk_size),
        "dt_days": dt.tolist(),
        "source": "analysis.distances_for_compute" if hasattr(analysis, "distances_for_compute") else "analysis.distances",
    }
    # A full data fingerprint is deliberately not computed on each restart.
    # Keep a unique prefix for each dataset and parameter combination.
    exists = state_path.exists()
    if exists and not resume:
        raise FileExistsError(f"Existing progress at {state_path}; choose new prefix")
    if exists:
        with state_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
        if progress.get("config") != config:
            raise ValueError("Existing checkpoint configuration differs; choose a new prefix")
        next_index = int(progress["next_index"])
        if not all(p.exists() for p in paths.values()):
            raise FileNotFoundError("Incomplete Kalman outputs: cannot safely resume")
        arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
        if any(arr.shape != distances.shape for arr in arrays.values()):
            raise ValueError("Saved Kalman array shapes do not match input")
    else:
        if any(p.exists() for p in [*paths.values(), meta_path]):
            raise FileExistsError("Output files already exist; choose a new prefix")
        arrays = {name: open_memmap(path, mode="w+", dtype="float32", shape=distances.shape) for name, path in paths.items()}
        next_index = 0
        _atomic_json_write(state_path, {"config": config, "next_index": 0})

    n_corepoints = distances.shape[0]
    for start in range(next_index, n_corepoints, chunk_size):
        end = min(start + chunk_size, n_corepoints)
        for cp in range(start, end):
            change, rate, sigma_change, sigma_rate = _kalman_smooth_series(
                observations=distances[cp],
                observation_sigma=obs_sigma[cp],
                dt_series=dt,
                process_sigma=process_sigma,
            )
            arrays["change"][cp] = change
            arrays["rate"][cp] = rate
            arrays["sigma_change"][cp] = sigma_change
            arrays["sigma_rate"][cp] = sigma_rate
        for arr in arrays.values():
            arr.flush()
        _atomic_json_write(state_path, {"config": config, "next_index": end})
        if verbose:
            print(f"Kalman+RTS: {end:,}/{n_corepoints:,} corepoints ({100*end/n_corepoints:.1f}%)", flush=True)

    _atomic_json_write(meta_path, {"config": config, "complete": True, "paths": {k: str(v) for k, v in paths.items()}})
    return {**paths, "progress": state_path, "metadata": meta_path}


def _atomic_json_write(path, payload):
    import os
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

# ============================================================
# STAGE 3 — FULL-DOMAIN KF-MAG / KF-RATE SEED DETECTION
# ============================================================

def detect_full_domain_kalman_seeds(
    output_dir,
    kalman_prefix="full106_kalman_rts_v1",
    seed_prefix="full106_kalman_seeds_v1",
    z_threshold=1.96,
    min_seed_duration=10,
    min_seed_magnitude=0.05,
    seed_direction="both",
    extend_magnitude_to_formation_start=True,
    formation_baseline_tolerance=0.02,
    formation_baseline_sigma_factor=1.0,
    max_formation_extension_epochs=45,
    min_extension_slope=0.001,
    chunk_size=2048,
    resume=True,
    verbose=True,
):
    """Detect KF-Mag and KF-Rate seeds from previously saved RTS arrays.

    Follows the original _detect_seed_records() decision rules. No Kalman
    recomputation, no spatial-support/NMS filtering, and no region growing.
    A seed interval is inclusive and uses zero-based epoch indices.
    Rate units follow Stage 2: metres/day when actual timedeltas are available.
    """
    import csv
    import os

    out = Path(output_dir)
    modes = ("magnitude", "rate")
    if seed_direction not in ("both", "positive", "negative"):
        raise ValueError("seed_direction must be both, positive or negative")
    if z_threshold <= 0 or min_seed_duration < 1 or min_seed_magnitude < 0 or chunk_size < 1:
        raise ValueError("Invalid seed-detection parameters")

    paths = {k: out / f"{kalman_prefix}_{k}.npy" for k in
             ("change", "rate", "sigma_change", "sigma_rate")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 2 result: {path}")
    metadata_path = out / f"{kalman_prefix}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as f:
        kalman_metadata = json.load(f)
    if not kalman_metadata.get("complete", False):
        raise ValueError("Stage 2 metadata does not confirm completion")

    arrays = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    shape = arrays["change"].shape
    if len(shape) != 2 or any(a.shape != shape for a in arrays.values()):
        raise ValueError("Stage 2 Kalman array shapes are inconsistent")
    n_cp, n_epochs = shape

    config = dict(
        version="kf-seeds-v1", kalman_prefix=kalman_prefix,
        kalman_config=kalman_metadata.get("config"),
        shape=list(shape), z_threshold=float(z_threshold),
        min_seed_duration=int(min_seed_duration),
        min_seed_magnitude=float(min_seed_magnitude),
        seed_direction=seed_direction,
        extend_magnitude_to_formation_start=bool(extend_magnitude_to_formation_start),
        formation_baseline_tolerance=float(formation_baseline_tolerance),
        formation_baseline_sigma_factor=float(formation_baseline_sigma_factor),
        max_formation_extension_epochs=int(max_formation_extension_epochs),
        min_extension_slope=float(min_extension_slope),
        chunk_size=int(chunk_size),
    )
    columns = ["corepoint_index_python", "start_epoch", "end_epoch",
               "duration_epochs", "raw_significant_start_epoch",
               "raw_significant_end_epoch", "direction", "event_magnitude",
               "net_change", "abs_net_change"]
    progress_path = out / f"{seed_prefix}_progress.json"
    csv_paths = {mode: out / f"{seed_prefix}_{mode}_unsorted.csv" for mode in modes}
    final_paths = {mode: out / f"{seed_prefix}_{mode}.csv" for mode in modes}
    summary_path = out / f"{seed_prefix}_summary.json"

    if progress_path.exists():
        if not resume:
            raise FileExistsError(f"Progress exists: {progress_path}")
        with progress_path.open(encoding="utf-8") as f:
            state = json.load(f)
        if state.get("config") != config:
            raise ValueError("Seed configuration changed: use a different seed_prefix")
        next_cp = int(state["next_cp"])
        # Checkpoint CSV byte sizes so rows written after an interrupted
        # checkpoint are removed before resuming.
        for mode in modes:
            if not csv_paths[mode].exists():
                raise FileNotFoundError(csv_paths[mode])
            with csv_paths[mode].open("rb+") as f:
                f.truncate(int(state["sizes"][mode]))
    else:
        if any(p.exists() for p in [*csv_paths.values(), *final_paths.values(), summary_path]):
            raise FileExistsError("Seed outputs already exist: use another seed_prefix")
        for mode in modes:
            with csv_paths[mode].open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columns)
        next_cp = 0
        state = {"config": config, "next_cp": 0,
                 "sizes": {mode: csv_paths[mode].stat().st_size for mode in modes}}
        _atomic_json_write(progress_path, state)

    if next_cp < 0 or next_cp > n_cp:
        raise ValueError("Invalid checkpoint position")
    for start in range(next_cp, n_cp, chunk_size):
        end = min(start + chunk_size, n_cp)
        records = {mode: [] for mode in modes}
        for cp in range(start, end):
            change = arrays["change"][cp]
            if not np.isfinite(change).any():
                continue
            sigma_change = arrays["sigma_change"][cp]
            for mode in modes:
                signal = change if mode == "magnitude" else arrays["rate"][cp]
                sigma = sigma_change if mode == "magnitude" else arrays["sigma_rate"][cp]
                significance = z_threshold * sigma
                directions = []
                if seed_direction in ("positive", "both"):
                    directions.append(("positive", signal > significance))
                if seed_direction in ("negative", "both"):
                    directions.append(("negative", signal < -significance))
                for direction, active in directions:
                    intervals = _extract_true_intervals(active)
                    for interval_id, (raw_start, raw_end) in enumerate(intervals):
                        earliest = 0 if interval_id == 0 else intervals[interval_id - 1][1] + 1
                        start_epoch = int(raw_start)
                        if mode == "magnitude" and extend_magnitude_to_formation_start:
                            start_epoch = _find_formation_start(
                                change=change, change_sigma=sigma_change,
                                raw_start_epoch=raw_start, direction=direction,
                                earliest_allowed_epoch=earliest,
                                baseline_tolerance=formation_baseline_tolerance,
                                baseline_sigma_factor=formation_baseline_sigma_factor,
                                max_extension_epochs=max_formation_extension_epochs,
                                min_extension_slope=min_extension_slope)
                        duration = int(raw_end - start_epoch + 1)
                        if duration < min_seed_duration:
                            continue
                        segment = change[start_epoch:raw_end + 1]
                        if not np.isfinite(segment).any():
                            continue
                        magnitude = (float(np.nanmax(segment)) if direction == "positive"
                                     else float(abs(np.nanmin(segment))))
                        if not np.isfinite(magnitude) or magnitude < min_seed_magnitude:
                            continue
                        net_change = float(change[raw_end] - change[start_epoch])
                        records[mode].append((cp, start_epoch, raw_end, duration,
                                              raw_start, raw_end, direction, magnitude,
                                              net_change, abs(net_change)))
        for mode in modes:
            with csv_paths[mode].open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(records[mode])
                f.flush()
                os.fsync(f.fileno())
        state = {"config": config, "next_cp": end,
                 "sizes": {mode: csv_paths[mode].stat().st_size for mode in modes}}
        _atomic_json_write(progress_path, state)
        if verbose:
            print(f"KF seed detection: {end:,}/{n_cp:,} corepoints ({100*end/n_cp:.1f}%)", flush=True)

    counts = {}
    for mode in modes:
        table = pd.read_csv(csv_paths[mode])
        if not table.empty:
            table = table.sort_values(["event_magnitude", "duration_epochs"],
                                      ascending=[False, False], kind="stable").reset_index(drop=True)
        table.insert(0, "seed_rank", np.arange(1, len(table) + 1, dtype=int))
        table.to_csv(final_paths[mode], index=False)
        counts[mode] = len(table)
    summary = {"config": config, "n_corepoints": n_cp, "n_epochs": n_epochs,
               "counts": counts, "files": {k: str(v) for k, v in final_paths.items()},
               "complete": True,
               "note": "Raw KF seed candidates; no spatial support, NMS or segmentation applied"}
    _atomic_json_write(summary_path, summary)
    if verbose:
        print("Stage 3 complete — KF-Mag:", counts["magnitude"],
              "KF-Rate:", counts["rate"])
    return {**final_paths, "summary": summary_path, "counts": counts}



# ============================================================
# STAGE 4 — KF-GUIDED SIGNED DTW REGION-GROWING PILOT
# ============================================================

def run_stage4_kf_signed_pilot(
    analysis,
    output_dir,
    method="magnitude",
    kalman_prefix="full106_kalman_rts_v1",
    seed_prefix="full106_kalman_seeds_v1",
    run_id=None,
    max_candidates=2000,
    max_growth_calls=100,
    min_abs_change=0.02,
    radius=0.6,
    max_visits=50000,
    checkpoint_every=20,
    resume=True,
    verbose=True,
):
    """
    Stage 4 exploratory Kalman-guided signed DTW segmentation.

    Uses saved KF-Mag or KF-Rate temporal seed candidates.
    Locates the strongest raw M3C2 transition matching the
    seed direction and applies the existing signed DTW grower.

    IMPORTANT:
    This is a two-epoch event-localization pilot, not the
    final full-temporal Kalman segmentation methodology.

    Existing Stage 1, 2 and 3 outputs are not modified.
    """

    import os
    import json
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    # grow_signed is bundled below from the validated v2 implementation

    if method not in ("magnitude", "rate"):
        raise ValueError(
            "method must be 'magnitude' or 'rate'"
        )

    if min(
        max_candidates,
        max_growth_calls,
        checkpoint_every
    ) < 1:
        raise ValueError(
            "Processing budgets must be positive."
        )

    out = Path(output_dir)

    run_id = run_id or (
        f"stage4_{method}_signed_pilot_v1"
    )

    seed_path = (
        out / f"{seed_prefix}_{method}.csv"
    )

    checkpoint_path = (
        out / f"{run_id}_checkpoint.npz"
    )

    objects_path = (
        out / f"{run_id}_objects.csv"
    )

    attempts_path = (
        out / f"{run_id}_attempts.csv"
    )

    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    meta_path = (
        out / f"{kalman_prefix}_metadata.json"
    )

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    if not meta.get("complete"):
        raise ValueError(
            "Stage 2 Kalman processing is incomplete."
        )

    xyz = np.asarray(
        analysis.corepoints.cloud
    )

    signals = np.asarray(
        analysis.distances_for_compute
    )

    seeds = pd.read_csv(seed_path)

    if tuple(meta["config"]["shape"]) != tuple(
        signals.shape
    ):
        raise ValueError(
            "Kalman metadata and M3C2 dimensions differ."
        )

    required = {
        "corepoint_index_python",
        "start_epoch",
        "end_epoch",
        "direction"
    }

    missing = required - set(seeds.columns)

    if missing:
        raise ValueError(
            f"Missing seed columns: {missing}"
        )

    config = {
        "version": "stage4-event-pilot-v1",
        "method": method,
        "kalman_prefix": kalman_prefix,
        "seed_prefix": seed_prefix,
        "seed_count": len(seeds),
        "shape": list(signals.shape),
        "min_abs_change": float(min_abs_change),
        "radius": float(radius),
        "max_visits": int(max_visits),
        "growth_window": (
            "raw_max_adjacent_change_two_epochs"
        ),
        "thresholds": [
            0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9
        ],
        "min_points": 20,
        "cv_limit": 0.8
    }

    # --------------------------------------------------------
    # RESUME EXISTING CHECKPOINT
    # --------------------------------------------------------

    if checkpoint_path.exists():

        if not resume:
            raise FileExistsError(checkpoint_path)

        with np.load(
            checkpoint_path,
            allow_pickle=False
        ) as saved:

            state = json.loads(
                saved["state_utf8"]
                .tobytes()
                .decode("utf-8")
            )

            flat = saved["indices"].copy()
            offsets = saved["offsets"].copy()

        if state["config"] != config:
            raise ValueError(
                "Checkpoint settings changed. "
                "Use a new run_id."
            )

        rows = state["rows"]
        attempts = state["attempts"]

        cursor = int(state["cursor"])
        calls = int(state["growth_calls"])

        members = [
            flat[offsets[k]:offsets[k + 1]].copy()
            for k in range(len(offsets) - 1)
        ]

        if len(members) != len(rows):
            raise ValueError(
                "Checkpoint membership mismatch."
            )

    else:

        if (
            objects_path.exists()
            or attempts_path.exists()
        ):
            raise FileExistsError(
                "Partial outputs already exist. "
                "Choose another run_id."
            )

        rows = []
        attempts = []
        members = []

        cursor = 0
        calls = 0

    # --------------------------------------------------------
    # RESTORE ALREADY COVERED EVENTS
    # --------------------------------------------------------

    covered = {
        1: {},
        -1: {}
    }

    for row, ids in zip(rows, members):

        sign = int(row["event_sign"])
        event_epoch = int(row["event_epoch"])

        for cp in ids:

            covered[sign].setdefault(
                int(cp), set()
            ).add(event_epoch)

    tree = cKDTree(xyz[:, :2])

    limit = min(
        len(seeds),
        int(max_candidates)
    )

    if (
        limit < cursor
        or max_growth_calls < calls
    ):
        raise ValueError(
            "Requested budgets are smaller "
            "than checkpoint progress."
        )

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # CHECKPOINT FUNCTION
    # --------------------------------------------------------

    def save():

        sizes = np.asarray(
            [len(m) for m in members],
            dtype=np.int64
        )

        offsets = np.r_[
            0,
            np.cumsum(sizes)
        ]

        flat = (
            np.concatenate(members).astype(
                np.int64
            )
            if members
            else np.empty(
                0, dtype=np.int64
            )
        )

        state = {
            "config": config,
            "rows": rows,
            "attempts": attempts,
            "cursor": cursor,
            "growth_calls": calls,
            "complete": cursor == len(seeds)
        }

        tmp = Path(
            str(checkpoint_path) + ".tmp"
        )

        encoded = json.dumps(
            state,
            allow_nan=False
        ).encode("utf-8")

        with tmp.open("wb") as f:

            np.savez_compressed(
                f,
                indices=flat,
                offsets=offsets,
                state_utf8=np.frombuffer(
                    encoded,
                    dtype=np.uint8
                )
            )

        os.replace(
            tmp,
            checkpoint_path
        )

        pd.DataFrame(rows).to_csv(
            objects_path,
            index=False
        )

        pd.DataFrame(attempts).to_csv(
            attempts_path,
            index=False
        )

    # --------------------------------------------------------
    # MAIN SEGMENTATION LOOP
    # --------------------------------------------------------

    while (
        cursor < limit
        and calls < max_growth_calls
    ):

        row = seeds.iloc[cursor]

        rank = cursor + 1
        cursor += 1

        cp = int(
            row["corepoint_index_python"]
        )

        a = int(row["start_epoch"])
        b = int(row["end_epoch"])

        sign = (
            1
            if row["direction"] == "positive"
            else -1
        )

        if not (
            0 <= cp < signals.shape[0]
            and 0 <= a < b < signals.shape[1]
        ):

            attempts.append({
                "seed_rank": rank,
                "status": "invalid_interval"
            })

            continue

        # Find adjacent-epoch changes matching
        # the seed's expected direction.

        steps = np.diff(
            signals[cp, a:b + 1]
        )

        valid_steps = (
            np.isfinite(steps)
            & (
                sign * steps
                > min_abs_change
            )
        )

        if not valid_steps.any():

            attempts.append({
                "seed_rank": rank,
                "seed_index": cp,
                "status": (
                    "no_raw_same_sign_transition"
                )
            })

            continue

        # Largest same-sign raw transition.
        local = int(
            np.argmax(
                np.where(
                    valid_steps,
                    sign * steps,
                    -np.inf
                )
            )
        )

        event_epoch = a + local

        # Skip events already covered by an
        # accepted object with the same sign.

        if event_epoch in covered[
            sign
        ].get(cp, set()):

            attempts.append({
                "seed_rank": rank,
                "seed_index": cp,
                "event_epoch": event_epoch,
                "status": (
                    "covered_same_sign_event"
                )
            })

            continue

        # ----------------------------------------------------
        # RUN EXISTING SIGNED DTW REGION GROWING
        # ----------------------------------------------------

        seed = SimpleNamespace(
            index=cp,
            start_epoch=event_epoch,
            end_epoch=event_epoch + 1
        )

        calls += 1

        tick = time.perf_counter()

        obj = grow_signed(
            signals,
            xyz,
            seed,
            tree=tree,
            radius=radius,
            min_abs_change=min_abs_change,
            max_visits=max_visits
        )

        status = (
            "accepted"
            if obj is not None
            else "no_accepted_object"
        )

        attempts.append({
            "seed_rank": rank,
            "seed_index": cp,
            "event_epoch": event_epoch,
            "event_sign": sign,
            "status": status,
            "elapsed_s": round(
                time.perf_counter() - tick,
                3
            )
        })

        if obj is not None:

            ids = np.asarray(
                obj.indices,
                dtype=np.int64
            )

            delta = (
                signals[ids, event_epoch + 1]
                - signals[ids, event_epoch]
            )

            rows.append({
                "object_number": len(rows) + 1,
                "method": method,
                "seed_rank": rank,
                "seed_index": cp,
                "original_start_epoch": a,
                "original_end_epoch": b,
                "start_epoch": event_epoch,
                "end_epoch": event_epoch + 1,
                "event_epoch": event_epoch,
                "event_sign": sign,
                "n_corepoints": len(ids),
                "threshold": float(
                    obj.threshold
                ),
                "cv_dtw": float(
                    obj.cv_dtw
                ),
                "median_event_change_m": float(
                    np.nanmedian(delta)
                ),
                "expected_sign_fraction": float(
                    np.mean(sign * delta > 0)
                )
            })

            members.append(ids)

            for point in ids:

                covered[sign].setdefault(
                    int(point), set()
                ).add(event_epoch)

        # ----------------------------------------------------
        # PERIODIC CHECKPOINT
        # ----------------------------------------------------

        if calls % checkpoint_every == 0:

            save()

            if verbose:

                elapsed = (
                    time.perf_counter() - t0
                )

                print(
                    f"{method}: "
                    f"scanned {cursor}/{limit}, "
                    f"growth {calls}/{max_growth_calls}, "
                    f"accepted {len(rows)}, "
                    f"elapsed {elapsed:.1f}s",
                    flush=True
                )

    # --------------------------------------------------------
    # SAVE FINAL PILOT RESULTS
    # --------------------------------------------------------

    save()

    elapsed = (
        time.perf_counter() - t0
    )

    return {
        "method": method,
        "scanned_candidates": cursor,
        "growth_calls": calls,
        "accepted_objects": len(rows),
        "total_seeds": len(seeds),
        "checkpoint": str(checkpoint_path),
        "objects": pd.DataFrame(rows),
        "attempts": pd.DataFrame(attempts),
        "memberships": members,
        "elapsed_s": round(elapsed, 2)
    }



# ============================================================
# STAGE 4D.1 + 4E — FULL-DOMAIN KALMAN SEGMENTATION
# Selected process_sigma = 0.5
# ============================================================

def _stage4e_runs(mask):
    a = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    return list(zip(
        np.flatnonzero(np.diff(a) == 1),
        np.flatnonzero(np.diff(a) == -1) - 1
    ))


def _stage4e_detect_events(
    y, r, sy, sr,
    z=1.96,
    min_change=0.02,
    min_run=2,
    gap=2,
    gradual_epochs=12
):
    """
    Detect temporal candidates from Kalman-derived states.

    Rate: significant-rate intervals.
    Magnitude: significance-state transitions.

    Labels are provisional, not ground-truth process types.
    RTS cross-epoch covariance is not available, so magnitude
    difference significance is approximate.
    """

    n = len(y)

    good = (
        np.isfinite(y)
        & np.isfinite(sy)
        & (sy > 0)
    )

    valid = np.flatnonzero(good)

    if not len(valid):
        return []

    ref = int(valid[0])
    diff = y - y[ref]
    sig_diff = np.sqrt(sy**2 + sy[ref]**2)

    state = np.zeros(n, dtype=np.int8)

    significant = (
        good
        & (np.abs(diff) >= min_change)
        & (np.abs(diff) > z * sig_diff)
    )

    state[significant] = np.sign(
        diff[significant]
    ).astype(np.int8)

    transitions = (
        np.flatnonzero(np.diff(state) != 0) + 1
    )

    candidates = []

    # --------------------------------------------------------
    # KF-RATE CANDIDATES
    # --------------------------------------------------------

    good_r = (
        np.isfinite(r)
        & np.isfinite(sr)
        & (sr > 0)
    )

    for sign in (-1, 1):

        intervals = _stage4e_runs(
            good_r & (sign * r > z * sr)
        )

        merged = []

        for a, b in intervals:

            if (
                merged
                and a - merged[-1][1] - 1 <= gap
            ):
                merged[-1][1] = b
            else:
                merged.append([a, b])

        for a, b in merged:

            if b - a + 1 < min_run:
                continue

            pre = max(0, a - 1)
            net = y[b] - y[pre]

            if (
                not np.isfinite(net)
                or sign * net < min_change
            ):
                continue

            duration = b - a + 1

            label = (
                "gradual"
                if duration >= gradual_epochs
                else "sudden_candidate"
            )

            candidates.append(dict(
                start=int(pre),
                end=int(b),
                sign=sign,
                label=label,
                source="rate",
                net=float(net),
                duration=int(b - pre + 1)
            ))

    # --------------------------------------------------------
    # KF-MAGNITUDE CANDIDATES
    # --------------------------------------------------------

    for t in transitions:

        old = int(state[t - 1])
        new = int(state[t])

        sign = new if new != 0 else -old

        if sign == 0:
            continue

        a = max(0, int(t) - 2)
        b = min(n - 1, int(t) + 2)

        net = y[b] - y[a]

        if (
            not np.isfinite(net)
            or sign * net < min_change
        ):
            continue

        kind = (
            "recovery_candidate"
            if old != 0 and new == 0
            else "sudden_candidate"
        )

        candidates.append(dict(
            start=a,
            end=b,
            sign=sign,
            label=kind,
            source="magnitude",
            net=float(net),
            duration=b - a + 1
        ))

    # --------------------------------------------------------
    # POSSIBLE TEMPORARY PROCESSES
    # --------------------------------------------------------

    for i, p in enumerate(candidates):

        if p["source"] != "magnitude":
            continue

        for q in candidates[i + 1:]:

            if (
                q["source"] != "magnitude"
                or q["sign"] != -p["sign"]
            ):
                continue

            if q["start"] <= p["end"]:
                continue

            before = y[max(0, p["start"])]
            after = y[min(n - 1, q["end"])]

            tolerance = max(
                min_change,
                0.25 * (
                    abs(p["net"])
                    + abs(q["net"])
                )
            )

            if (
                np.isfinite(before)
                and np.isfinite(after)
                and abs(after - before) <= tolerance
            ):
                p["label"] = "temporary_component"
                q["label"] = "temporary_component"
                break

    return candidates


def run_full_domain_processaware_segmentation(
    analysis,
    output_dir,
    kalman_prefix="full106_kalman_rts_q05_v1",
    run_id="full106_processaware_q05_v1",
    sample_corepoints=None,
    max_candidates=None,
    max_growth_calls=None,
    radius=0.6,
    min_abs_change=0.02,
    max_visits=50000,
    checkpoint_every=100,
    resume=True,
    verbose=True
):
    """
    Full-domain temporal detection + signed DTW segmentation.

    Uses completed process_sigma=0.5 Kalman + RTS arrays.
    Does not repeat Kalman processing.

    Temporal candidates are Kalman-derived.
    Spatial growth uses raw M3C2 over a bounded 12-epoch
    window, preserving the original signed-DTW grower.

    Outputs are checkpointed and resumable.
    """

    import os
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    # grow_signed is bundled below from the validated v2 implementation

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dest = out / run_id
    dest.mkdir(parents=True, exist_ok=True)

    cp_path = dest / "checkpoint.npz"
    candidate_path = dest / "candidates.csv"
    object_path = dest / "objects.csv"
    attempt_path = dest / "attempts.csv"
    candidate_config_path = dest / "candidate_config.json"

    raw = np.asarray(
        analysis.distances_for_compute
    )

    xyz = np.asarray(
        analysis.corepoints.cloud
    )

    n, t = raw.shape

    # --------------------------------------------------------
    # VERIFY COMPLETED KALMAN MODEL
    # --------------------------------------------------------

    meta_path = (
        out / f"{kalman_prefix}_metadata.json"
    )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing Kalman metadata: {meta_path}"
        )

    with meta_path.open(
        "r", encoding="utf-8"
    ) as fh:
        kalman_meta = json.load(fh)

    if (
        not kalman_meta.get("complete")
        or abs(
            float(
                kalman_meta["config"]["process_sigma"]
            ) - 0.5
        ) > 1e-12
    ):
        raise ValueError(
            "Completed process_sigma=0.5 "
            "Kalman arrays are required."
        )

    fields = {}

    for key in (
        "change",
        "rate",
        "sigma_change",
        "sigma_rate"
    ):

        path = out / f"{kalman_prefix}_{key}.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        fields[key] = np.load(
            path, mmap_mode="r"
        )

        if fields[key].shape != raw.shape:
            raise ValueError(
                f"Shape mismatch: {key}"
            )

    config = dict(
        version="stage4e-processaware-v1",
        shape=[n, t],
        prefix=kalman_prefix,
        sample_corepoints=(
            None if sample_corepoints is None
            else int(sample_corepoints)
        ),
        radius=float(radius),
        min_abs_change=float(min_abs_change),
        max_visits=int(max_visits),
        grower="signed_dtw",
        max_window_epochs=12,
        z=1.96,
        min_run=2,
        gap=2,
        gradual_epochs=12
    )

    # --------------------------------------------------------
    # TEMPORAL CANDIDATE DETECTION
    # --------------------------------------------------------

    if candidate_path.exists():

        if not candidate_config_path.exists():
            raise ValueError(
                "Candidate configuration missing. "
                "Use a new run_id."
            )

        with candidate_config_path.open(
            "r", encoding="utf-8"
        ) as fh:
            old_config = json.load(fh)

        if old_config != config:
            raise ValueError(
                "Candidate configuration differs. "
                "Use a new run_id."
            )

    else:

        if cp_path.exists():
            raise RuntimeError(
                "Checkpoint exists but candidate "
                "table is missing."
            )

        selected = (
            np.arange(n, dtype=int)
            if sample_corepoints is None
            else np.unique(
                np.linspace(
                    0,
                    n - 1,
                    min(int(sample_corepoints), n),
                    dtype=int
                )
            )
        )

        records = []

        for k, cp in enumerate(selected, 1):

            series = [
                np.asarray(
                    fields[key][cp],
                    dtype=float
                )
                for key in (
                    "change",
                    "rate",
                    "sigma_change",
                    "sigma_rate"
                )
            ]

            events = _stage4e_detect_events(
                *series
            )

            for event in events:
                records.append(dict(
                    corepoint=int(cp),
                    **event
                ))

            if verbose and k % 10000 == 0:
                print(
                    f"Temporal candidates: "
                    f"{k:,}/{len(selected):,} "
                    f"corepoints, "
                    f"{len(records):,} events",
                    flush=True
                )

        columns = [
            "corepoint", "start", "end",
            "sign", "label", "source",
            "net", "duration"
        ]

        candidates = pd.DataFrame(
            records,
            columns=columns
        )

        if not candidates.empty:

            candidates = candidates.sort_values(
                ["net", "corepoint", "start"],
                key=lambda s: (
                    s.abs()
                    if s.name == "net"
                    else s
                ),
                ascending=[False, True, True],
                kind="mergesort"
            ).reset_index(drop=True)

        candidates.to_csv(
            candidate_path,
            index=False
        )

        _atomic_json_write(
            candidate_config_path,
            config
        )

    candidates = pd.read_csv(
        candidate_path
    )

    if not candidates.empty:

        if (
            candidates.corepoint.min() < 0
            or candidates.corepoint.max() >= n
        ):
            raise ValueError(
                "Candidate indices do not match analysis."
            )

    # --------------------------------------------------------
    # LOAD OR INITIALIZE SEGMENTATION CHECKPOINT
    # --------------------------------------------------------

    if cp_path.exists():

        if not resume:
            raise FileExistsError(cp_path)

        with np.load(
            cp_path,
            allow_pickle=False
        ) as saved:

            state = json.loads(
                saved["state"]
                .tobytes()
                .decode()
            )

            flat = saved["members"].copy()
            offsets = saved["offsets"].copy()

        if state["config"] != config:
            raise ValueError(
                "Checkpoint settings differ. "
                "Use a new run_id."
            )

        cursor = int(state["cursor"])
        calls = int(state["calls"])

        rows = state["objects"]
        attempts = state["attempts"]

        memberships = [
            flat[
                offsets[i]:offsets[i + 1]
            ].copy()
            for i in range(len(offsets) - 1)
        ]

    else:

        cursor = 0
        calls = 0
        rows = []
        attempts = []
        memberships = []

    if len(rows) != len(memberships):
        raise ValueError(
            "Checkpoint membership count mismatch."
        )

    limit = (
        len(candidates)
        if max_candidates is None
        else min(
            len(candidates),
            int(max_candidates)
        )
    )

    growth_limit = (
        len(candidates)
        if max_growth_calls is None
        else int(max_growth_calls)
    )

    if (
        limit < cursor
        or growth_limit < calls
    ):
        raise ValueError(
            "New budgets are smaller than "
            "checkpoint progress."
        )

    tree = cKDTree(
        xyz[:, :2]
    )

    started = time.perf_counter()

    # --------------------------------------------------------
    # RECONSTRUCT COVERED SPATIOTEMPORAL REGIONS
    # --------------------------------------------------------

    covered = {
        1: {},
        -1: {}
    }

    for obj, ids in zip(
        rows, memberships
    ):

        sign = int(obj["sign"])
        a = int(obj["start"])
        b = int(obj["end"])

        for cp in ids:

            covered[sign].setdefault(
                int(cp), []
            ).append((a, b))

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save():

        sizes = np.asarray(
            [len(m) for m in memberships],
            dtype=np.int64
        )

        offsets = np.r_[
            0,
            np.cumsum(sizes)
        ]

        flat = (
            np.concatenate(
                memberships
            ).astype(np.int64)
            if memberships
            else np.empty(
                0, dtype=np.int64
            )
        )

        state = dict(
            config=config,
            cursor=cursor,
            calls=calls,
            objects=rows,
            attempts=attempts
        )

        tmp = Path(
            str(cp_path) + ".tmp"
        )

        with tmp.open("wb") as fh:

            np.savez_compressed(
                fh,
                members=flat,
                offsets=offsets,
                state=np.frombuffer(
                    json.dumps(
                        state,
                        allow_nan=False
                    ).encode(),
                    dtype=np.uint8
                )
            )

        os.replace(
            tmp,
            cp_path
        )

        pd.DataFrame(rows).to_csv(
            object_path,
            index=False
        )

        pd.DataFrame(attempts).to_csv(
            attempt_path,
            index=False
        )

    # --------------------------------------------------------
    # SIGNED DTW SPATIAL REGION GROWING
    # --------------------------------------------------------

    while (
        cursor < limit
        and calls < growth_limit
    ):

        c = candidates.iloc[cursor]

        rank = cursor + 1
        cursor += 1

        cp = int(c.corepoint)
        a = int(c.start)
        b = int(c.end)
        sign = int(c.sign)

        if not (
            0 <= cp < n
            and 0 <= a < b < t
            and sign in (-1, 1)
        ):

            attempts.append(dict(
                rank=rank,
                status="invalid_candidate"
            ))
            continue

        # Suppress covered candidates of same sign
        # with overlapping temporal intervals.

        if any(
            max(a, x) <= min(b, y)
            for x, y in covered[sign].get(
                cp, []
            )
        ):

            attempts.append(dict(
                rank=rank,
                status="covered_same_sign_interval"
            ))
            continue

        # ----------------------------------------------------
        # SELECT BOUNDED SPATIAL-GROWTH WINDOW
        # ----------------------------------------------------

        steps = np.diff(
            raw[cp, a:b + 1]
        )

        eligible = (
            np.isfinite(steps)
            & (
                sign * steps > min_abs_change
            )
        )

        if not eligible.any():

            attempts.append(dict(
                rank=rank,
                status="no_same_sign_raw_step"
            ))
            continue

        peak = a + int(
            np.argmax(
                np.where(
                    eligible,
                    sign * steps,
                    -np.inf
                )
            )
        )

        half = 6

        wa = max(
            a,
            peak - half + 1
        )

        wb = min(
            b,
            wa + 11
        )

        wa = max(
            a,
            wb - 11
        )

        if not (
            wa <= peak < wb
        ):

            attempts.append(dict(
                rank=rank,
                status="invalid_growth_window"
            ))
            continue

        seed = SimpleNamespace(
            index=cp,
            start_epoch=wa,
            end_epoch=wb
        )

        # ----------------------------------------------------
        # RUN ORIGINAL SIGNED DTW GROWER
        # ----------------------------------------------------

        calls += 1

        obj = grow_signed(
            raw,
            xyz,
            seed,
            tree=tree,
            radius=radius,
            min_abs_change=min_abs_change,
            max_visits=max_visits
        )

        if obj is None:

            attempts.append(dict(
                rank=rank,
                status="no_accepted_object"
            ))

        elif int(obj.event_sign) != sign:

            attempts.append(dict(
                rank=rank,
                status="direction_mismatch"
            ))

        else:

            ids = np.asarray(
                obj.indices,
                dtype=np.int64
            )

            rows.append(dict(
                object_id=len(rows) + 1,
                source=str(c.source),
                process_label=str(c.label),
                seed_corepoint=cp,
                sign=sign,
                start=a,
                end=b,
                growth_start=wa,
                growth_end=wb,
                event_epoch=int(obj.event_epoch),
                n_corepoints=len(ids),
                dtw_threshold=float(
                    obj.threshold
                ),
                cv_dtw=float(
                    obj.cv_dtw
                ),
                net_kalman_change_m=float(
                    c.net
                )
            ))

            memberships.append(ids)

            for p in ids:

                covered[sign].setdefault(
                    int(p), []
                ).append((a, b))

            attempts.append(dict(
                rank=rank,
                status="accepted"
            ))

        # ----------------------------------------------------
        # PERIODIC CHECKPOINT
        # ----------------------------------------------------

        if calls % checkpoint_every == 0:

            save()

            if verbose:

                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    f"Stage 4E: scanned "
                    f"{cursor}/{limit}, "
                    f"growth {calls}/{growth_limit}, "
                    f"accepted {len(rows)}, "
                    f"elapsed {elapsed:.1f}s",
                    flush=True
                )

    # --------------------------------------------------------
    # FINAL SAVE AND RETURN
    # --------------------------------------------------------

    save()

    return dict(
        candidates=candidates,
        objects=pd.DataFrame(rows),
        attempts=pd.DataFrame(attempts),
        memberships=memberships,
        cursor=cursor,
        growth_calls=calls,
        elapsed_s=(
            time.perf_counter() - started
        ),
        output_dir=str(dest)
    )



# ============================================================
# STAGE 4F-A — ORIGINAL KF-MAG / KF-RATE SEED NMS
# Reuses saved sigma=0.5 Kalman arrays
# ============================================================

def _stage4f_indexed_nms(
    table,
    xyz,
    radius=3.0,
    min_support_count=3,
    min_support_fraction=0.30,
    verbose=True
):
    """
    Indexed implementation of the original _apply_nms rules.

    Rank by absolute net change, support fraction and duration.
    Count neighbouring seed records of the same direction with
    overlapping temporal intervals.
    Retained seeds suppress lower-ranked matching neighbours.

    This is seed-level NMS, not object-level IoU NMS.
    """
    from collections import defaultdict

    if table.empty:
        return table.copy()

    table = table.copy()

    if "support_start_epoch" not in table:
        table["support_start_epoch"] = table["start_epoch"]

    if "support_end_epoch" not in table:
        table["support_end_epoch"] = table["end_epoch"]

    table["support_duration_epochs"] = (
        table["support_end_epoch"]
        - table["support_start_epoch"] + 1
    )

    if "support_fraction" not in table:
        table["support_fraction"] = 0.0

    table = table.sort_values(
        [
            "abs_net_change",
            "support_fraction",
            "support_duration_epochs"
        ],
        ascending=[False, False, False],
        kind="stable"
    ).reset_index(drop=True)

    coords = np.asarray(xyz)[:, :2]
    tree = cKDTree(coords)

    cp = table["corepoint_index_python"].to_numpy(
        dtype=np.int64
    )
    starts = table["support_start_epoch"].to_numpy(
        dtype=np.int64
    )
    ends = table["support_end_epoch"].to_numpy(
        dtype=np.int64
    )
    dirs = table["direction"].astype(str).to_numpy()

    by_cp = defaultdict(list)

    for i, p in enumerate(cp):
        by_cp[int(p)].append(i)

    suppressed = np.zeros(len(table), dtype=bool)

    retained = []
    support_counts = []
    support_fracs = []

    for i in range(len(table)):

        if suppressed[i]:
            continue

        neighbors = tree.query_ball_point(
            coords[cp[i]],
            r=radius
        )

        neighbors = [
            int(p) for p in neighbors
            if int(p) != int(cp[i])
        ]

        matches = []

        for p in neighbors:
            for j in by_cp.get(p, ()):

                if (
                    dirs[j] == dirs[i]
                    and min(ends[i], ends[j])
                    >= max(starts[i], starts[j])
                ):
                    matches.append(j)

        count = len(matches)
        fraction = count / max(len(neighbors), 1)

        if (
            count < min_support_count
            or fraction < min_support_fraction
        ):
            continue

        retained.append(i)
        support_counts.append(count)
        support_fracs.append(fraction)

        for j in matches:
            if j > i:
                suppressed[j] = True

        if verbose and len(retained) % 1000 == 0:
            print(
                f"NMS retained {len(retained):,}; "
                f"examined {i+1:,}/{len(table):,}",
                flush=True
            )

    result = table.iloc[retained].copy()

    result["nms_support_count"] = support_counts
    result["nms_support_fraction"] = support_fracs

    result = result.sort_values(
        [
            "abs_net_change",
            "nms_support_fraction",
            "support_duration_epochs"
        ],
        ascending=[False, False, False],
        kind="stable"
    ).reset_index(drop=True)

    result["nms_rank"] = np.arange(
        1, len(result) + 1
    )

    return result


def run_stage4f_seed_nms(
    analysis,
    output_dir,
    kalman_prefix="full106_kalman_rts_q05_v1",
    seed_prefix="stage4f_q05_seeds_v1",
    nms_radius=3.0,
    nms_min_support_count=3,
    nms_min_support_fraction=0.30,
    max_seeds_per_mode=None,
    verbose=True
):
    """
    Generate KF-Mag, KF-Rate, KF-Mag-NMS and KF-Rate-NMS
    seed tables from the saved full-domain Kalman arrays.

    max_seeds_per_mode is for diagnostic pilots only.
    Spatial support validation is disabled to isolate NMS.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / f"{kalman_prefix}_metadata.json"

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    if (
        not meta.get("complete")
        or abs(
            float(meta["config"]["process_sigma"]) - 0.5
        ) > 1e-12
    ):
        raise ValueError(
            "Completed sigma=0.5 Kalman arrays required"
        )

    shape = tuple(meta["config"]["shape"])
    xyz = np.asarray(analysis.corepoints.cloud)

    if len(xyz) != shape[0]:
        raise ValueError(
            "Corepoint count differs from Kalman metadata"
        )

    seed_files = detect_full_domain_kalman_seeds(
        output_dir=out,
        kalman_prefix=kalman_prefix,
        seed_prefix=seed_prefix,
        resume=True,
        verbose=verbose
    )

    results = {}

    for mode, short in [
        ("magnitude", "kfmag"),
        ("rate", "kfrate")
    ]:

        source = Path(seed_files[mode])
        table = pd.read_csv(source)

        if max_seeds_per_mode is not None:
            table = table.head(
                int(max_seeds_per_mode)
            ).copy()

        nms_path = (
            out / f"{seed_prefix}_{short}_nms.csv"
        )

        config_path = (
            out / f"{seed_prefix}_{short}_nms_config.json"
        )

        config = dict(
            kalman_prefix=kalman_prefix,
            mode=mode,
            source=str(source),
            source_rows=int(len(table)),
            radius=float(nms_radius),
            min_support_count=int(nms_min_support_count),
            min_support_fraction=float(
                nms_min_support_fraction
            ),
            max_seeds_per_mode=max_seeds_per_mode,
            spatial_support_validation=False
        )

        if nms_path.exists():

            if not config_path.exists():
                raise ValueError(
                    f"Missing configuration for {nms_path}"
                )

            with config_path.open(
                encoding="utf-8"
            ) as f:
                old = json.load(f)

            if old != config:
                raise ValueError(
                    "Existing NMS settings differ; "
                    "change seed_prefix"
                )

            nms = pd.read_csv(nms_path)

        else:

            if verbose:
                print(
                    f"Running {short.upper()}-NMS "
                    f"on {len(table):,} seeds",
                    flush=True
                )

            nms = _stage4f_indexed_nms(
                table,
                xyz,
                radius=nms_radius,
                min_support_count=nms_min_support_count,
                min_support_fraction=nms_min_support_fraction,
                verbose=verbose
            )

            nms.to_csv(nms_path, index=False)
            _atomic_json_write(
                config_path, config
            )

        results[short] = dict(
            raw_count=len(table),
            nms_count=len(nms),
            raw_file=str(source),
            nms_file=str(nms_path)
        )

        print(
            f"{short.upper()}: "
            f"{len(table):,} raw -> "
            f"{len(nms):,} NMS seeds",
            flush=True
        )

    return results



# ============================================================
# STAGE 4F-B — FOUR KF VARIANTS, SIGNED-DTW SPATIAL GROWTH
# Append this block to m3c2_kalman_full.py (do not replace file)
# ============================================================

def run_stage4f_kf_spatial_growth(
    analysis, output_dir, method,
    seed_prefix="stage4f_q05_full_v1",
    kalman_prefix="full106_kalman_rts_q05_v1",
    run_id=None, radius=0.6, min_abs_change=0.02,
    max_visits=50000, max_candidates=None,
    max_growth_calls=None, checkpoint_every=100,
    resume=True, verbose=True,
):
    """Resumable, seed-level signed-DTW extraction for one KF variant.

    method: kfmag, kfrate, kfmag_nms, kfrate_nms
    Identical grower, ranking, raw-event localization and coverage policy
    for all four variants. KF NMS is performed upstream on seed records.
    No Kalman filtering or NMS is recomputed here.

    A successful grown object's (sign, event_epoch, member corepoints)
    prevents redundant growth from another seed at that same event epoch.
    This *post-growth* coverage check is NOT the pre-growth KF-NMS method.
    """
    import os
    import time
    import hashlib
    from types import SimpleNamespace
    from scipy.spatial import cKDTree
    # grow_signed is bundled below from the validated v2 implementation

    variants = {
        "kfmag": ("magnitude", False),
        "kfrate": ("rate", False),
        "kfmag_nms": ("magnitude", True),
        "kfrate_nms": ("rate", True),
    }
    if method not in variants:
        raise ValueError(f"method must be one of {tuple(variants)}")
    if checkpoint_every < 1 or radius <= 0 or min_abs_change < 0 or max_visits < 1:
        raise ValueError("Invalid growth or checkpoint parameters")
    for name, val in (("max_candidates", max_candidates),
                      ("max_growth_calls", max_growth_calls)):
        if val is not None and int(val) < 1:
            raise ValueError(f"{name} must be positive or None")

    out = Path(output_dir)
    mode, use_nms = variants[method]
    seed_name = (f"{seed_prefix}_{method}.csv" if use_nms
                 else f"{seed_prefix}_{mode}.csv")
    seed_path = out / seed_name
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Missing {seed_path}. Confirm Stage 4F-A filenames; "
            "use a separate seed_prefix for each seed-detection run."
        )
    meta_path = out / f"{kalman_prefix}_metadata.json"
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not meta.get("complete") or abs(float(meta["config"]["process_sigma"]) - 0.5) > 1e-12:
        raise ValueError("Completed sigma=0.5 Kalman metadata required")

    raw = np.asarray(analysis.distances_for_compute)
    xyz = np.asarray(analysis.corepoints.cloud)
    if raw.ndim != 2 or xyz.shape[0] != raw.shape[0] or xyz.shape[1] < 2:
        raise ValueError("Invalid analysis distances/corepoint dimensions")
    if tuple(meta["config"]["shape"]) != raw.shape:
        raise ValueError("Kalman metadata shape differs from analysis")

    # Fingerprint seed file to prevent silently resuming with altered seeds.
    digest = hashlib.sha256()
    with seed_path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    seed_hash = digest.hexdigest()

    seeds = pd.read_csv(seed_path)
    needed = {"corepoint_index_python", "start_epoch", "end_epoch",
              "direction", "abs_net_change"}
    if not needed.issubset(seeds.columns):
        raise ValueError(f"Missing seed columns: {needed - set(seeds.columns)}")
    if use_nms:
        # Preserve Stage 4F-A NMS priority.
        rank_cols = ["abs_net_change", "nms_support_fraction", "support_duration_epochs"]
    else:
        # Use the same priority as NMS to avoid a ranking confound.
        # Without spatial support filtering, support_fraction=0 for all.
        if "support_duration_epochs" not in seeds:
            seeds["support_duration_epochs"] = (
                seeds["end_epoch"] - seeds["start_epoch"] + 1
            )
        seeds["nms_support_fraction"] = 0.0
        rank_cols = ["abs_net_change", "nms_support_fraction", "support_duration_epochs"]
    if not set(rank_cols).issubset(seeds.columns):
        raise ValueError(f"Missing ranking columns: {set(rank_cols)-set(seeds.columns)}")
    seeds = seeds.sort_values(rank_cols, ascending=[False, False, False],
                              kind="stable").reset_index(drop=True)

    run_id = run_id or f"stage4f_b_q05_{method}_v1"
    dest = out / run_id
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dest / "checkpoint.npz"
    object_path = dest / "objects.csv"
    attempts_path = dest / "attempts.csv"
    config_path = dest / "config.json"
    config = dict(version="stage4f-b-signed-twoepoch-v1", method=method,
                  kalman_prefix=kalman_prefix, seed_path=str(seed_path),
                  seed_sha256=seed_hash, seed_count=len(seeds),
                  shape=list(raw.shape), radius=float(radius),
                  min_abs_change=float(min_abs_change), max_visits=int(max_visits),
                  growth_window="strongest_same_sign_raw_transition_two_epochs",
                  ranking=rank_cols,
                  coverage="accepted_object_member_same_sign_same_event_epoch",
                  grower="a5_direction_constrained_growing_v2.grow_signed")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            if json.load(fh) != config:
                raise ValueError("Run configuration changed: choose a new run_id")
    elif checkpoint_path.exists() or object_path.exists() or attempts_path.exists():
        raise ValueError("Partial results without config; choose a new run_id")
    else:
        _atomic_json_write(config_path, config)

    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(checkpoint_path)
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            state = json.loads(saved["state"].tobytes().decode("utf-8"))
            flat = saved["members"].copy()
            offsets = saved["offsets"].copy()
        if state["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        rows = state["objects"]
        attempts = state["attempts"]
        cursor = int(state["cursor"])
        calls = int(state["calls"])
        memberships = [flat[offsets[i]:offsets[i+1]].copy()
                       for i in range(len(offsets)-1)]
    else:
        rows, attempts, memberships, cursor, calls = [], [], [], 0, 0
    if len(rows) != len(memberships):
        raise ValueError("Corrupt checkpoint: membership count mismatch")

    covered = {1: {}, -1: {}}
    for obj, ids in zip(rows, memberships):
        sign = int(obj["event_sign"])
        epoch = int(obj["event_epoch"])
        for cp in ids:
            covered[sign].setdefault(int(cp), set()).add(epoch)

    limit = len(seeds) if max_candidates is None else min(len(seeds), int(max_candidates))
    growth_limit = float("inf") if max_growth_calls is None else int(max_growth_calls)
    if cursor > limit or calls > growth_limit:
        raise ValueError("New budget is smaller than checkpoint progress")
    tree = cKDTree(xyz[:, :2])
    t0 = time.perf_counter()

    def save():
        sizes = np.asarray([len(m) for m in memberships], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes)]
        flat = (np.concatenate(memberships).astype(np.int64)
                if memberships else np.empty(0, dtype=np.int64))
        state = dict(config=config, cursor=cursor, calls=calls,
                     objects=rows, attempts=attempts,
                     complete=(cursor == len(seeds)))
        tmp = Path(str(checkpoint_path) + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, members=flat, offsets=offsets,
                state=np.frombuffer(json.dumps(state, allow_nan=False).encode("utf-8"),
                                    dtype=np.uint8))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, checkpoint_path)
        pd.DataFrame(rows).to_csv(object_path, index=False)
        pd.DataFrame(attempts).to_csv(attempts_path, index=False)

    while cursor < limit and calls < growth_limit:
        row = seeds.iloc[cursor]
        rank = cursor + 1
        cursor += 1
        cp = int(row["corepoint_index_python"])
        a, b = int(row["start_epoch"]), int(row["end_epoch"])
        direction = str(row["direction"])
        if direction not in ("positive", "negative") or not (0 <= cp < raw.shape[0] and 0 <= a < b < raw.shape[1]):
            attempts.append(dict(rank=rank, status="invalid_seed"))
            continue
        sign = 1 if direction == "positive" else -1
        steps = np.diff(raw[cp, a:b+1])
        eligible = np.isfinite(steps) & (sign * steps > min_abs_change)
        if not eligible.any():
            attempts.append(dict(rank=rank, status="no_same_sign_raw_step"))
            continue
        epoch = a + int(np.argmax(np.where(eligible, sign * steps, -np.inf)))
        if epoch in covered[sign].get(cp, set()):
            attempts.append(dict(rank=rank, status="covered_same_sign_event",
                                 event_epoch=epoch))
            continue
        seed = SimpleNamespace(index=cp, start_epoch=epoch, end_epoch=epoch+1)
        calls += 1
        tick = time.perf_counter()
        obj = grow_signed(raw, xyz, seed, tree=tree, radius=radius,
                          min_abs_change=min_abs_change, max_visits=max_visits)
        if obj is None:
            status = "no_accepted_object"
        elif int(obj.event_sign) != sign:
            status = "direction_mismatch"
        else:
            status = "accepted"
            ids = np.unique(np.asarray(obj.indices, dtype=np.int64))
            rows.append(dict(object_id=len(rows)+1, method=method,
                seed_rank=rank, seed_corepoint=cp,
                original_start_epoch=a, original_end_epoch=b,
                event_epoch=epoch, event_sign=sign,
                growth_start_epoch=epoch, growth_end_epoch=epoch+1,
                n_corepoints=int(len(ids)),
                dtw_threshold=float(obj.threshold), cv_dtw=float(obj.cv_dtw),
                kalman_seed_abs_net_change=float(row["abs_net_change"])))
            memberships.append(ids)
            for p in ids:
                covered[sign].setdefault(int(p), set()).add(epoch)
        attempts.append(dict(rank=rank, seed_corepoint=cp,
                             event_epoch=epoch, event_sign=sign,
                             status=status,
                             elapsed_s=round(time.perf_counter()-tick, 4)))
        if calls % checkpoint_every == 0:
            save()
            if verbose:
                print(f"{method}: scanned {cursor:,}/{limit:,}; "
                      f"growth calls {calls:,}; objects {len(rows):,}; "
                      f"elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    save()
    if verbose:
        print(f"{method} COMPLETE/PAUSED: scanned {cursor:,}/{len(seeds):,}, "
              f"growth calls {calls:,}, objects {len(rows):,}, "
              f"elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    return dict(method=method, objects=pd.DataFrame(rows),
                attempts=pd.DataFrame(attempts), memberships=memberships,
                cursor=cursor, seed_count=len(seeds), growth_calls=calls,
                complete=(cursor == len(seeds)), output_dir=str(dest))




# ---- Embedded validated signed-DTW growth (verbatim from v2) ----
from collections import deque
from dataclasses import dataclass
import _py4dgeo

@dataclass
class SignedObject:
    indices: np.ndarray
    start_epoch: int
    end_epoch: int
    seed_index: int
    event_epoch: int
    event_sign: int
    threshold: float
    mean_dtw: float
    cv_dtw: float
    n_growth_visits: int
    reached_visit_limit: bool

def _event(signals, seed):
    i, a, b = int(seed.index), int(seed.start_epoch), int(seed.end_epoch)
    steps = np.diff(signals[i, a:b+1])
    if not np.isfinite(steps).any():
        return None
    t = a + int(np.nanargmax(np.abs(steps)))
    jump = float(signals[i, t+1] - signals[i, t])
    return (t, int(np.sign(jump))) if jump != 0 else None

def grow_signed(signals, xy, seed, tree=None, radius=0.6,
                thresholds=(0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                min_points=20, min_abs_change=0.0,
                max_visits=50000, cv_limit=0.8):
    """Grow sign-coherent connected regions, choose adaptive threshold.

    The sign test uses the strongest adjacent-epoch change at the seed; DTW
    uses the seed's complete [start_epoch, end_epoch] interval. The DTW metric
    is py4dgeo's compiled normalized_dtw_distance. The growth topology and
    adaptive selection are experimental, NOT identical to reference 4D-OBC.
    max_visits bounds work, NOT final object size; capped candidates are rejected.
    """
    signals = np.asarray(signals)
    xy = np.asarray(xy)[:, :2]
    if tree is None: tree = cKDTree(xy)
    event = _event(signals, seed)
    if event is None: return None
    epoch, sign = event
    a, b, i = int(seed.start_epoch), int(seed.end_epoch), int(seed.index)
    if not (0 <= a < b < signals.shape[1]): return None
    delta = signals[:, epoch+1] - signals[:, epoch]
    valid = np.isfinite(delta) & (sign * delta > min_abs_change)
    if not valid[i]: return None
    reference = np.asarray(signals[i, a:b+1], dtype=np.float64)
    if not np.isfinite(reference).all(): return None
    metric = _py4dgeo.normalized_dtw_distance
    distance_cache = {i: 0.0}
    neighbor_cache = {}
    def neighbors(j):
        if j not in neighbor_cache:
            neighbor_cache[j] = tree.query_ball_point(xy[j], r=radius)
        return neighbor_cache[j]
    def distance(j):
        if j not in distance_cache:
            other = np.asarray(signals[j, a:b+1], dtype=np.float64)
            if not np.isfinite(other).all():
                distance_cache[j] = np.inf
            else:
                arg = _py4dgeo.TimeseriesDistanceFunctionData(reference, other)
                distance_cache[j] = float(metric(arg))
        return distance_cache[j]
    candidates = []
    # Nested threshold solutions are grown separately to preserve connectivity.
    for threshold in sorted(set(float(t) for t in thresholds)):
        visited = {i}; members = {i}; frontier = deque([i]); visits = 0
        capped = False
        while frontier:
            cur = frontier.popleft()
            for j in neighbors(cur):
                j = int(j)
                if j in visited: continue
                visited.add(j)
                visits += 1
                if visits >= max_visits:
                    capped = True; break
                if valid[j] and distance(j) <= threshold:
                    members.add(j)
                    frontier.append(j)
            if capped: break
        if capped: continue  # never treat truncated region as valid
        if len(members) < min_points: continue
        ids = np.fromiter(members, dtype=np.int64)
        ds = np.array([distance(j) for j in ids], dtype=float)
        avg = float(ds.mean())
        cv = float(ds.std() / max(abs(avg), 1e-10))
        if cv > cv_limit: continue
        candidates.append((threshold, ids, avg, cv, visits))
    if not candidates: return None
    # Reference-inspired plateau selection: prefer the largest solution BEFORE
    # a sharp size increase (>2x). This is a documented experimental heuristic.
    chosen = candidates[0]
    for nxt in candidates[1:]:
        if len(nxt[1]) > 2 * len(chosen[1]): break
        chosen = nxt
    t, ids, avg, cv, visits = chosen
    return SignedObject(np.sort(ids), a, b, i, epoch, sign, t, avg, cv, visits, False)

# ---- Original full-series signed extraction (unchanged algorithm) ----
"""Checkpointed experimental signed region growing across all cached temporal seeds.

No seed redetection, no source-archive writes, no Kalman parameter changes.
Full temporal overlap is checked before spatial seed-coverage suppression.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _seed_event(signals, seed):
    i, a, b = int(seed.index), int(seed.start_epoch), int(seed.end_epoch)
    if not (0 <= i < signals.shape[0] and 0 <= a < b < signals.shape[1]):
        return None
    steps = np.diff(signals[i, a:b + 1])
    if not np.isfinite(steps).all():
        return None
    local = int(np.argmax(np.abs(steps)))
    t, change = a + local, float(steps[local])
    return (t, 1 if change > 0 else -1, change) if change != 0 else None


def _pack(rows, attempts, members, config, cursor, growth_count, completed):
    sizes = np.asarray([len(m) for m in members], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    flat = np.concatenate(members).astype(np.int64) if members else np.empty(0, dtype=np.int64)
    state = dict(config=config, cursor=int(cursor), growth_count=int(growth_count),
                 completed=bool(completed), rows=rows, attempts=attempts)
    return dict(indices=flat, offsets=offsets,
                state_utf8=np.frombuffer(json.dumps(state, allow_nan=False).encode('utf-8'), dtype=np.uint8))


def run_full_series(analysis, output_dir, run_id='full106_signed20_pilot_v1',
                    max_candidates=3000, max_growth_calls=100,
                    min_abs_change=0.02, radius=0.6, max_visits=50000,
                    checkpoint_every=20, resume=True, verbose=True):
    """Run bounded pilot, resume with higher budgets using same run_id.

    Candidate ordering follows original cached seed order, NOT event-specific ranking.
    Seed skipping requires SAME sign, a spatially covered seed point AND overlapping
    temporal intervals. Intervals [start,end] are treated as inclusive epoch windows.
    No global spatial exclusion across disjoint intervals. Accepted objects can still
    overlap; use post-extraction 4D containment diagnostics, not spatial-only pruning.
    """
    if max_candidates < 1 or max_growth_calls < 1 or checkpoint_every < 1:
        raise ValueError('Budgets and checkpoint_every must be positive')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{run_id}_checkpoint.npz'
    summary_path = folder / f'{run_id}_objects.csv'
    attempts_path = folder / f'{run_id}_attempts.csv'
    xyz = np.asarray(analysis.corepoints.cloud)
    signals = np.asarray(analysis.distances_for_compute)
    seeds = analysis.seeds
    if seeds is None:
        raise ValueError('Cached temporal seeds missing')
    config = dict(min_abs_change=float(min_abs_change), radius=float(radius),
                  max_visits=int(max_visits), thresholds=list(THRESHOLDS),
                  seed_count=len(seeds), corepoint_count=len(xyz),
                  n_epochs=signals.shape[1], seed_order='archive_order',
                  algorithm='a5_direction_constrained_growing_v2.grow_signed')
    if path.exists():
        if not resume:
            raise FileExistsError(f'Run already exists: {path}')
        with np.load(path, allow_pickle=False) as saved:
            state = json.loads(saved['state_utf8'].tobytes().decode('utf-8'))
            flat = saved['indices'].copy()
            offsets = saved['offsets'].copy()
        if state['config'] != config:
            raise ValueError('Run configuration differs from checkpoint. Choose new run_id.')
        members = [flat[offsets[k]:offsets[k + 1]].copy()
                   for k in range(len(offsets) - 1)]
        rows, attempts = state['rows'], state['attempts']
        cursor, growth_count = int(state['cursor']), int(state['growth_count'])
        if len(rows) != len(members):
            raise RuntimeError('Checkpoint membership mismatch')
    else:
        if summary_path.exists() or attempts_path.exists():
            raise FileExistsError('Partial output files exist; choose a new run_id')
        rows, attempts, members, cursor, growth_count = [], [], [], 0, 0
    # Point -> accepted temporal windows; separate by change direction.
    covered = {1: {}, -1: {}}
    for row, ids in zip(rows, members):
        sign = int(row['event_sign'])
        window = (int(row['start_epoch']), int(row['end_epoch']))
        for point in ids:
            covered[sign].setdefault(int(point), []).append(window)
    tree = cKDTree(xyz[:, :2])
    limit = min(int(max_candidates), len(seeds))
    if limit < cursor or max_growth_calls < growth_count:
        raise ValueError('Resume budgets cannot be smaller than checkpoint progress')
    start_time = time.perf_counter()

    def checkpoint():
        payload = _pack(rows, attempts, members, config, cursor, growth_count,
                        completed=cursor == len(seeds))
        temp = Path(str(path) + '.tmp')
        with open(temp, 'wb') as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temp, path)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        pd.DataFrame(attempts).to_csv(attempts_path, index=False)

    while cursor < limit and growth_count < max_growth_calls:
        seed = seeds[cursor]
        rank = cursor + 1
        cursor += 1
        event = _seed_event(signals, seed)
        if event is None:
            attempts.append(dict(seed_rank=rank, status='invalid_seed'))
            continue
        t, sign, jump = event
        i, a, b = int(seed.index), int(seed.start_epoch), int(seed.end_epoch)
        if abs(jump) < min_abs_change:
            attempts.append(dict(seed_rank=rank, seed_index=i, event_epoch=t,
                                 event_sign=sign, status='below_seed_magnitude'))
            continue
        # Temporal-overlap-aware skip: a location can participate in later events.
        windows = covered[sign].get(i, ())
        if any(a <= end and start <= b for start, end in windows):
            attempts.append(dict(seed_rank=rank, seed_index=i, event_epoch=t,
                                 event_sign=sign, status='covered_same_sign_time'))
            continue
        growth_count += 1
        tick = time.perf_counter()
        obj = grow_signed(signals, xyz, seed, tree=tree, radius=radius,
                          thresholds=THRESHOLDS, min_abs_change=min_abs_change,
                          max_visits=max_visits)
        status = 'no_accepted_object' if obj is None else 'accepted'
        attempts.append(dict(seed_rank=rank, seed_index=i, event_epoch=t,
                             event_sign=sign, status=status,
                             elapsed_s=round(time.perf_counter()-tick, 3)))
        if obj is not None:
            ids = np.asarray(obj.indices, dtype=np.int64)
            delta = signals[ids, t+1] - signals[ids, t]
            rows.append(dict(object_number=len(rows)+1, seed_rank=rank,
                             seed_index=i, start_epoch=int(obj.start_epoch),
                             end_epoch=int(obj.end_epoch), event_epoch=t,
                             event_sign=sign, n_corepoints=len(ids),
                             threshold=float(obj.threshold), cv_dtw=float(obj.cv_dtw),
                             median_event_change_m=float(np.nanmedian(delta)),
                             expected_sign_fraction=float(np.mean(sign*delta>0))))
            members.append(ids)
            window = (a, b)
            for point in ids:
                covered[sign].setdefault(int(point), []).append(window)
        if growth_count % checkpoint_every == 0:
            checkpoint()
            if verbose:
                print(f'Checkpoint: seeds scanned {cursor}/{limit}, growth calls '
                      f'{growth_count}/{max_growth_calls}, accepted {len(rows)}, '
                      f'elapsed {time.perf_counter()-start_time:.1f}s', flush=True)
    checkpoint()
    return dict(seed_count=len(seeds), scanned_candidates=cursor,
                growth_calls=growth_count, accepted_objects=len(rows),
                full_series_scanned=(cursor == len(seeds)),
                checkpoint=str(path), objects=pd.DataFrame(rows),
                attempts=pd.DataFrame(attempts), memberships=members,
                elapsed_s=round(time.perf_counter()-start_time, 2))


def four_d_overlap_table(objects, memberships):
    """Diagnostic pairwise 4D overlap; temporal IoU uses epoch intervals.

    Containment is measured relative to smaller spatial membership, and only
    overlapping same-sign intervals are compared. No objects are removed.
    """
    records = []
    if len(objects) != len(memberships):
        raise ValueError('Objects and memberships length mismatch')
    for i in range(len(objects)):
        a = objects.iloc[i]
        A = set(map(int, memberships[i]))
        for j in range(i+1, len(objects)):
            b = objects.iloc[j]
            if int(a.event_sign) != int(b.event_sign):
                continue
            ta, tb = (int(a.start_epoch), int(a.end_epoch)), (int(b.start_epoch), int(b.end_epoch))
            overlap_epochs = max(0, min(ta[1], tb[1]) - max(ta[0], tb[0]) + 1)
            if not overlap_epochs:
                continue
            B = set(map(int, memberships[j]))
            intersection = len(A & B)
            if not intersection:
                continue
            time_union = max(ta[1], tb[1]) - min(ta[0], tb[0]) + 1
            records.append(dict(object_a=int(a.object_number), object_b=int(b.object_number),
                                spatial_intersection=intersection,
                                spatial_containment_smaller=intersection/min(len(A),len(B)),
                                spatial_iou=intersection/len(A|B),
                                temporal_iou=overlap_epochs/time_union,
                                same_event_transition=(int(a.event_epoch)==int(b.event_epoch))))
    return pd.DataFrame(records)


def suppress_event_aware_duplicates(
    method_objects,
    containment_threshold=0.95,
    n_corepoints=None,
):
    """
    Suppress spatially nested change objects within the same event.

    Reproduces the event-aware containment algorithm from
    Stage 4H.10B of the validated thesis notebook.

    Parameters
    ----------
    method_objects : dict
        Mapping from method name to object sequences.
        Each object must contain:
        indices, event_epoch, event_sign.

    containment_threshold : float
        Minimum fraction of a smaller object's corepoints
        contained in a larger retained object.

    n_corepoints : int or None
        Optional total number of analysis corepoints for
        index validation.

    Returns
    -------
    dict
        object_catalogues
        refined_catalogues
        refined_catalogue_df
        pairwise_overlap_df
        coverage_summary_df
    """
    import numpy as np
    import pandas as pd

    if not 0 <= containment_threshold <= 1:
        raise ValueError(
            "containment_threshold must be between 0 and 1."
        )

    method_order = [
        "original",
        "kfmag",
        "kfrate",
        "kfmag_nms",
        "kfrate_nms",
    ]

    object_catalogues = {}

    for method in method_order:

        objects = []

        for object_row, obj in enumerate(method_objects[method]):

            indices = np.unique(
                np.asarray(obj["indices"], dtype=np.int64)
            )

            if len(indices) == 0:
                continue

            if np.any(indices < 0):
                raise ValueError(
                    f"Negative corepoint index: {method}, {object_row}"
                )

            if (
                n_corepoints is not None
                and np.any(indices >= n_corepoints)
            ):
                raise ValueError(
                    f"Out-of-range corepoint index: "
                    f"{method}, {object_row}"
                )

            objects.append({
                "object_row": object_row,
                "object_id": object_row + 1,
                "indices": indices,
                "event_epoch": int(obj["event_epoch"]),
                "event_sign": int(obj["event_sign"]),
                "size": len(indices),
            })

        object_catalogues[method] = objects

    refined_catalogues = {}
    catalogue_rows = []
    overlap_rows = []
    coverage_rows = []

    for method in method_order:

        objects = object_catalogues[method]

        groups = {}

        for obj in objects:
            key = (obj["event_epoch"], obj["event_sign"])
            groups.setdefault(key, []).append(obj)

        retained_rows = set()
        suppressed_info = {}

        for (epoch, sign), group in groups.items():

            priority = sorted(
                group,
                key=lambda o: (-o["size"], o["object_row"])
            )

            retained_group = []

            for obj in priority:

                current = obj["indices"]

                best_parent = None
                best_containment = 0.0
                best_iou = 0.0

                for parent in retained_group:

                    parent_idx = parent["indices"]

                    intersection = np.intersect1d(
                        current,
                        parent_idx,
                        assume_unique=True,
                    ).size

                    if intersection == 0:
                        continue

                    containment = intersection / len(current)

                    union = (
                        len(current)
                        + len(parent_idx)
                        - intersection
                    )

                    iou = intersection / union

                    overlap_rows.append({
                        "method": method,
                        "event_epoch": epoch,
                        "event_sign": sign,
                        "object_row": obj["object_row"],
                        "parent_row": parent["object_row"],
                        "intersection": intersection,
                        "containment": containment,
                        "spatial_iou": iou,
                    })

                    if containment > best_containment:
                        best_parent = parent["object_row"]
                        best_containment = containment
                        best_iou = iou

                if (
                    best_parent is not None
                    and best_containment >= containment_threshold
                ):
                    suppressed_info[obj["object_row"]] = {
                        "parent_row": best_parent,
                        "containment": best_containment,
                        "iou": best_iou,
                    }
                else:
                    retained_group.append(obj)
                    retained_rows.add(obj["object_row"])

        refined_objects = []

        for obj in objects:

            row_id = obj["object_row"]
            retained = row_id in retained_rows
            info = suppressed_info.get(row_id, {})

            catalogue_rows.append({
                "method": method,
                "object_row": row_id,
                "object_id": obj["object_id"],
                "event_epoch": obj["event_epoch"],
                "event_sign": obj["event_sign"],
                "corepoints": obj["size"],
                "retained": retained,
                "nested_in_row": info.get("parent_row", np.nan),
                "containment_fraction": info.get(
                    "containment", np.nan
                ),
                "iou_with_parent": info.get("iou", np.nan),
            })

            if retained:
                refined_objects.append(obj)

        refined_catalogues[method] = refined_objects

        def coverage_union(items):
            if not items:
                return np.array([], dtype=np.int64)

            return np.unique(
                np.concatenate([
                    item["indices"] for item in items
                ])
            )

        original_union = coverage_union(objects)
        refined_union = coverage_union(refined_objects)

        lost = np.setdiff1d(
            original_union,
            refined_union,
            assume_unique=True,
        )

        coverage_rows.append({
            "method": method,
            "original_objects": len(objects),
            "retained_objects": len(refined_objects),
            "suppressed_objects": (
                len(objects) - len(refined_objects)
            ),
            "original_unique_corepoints": len(original_union),
            "retained_unique_corepoints": len(refined_union),
            "lost_corepoints": len(lost),
            "coverage_preserved_pct": (
                100 * len(refined_union) / len(original_union)
                if len(original_union) else 100.0
            ),
        })

    return {
        "object_catalogues": object_catalogues,
        "refined_catalogues": refined_catalogues,
        "refined_catalogue_df": pd.DataFrame(catalogue_rows),
        "pairwise_overlap_df": pd.DataFrame(overlap_rows),
        "coverage_summary_df": pd.DataFrame(coverage_rows),
    }

# Descriptive thesis API aliases
apply_seed_nonmaximum_suppression = run_stage4f_seed_nms
extract_kalman_change_objects = run_stage4f_kf_spatial_growth


# ===== Reference seed detection (consolidated) =====
"""Reference-informed automatic temporal seeds for py4dgeo region growing.

Experimental subclass; does not modify installed py4dgeo source.
The detector follows the reference's gradient/decision-tree + fitted slope approach.
The spatial growth remains py4dgeo's own region_growing implementation.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from py4dgeo.segmentation import RegionGrowingAlgorithm, RegionGrowingSeed


class A5ReferenceRegionGrowing(RegionGrowingAlgorithm):
    def __init__(self, max_seed_epochs=12, max_seed_neighbor_dtw=0.2,
                 tree_min_samples_split=3, **kwargs):
        super().__init__(**kwargs)
        self.max_seed_epochs = int(max_seed_epochs)
        self.max_seed_neighbor_dtw = max_seed_neighbor_dtw
        self.tree_min_samples_split = int(tree_min_samples_split)
        self.seed_diagnostics = {}

    def find_seedpoints(self):
        data = self.analysis.distances_for_compute
        indices = (range(0, data.shape[0], self.seed_subsampling)
                   if self.seed_candidates is None else self.seed_candidates)
        seeds = []
        counts = dict(inspected=0, temporal=0, duration_rejected=0,
                      neighborhood_rejected=0, accepted=0)
        for idx in indices:
            idx = int(idx)
            counts['inspected'] += 1
            ts = np.asarray(data[idx], dtype=float).copy()
            good = np.isfinite(ts)
            if good.sum() < 3:
                continue
            x = np.arange(ts.size, dtype=float)
            if not good.all():
                ts[~good] = np.interp(x[~good], x[good], ts[good])
            gradient = np.gradient(ts, x)
            tree = DecisionTreeRegressor(min_samples_split=self.tree_min_samples_split,
                                         random_state=0)
            tree.fit(x[:, None], gradient)
            labels = tree.apply(x[:, None])
            # Leaf IDs preserve contiguous intervals; unlike unique predicted
            # gradient values, distinct leaves cannot accidentally be merged.
            for leaf in np.unique(labels):
                positions = np.flatnonzero(labels == leaf)
                if positions.size < 2:
                    continue
                first, last = int(positions[0]), int(positions[-1])
                if last - first < 1:
                    continue
                fit = LinearRegression().fit(x[positions, None], ts[positions])
                magnitude = float(fit.coef_[0] * (last - first))
                if abs(magnitude) < self.height_threshold:
                    continue
                start, stop = max(0, first - 1), min(ts.size - 1, last + 1)
                if start == 0 and stop == ts.size - 1:
                    continue
                if stop - start + 1 < self.minperiod:
                    continue
                counts['temporal'] += 1
                if stop - start + 1 > self.max_seed_epochs:
                    counts['duration_rejected'] += 1
                    continue
                seeds.append(RegionGrowingSeed(idx, start, stop))
        # The py4dgeo scorer uses its compiled normalized DTW metric. Its
        # neighbourhood mean is reference-inspired, not an exact raster test.
        if self.max_seed_neighbor_dtw is not None:
            score = self.seed_sorting_scorefunction()
            accepted = []
            for seed in seeds:
                value = float(score(seed))
                if np.isfinite(value) and value <= self.max_seed_neighbor_dtw:
                    accepted.append(seed)
                else:
                    counts['neighborhood_rejected'] += 1
            seeds = accepted
        counts['accepted'] = len(seeds)
        self.seed_diagnostics = counts
        print('Reference-style seed detection:', counts)
        return seeds


# ===== Reference seed ranking (consolidated) =====
"""Full-domain reference-informed 4D-OBC orchestration (experimental).

No changes to installed py4dgeo or Kalman source. Python 3.10+.
This module uses py4dgeo's compiled region-growing implementation.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from py4dgeo.segmentation import RegionGrowingAlgorithm, RegionGrowingSeed



def clone_archive(source, destination):
    """Copy an archive, removing only cached segmentation results.

    Raises if destination exists; never silently overwrites experiments.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f'Existing archive: {destination}; choose a new run name')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    try:
        with zipfile.ZipFile(source) as zin, zipfile.ZipFile(temporary, 'w') as zout:
            latest = {info.filename: info for info in zin.infolist()}
            for name, info in latest.items():
                if name in {'objects.pickle', 'seeds.pickle'}:
                    continue
                zout.writestr(info, zin.read(info))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def spatial_candidates(xyz, *, mode='grid', cell_size=2.0, max_candidates=None):
    """Deterministic spatial sampling covering the whole cloud's XY extent.

    mode='all' evaluates every corepoint. 'grid' chooses one point per occupied
    XY grid cell. A cap on grid representatives reduces coverage; report it.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError('Expected Nx3 or Nx2 coordinates')
    if mode == 'all':
        result = np.arange(len(xyz), dtype=np.int64)
    elif mode == 'grid':
        if cell_size <= 0:
            raise ValueError('cell_size must be positive')
        coords = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / cell_size).astype(np.int64)
        _, first = np.unique(coords, axis=0, return_index=True)
        result = np.sort(first)
    else:
        raise ValueError("mode must be 'grid' or 'all'")
    n_before_cap = len(result)
    if max_candidates is not None and len(result) > max_candidates:
        # Spatially systematic reduction, not a crop around A5.
        result = result[np.linspace(0, len(result) - 1, max_candidates, dtype=int)]
    return result.tolist(), {'total_corepoints':len(xyz), 'candidates_before_cap':n_before_cap,
                              'candidates_used':len(result), 'mode':mode,
                              'cell_size':cell_size if mode == 'grid' else None,
                              'capped':len(result) < n_before_cap}


def priority_order(seeds, xyz, signals, *, threshold=0.05, radius=1.0,
                   max_bins=None, per_bin=6, min_spacing=1.5):
    """Rank automatically found events without using a known A5 event time.

    Group by strongest signed adjacent-epoch change inside each seed interval;
    round-robin top spatially supported event/sign bins, then append remaining
    seeds. This is not the reference's original sorting criterion.
    """
    xyz = np.asarray(xyz); signals = np.asarray(signals)
    tree = cKDTree(xyz[:, :2]); rows=[]
    for seed in seeds:
        i, a, b = int(seed.index), int(seed.start_epoch), int(seed.end_epoch)
        a=max(0,a); b=min(signals.shape[1]-1,b)
        if b<=a: continue
        delta=np.diff(signals[i,a:b+1]); good=np.isfinite(delta)
        if not good.any(): continue
        offset=int(np.nanargmax(np.abs(delta))); event=a+offset
        jump=float(delta[offset])
        if abs(jump)<threshold: continue
        near=np.asarray(tree.query_ball_point(xyz[i,:2],r=radius),dtype=int)
        local=signals[near,event+1]-signals[near,event]
        support=int(np.sum(np.isfinite(local)&(np.sign(local)==np.sign(jump))&(np.abs(local)>=threshold)))
        rows.append((seed,i,int(seed.start_epoch),int(seed.end_epoch),event,
                     1 if jump>0 else -1,abs(jump),support))
    cols=['seed','index','start','stop','event','sign','abs_jump','support']
    df=pd.DataFrame(rows,columns=cols)
    if df.empty: return list(seeds),pd.DataFrame()
    scores=(df.groupby(['event','sign']).agg(n_seeds=('index','nunique'),
            max_support=('support','max'),max_jump=('abs_jump','max')).reset_index()
            .sort_values(['n_seeds','max_support','max_jump'],ascending=False))
    bins=scores if max_bins is None else scores.head(max_bins)
    chosen=[]; used=set()
    for eventrow in bins.itertuples():
        subset=df[(df.event==eventrow.event)&(df.sign==eventrow.sign)]
        selected_xy=[]
        for r in subset.sort_values(['support','abs_jump'],ascending=False).itertuples():
            key=(r.index,r.start,r.stop)
            if key in used: continue
            xy=xyz[r.index,:2]
            if any(np.linalg.norm(xy-p)<min_spacing for p in selected_xy):continue
            chosen.append(r.seed); used.add(key); selected_xy.append(xy)
            if len(selected_xy)>=per_bin:break
    chosen.extend(s for s in seeds if (int(s.index),int(s.start_epoch),int(s.end_epoch)) not in used)
    return chosen,scores


def object_inventory(objects, n_epochs, max_segments=None):
    rows=[]
    for k,obj in enumerate(objects or [],1):
        ix=np.asarray(obj.indices,dtype=np.int64)
        a,b=int(obj.start_epoch),int(obj.end_epoch)
        rows.append({'object':k,'unique_corepoints':len(np.unique(ix)),
                     'index_entries':len(ix),'start_epoch':a,'stop_epoch':b,
                     'duration_epochs':b-a+1,'valid_interval':0<=a<=b<n_epochs,
                     'at_or_above_cap':max_segments is not None and len(np.unique(ix))>=max_segments})
    return pd.DataFrame(rows,columns=['object','unique_corepoints','index_entries','start_epoch',
                                      'stop_epoch','duration_epochs','valid_interval','at_or_above_cap'])


def run_reference(analysis, candidates, *, stop_at_seed=101, max_segments=1500,
                  max_seed_neighbor_dtw=0.2):
    algo=A5ReferenceRegionGrowing(seed_candidates=candidates,
        tree_min_samples_split=3,max_seed_epochs=12,
        max_seed_neighbor_dtw=max_seed_neighbor_dtw,
        minperiod=2,height_threshold=0.05,neighborhood_radius=1.0,
        thresholds=[0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        min_segments=20,max_segments=max_segments,
        stop_at_seed=stop_at_seed,intermediate_saving=25,use_unfinished=False)
    return algo.run(analysis,force=False),algo


# ===== Thesis workflow orchestration (consolidated) =====
"""End-to-end Schneeferner thesis orchestration (new runs, original validated algorithms).

Install alongside the final notebooks in PROJECT/jupyter, and copy m3c2_kalman_full.py
into PROJECT/src/py4dgeo only if your local version differs. The original module
is preserved. Stage 4H.12 is executed from verbatim archived source with only
project/output directory assignments redirected to this run's folder.
"""
from pathlib import Path
import ast
import json
import re
import sys
import numpy as np
import pandas as pd

METHODS = ('original','kfmag','kfrate','kfmag_nms','kfrate_nms')
REFERENCES = ('A4_D','A4_E','A5_D','A5_E','A6_D','A6_E')

def project_paths(project, run_id):
    root = Path(project).resolve()
    if not (root / 'src' / 'py4dgeo').is_dir():
        raise FileNotFoundError(f'Expected editable py4dgeo at {root / "src" / "py4dgeo"}')
    if not run_id or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('RUN_ID must contain only letters, numbers, _ or -')
    run = root / 'results' / 'thesis_runs' / run_id
    return root,run

def add_import_paths(root):
    for directory in (root / 'src' / 'py4dgeo',root / 'jupyter'):
        if str(directory) not in sys.path:
            sys.path.insert(0,str(directory))

def prepare_raw_analysis(project, run_id, pointcloud_dir=None, smoothing_window=3,
                         smoothing_method="median", normal_radii=(1.0,),
                         cyl_radius=1.0, max_distance=10.0, registration_error=0.025):
    import py4dgeo, json, hashlib, numpy as np
    from datetime import datetime
    from scipy.ndimage import median_filter

    root, run = project_paths(project, run_id)
    folder = Path(pointcloud_dir) if pointcloud_dir else root/"schneeferner"/"pointclouds"
    paths = sorted(folder.glob("*.laz"))
    if len(paths) < 2: raise ValueError("At least 2 LAZ epochs required")
    times = [datetime.strptime("_".join(p.stem.split("_")[1:]), "%y%m%d_%H%M%S") for p in paths]
    if len(set(times)) != len(times) or times != sorted(times):
        raise ValueError("Invalid epoch timestamps")
    if smoothing_method != "median" or smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("Median smoothing requires a positive odd window")

    params = dict(
        input_files=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
        smoothing_window=smoothing_window, smoothing_method=smoothing_method,
        normal_radii=list(normal_radii), cyl_radius=cyl_radius,
        max_distance=max_distance, registration_error=registration_error
    )
    fingerprint = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    run.mkdir(parents=True, exist_ok=True)
    archive, config = run/"analysis.zip", run/"input_config.json"

    if archive.exists():
        if not config.exists() or json.loads(config.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("Cache parameters differ. Use a new RUN_ID.")
        print("Matching cache found — loading.")
        return py4dgeo.SpatiotemporalAnalysis(str(archive))
    if config.exists(): raise RuntimeError("Incomplete run configuration exists.")

    print("No matching cache — processing raw data.")
    analysis = py4dgeo.SpatiotemporalAnalysis(str(archive), force=True)
    reference = py4dgeo.read_from_las(str(paths[0]))
    reference.timestamp = times[0]
    analysis.reference_epoch = reference
    analysis.corepoints = reference.cloud
    analysis.m3c2 = py4dgeo.M3C2(
        normal_radii=tuple(normal_radii), cyl_radius=cyl_radius,
        max_distance=max_distance, registration_error=registration_error
    )
    for path, timestamp in zip(paths[1:], times[1:]):
        epoch = py4dgeo.read_from_las(str(path))
        epoch.timestamp = timestamp
        analysis.add_epochs(epoch)

    analysis.smoothed_distances = median_filter(
        np.asarray(analysis.distances, dtype=float),
        size=(1, smoothing_window), mode="nearest"
    )
    config.write_text(json.dumps({**params, "fingerprint": fingerprint}, indent=2))
    print("Analysis saved:", archive)
    return analysis
def load_analysis(project,run_id):
    import py4dgeo
    _,run=project_paths(project,run_id)
    path=run/'analysis.zip'
    if not path.is_file():raise FileNotFoundError(f'Run notebook 01 raw-data preparation first: {path}')
    return py4dgeo.SpatiotemporalAnalysis(str(path))

def compute_original_seeds(analysis, project, run_id, force=False):
    """Reproduce the historical full-domain 1 m grid seed detector and ranking.

    Detects from scratch when this run has no seeds; never imports historical
    seeds or checkpoints. Refuses ambiguous existing seed state.
    """
    import hashlib

    _, run = project_paths(project, run_id)
    run.mkdir(parents=True, exist_ok=True)
    config = run / "original_seed_config.json"
    params = dict(
        detector="A5ReferenceRegionGrowing",
        mode="grid", cell_size=1.0, max_candidates=None,
        tree_min_samples_split=3, max_seed_epochs=12,
        max_seed_neighbor_dtw=0.2, minperiod=2,
        height_threshold=0.05, neighborhood_radius=1.0,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_segments=20, max_segments=1500,
        ranking=dict(threshold=0.05, radius=1.0,
                     max_bins=None, per_bin=6),
    )
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    existing = len(analysis.seeds) if analysis.seeds is not None else 0
    if force:
        raise ValueError("For a fresh seed computation, use a new RUN_ID; force overwrite is disabled.")
    if existing or config.exists():
        if existing and config.exists():
            saved = json.loads(config.read_text(encoding="utf-8"))
            if saved.get("signature") == signature and saved.get("seed_count") == existing:
                print("Verified matching seed cache:", existing)
                return existing
        raise RuntimeError("Existing seed state cannot be verified. Use a new RUN_ID.")

    xyz = np.asarray(analysis.corepoints.cloud)
    candidates, coverage = spatial_candidates(
        xyz, mode="grid", cell_size=1.0, max_candidates=None
    )
    print("Candidate coverage:", coverage)
    algo = A5ReferenceRegionGrowing(
        seed_candidates=candidates,
        tree_min_samples_split=3, max_seed_epochs=12,
        max_seed_neighbor_dtw=0.2, minperiod=2,
        height_threshold=0.05, neighborhood_radius=1.0,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_segments=20, max_segments=1500,
        stop_at_seed=101, intermediate_saving=25,
        use_unfinished=False,
    )
    algo._analysis = analysis
    seeds = algo.find_seedpoints()
    print("Detected seeds before ranking:", len(seeds))
    ranked, events = priority_order(
        seeds, xyz, analysis.distances_for_compute,
        threshold=0.05, radius=1.0, max_bins=None, per_bin=6
    )
    if len(ranked) != len(seeds):
        raise RuntimeError("Seed ranking unexpectedly changed seed count")
    analysis.seeds = ranked
    events.to_csv(run / "original_seed_event_bins.csv", index=False)
    config.write_text(json.dumps({
        "signature": signature, "parameters": params,
        "coverage": coverage, "diagnostics": algo.seed_diagnostics,
        "seed_count": len(ranked), "event_bins": len(events)
    }, indent=2), encoding="utf-8")
    print("Original 4D-OBC seeds:", len(ranked), "event bins:", len(events))
    return len(ranked)

def run_kalman(analysis, project, run_id, chunk_size=256, resume=True):
    import json, hashlib
    root, run = project_paths(project, run_id)
    add_import_paths(root)

    params = dict(process_sigma=0.5, min_sigma_obs=0.005,
                  chunk_size=chunk_size, prefix="kalman_q05")
    config = run / "kalman_config.json"
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

    if config.exists():
        saved = json.loads(config.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError("Kalman parameters changed. Use a new RUN_ID.")
    else:
        config.write_text(json.dumps(
            {"signature": signature, "parameters": params}, indent=2
        ), encoding="utf-8")

    return run_full_domain_kalman_rts(
        analysis, run, prefix="kalman_q05",
        process_sigma=0.5, min_sigma_obs=0.005,
        chunk_size=chunk_size, resume=resume, verbose=True
    )

def run_extraction(analysis, project, run_id, resume=True):
    import json, hashlib
    root, run = project_paths(project, run_id)
    add_import_paths(root)

    if not (run/"kalman_q05_metadata.json").exists():
        raise FileNotFoundError("Kalman results missing. Complete Notebook 01 first.")
    if analysis.seeds is None or not len(analysis.seeds):
        raise ValueError("Original 4D-OBC seeds missing.")

    params = dict(radius=0.6, min_abs_change=0.02, max_visits=50000,
                  checkpoint_every=100, nms_radius=3.0,
                  nms_min_support_count=3, nms_min_support_fraction=0.30,
                  methods=["original", "kfmag", "kfrate", "kfmag_nms", "kfrate_nms"])
    config = run/"extraction_config.json"
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

    if config.exists():
        saved = json.loads(config.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError("Extraction parameters changed. Use a new RUN_ID.")
    else:
        config.write_text(json.dumps(
            {"signature": signature, "parameters": params}, indent=2
        ), encoding="utf-8")

    original = run_full_series(
        analysis, run, run_id="original_signed20",
        max_candidates=len(analysis.seeds), max_growth_calls=len(analysis.seeds),
        min_abs_change=0.02, radius=0.6, max_visits=50000,
        checkpoint_every=100, resume=resume, verbose=True
    )

    nms = run_stage4f_seed_nms(
        analysis, run, kalman_prefix="kalman_q05", seed_prefix="kf_seeds",
        nms_radius=3.0, nms_min_support_count=3,
        nms_min_support_fraction=0.30,
        max_seeds_per_mode=None, verbose=True
    )

    extracted = {}
    for method in ("kfmag", "kfrate", "kfmag_nms", "kfrate_nms"):
        extracted[method] = run_stage4f_kf_spatial_growth(
            analysis, run, method=method, seed_prefix="kf_seeds",
            kalman_prefix="kalman_q05", radius=0.6,
            min_abs_change=0.02, max_visits=50000,
            max_candidates=None, max_growth_calls=None,
            checkpoint_every=100, resume=resume, verbose=True
        )

    return {"original": original, "nms": nms, "kalman_methods": extracted}

def checkpoint_paths(run):
    run=Path(run)
    return {'original':run/'original_signed20_checkpoint.npz',
        **{m:run/f'stage4f_b_q05_{m}_v1'/'checkpoint.npz' for m in METHODS[1:]}}

def load_object_checkpoints(run):
    result={};metadata={}
    for method,path in checkpoint_paths(run).items():
        if not path.is_file():raise FileNotFoundError(f'Missing {method} checkpoint: {path}')
        with np.load(path,allow_pickle=False) as data:
            offsets=data['offsets'].astype(np.int64)
            members=data['indices' if method=='original' else 'members'].astype(np.int64)
            raw=data['state_utf8' if method=='original' else 'state']
            state=json.loads(raw.tobytes().decode('utf-8') if raw.dtype==np.uint8
                             else (raw.item() if raw.ndim==0 else bytes(raw).decode('utf-8')))
            rows=state['rows' if method=='original' else 'objects']
            if len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(members):
                raise ValueError(f'Invalid checkpoint offsets: {method}')
            result[method]=[{'row':i,'indices':np.unique(members[offsets[i]:offsets[i+1]]),
                'event_epoch':int(row['event_epoch']),'event_sign':int(row['event_sign'])}
                for i,row in enumerate(rows)]
            metadata[method]=rows
    return result,metadata

def refine_objects(project,run_id,analysis=None):
    root,run=project_paths(project,run_id);add_import_paths(root)
    objects,metadata=load_object_checkpoints(run)
    n_corepoints=len(analysis.corepoints.cloud) if analysis is not None else None
    refined=suppress_event_aware_duplicates(objects,containment_threshold=0.95,n_corepoints=n_corepoints)
    out=run/'refinement';out.mkdir(parents=True,exist_ok=True)
    for key,filename in [('refined_catalogue_df','refined_object_catalogue.csv'),
                         ('coverage_summary_df','coverage_preservation_summary.csv'),
                         ('pairwise_overlap_df','pairwise_overlap.csv')]:
        refined[key].to_csv(out/filename,index=False)
    return refined,metadata

def load_reference_masks(mask_npz,tolerance=0.35,references=REFERENCES):
    masks={}; suffix=f'__tol_{str(tolerance).replace(".","p")}'
    with np.load(mask_npz,allow_pickle=False) as data:
        for reference in references:
            key=reference.lower()+suffix
            if key not in data.files:raise KeyError(f'{mask_npz}: missing {key}')
            masks[reference]=np.unique(np.asarray(data[key],dtype=np.int64))
            if not len(masks[reference]):raise ValueError(f'Empty reference mask: {reference}')
    return masks

def load_alignment_csv(path):
    import ast as _ast
    df=pd.read_csv(path)
    for col in ('reference','candidate_transition_indices'):
        if col not in df:raise ValueError(f'Missing alignment column {col}')
    def parse(value):
        if isinstance(value,(list,tuple,np.ndarray)):return [int(x) for x in value]
        if isinstance(value,(int,np.integer)):return [int(value)]
        if pd.isna(value):return []
        value=str(value).strip()
        try: return [int(x) for x in _ast.literal_eval(value)]
        except (ValueError,SyntaxError,TypeError):
            return [int(x) for x in re.findall(r'-?\d+',value)]
    df['candidate_transition_indices']=df['candidate_transition_indices'].map(parse)
    return df

# Embedded, unchanged scientific evaluation scripts from validated Stage 4H.12.
# Only PROJECT_DIR and OUT_DIR are overridden for the current run.
_VALIDATED_STAGE4H12_SOURCES = {46: '\n# ============================================================\n# STAGE 4H.12A — SIX-REFERENCE OBJECT AND TEMPORAL INSPECTION\n# ============================================================\n#\n# References:\n#   A4_D, A4_E, A5_D, A5_E, A6_D, A6_E\n#\n# Methods:\n#   Original 4D-OBC\n#   KF magnitude\n#   KF rate\n#   KF magnitude + NMS\n#   KF rate + NMS\n#\n# PURPOSE:\n#   Identify eligible retained objects\n#   Inspect their spatial overlap and temporal intervals\n#   Verify reference event epochs\n#\n# NO segmentation rerun\n# NO duplicate-suppression rerun\n# NO checkpoint modifications\n# ============================================================\n\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\nfrom IPython.display import display\n\n# ------------------------------------------------------------\n# 1. CONFIGURATION\n# ------------------------------------------------------------\n\nPROJECT_DIR = Path(\n    r"C:\\Users\\tenbi\\py4dgeo-kalmanfiltering"\n)\n\nOUT_DIR = (\n    PROJECT_DIR\n    / "schneeferner"\n    / "stage4h_validation"\n    / "stage4h12_six_reference_validation"\n)\n\nOUT_DIR.mkdir(parents=True, exist_ok=True)\n\nMETHOD_ORDER = [\n    "original",\n    "kfmag",\n    "kfrate",\n    "kfmag_nms",\n    "kfrate_nms"\n]\n\nREFERENCE_ORDER = [\n    "A4_D",\n    "A4_E",\n    "A5_D",\n    "A5_E",\n    "A6_D",\n    "A6_E"\n]\n\nREFERENCE_SIGNS = {\n    "A4_D": 1,\n    "A4_E": -1,\n    "A5_D": 1,\n    "A5_E": -1,\n    "A6_D": 1,\n    "A6_E": -1\n}\n\n# Include any retained object that has at least\n# one corepoint inside the reference mask.\n#\n# This is an inclusive diagnostic selection rule.\n# It is NOT a final object-matching decision.\nMIN_REFERENCE_OVERLAP = 1\n\n# ------------------------------------------------------------\n# 2. VERIFY REQUIRED VARIABLES\n# ------------------------------------------------------------\n\nrequired = [\n    "reference_masks",\n    "refined_catalogues",\n    "metadata_by_method",\n    "alignment_df"\n]\n\nfor name in required:\n    if name not in globals():\n        raise RuntimeError(\n            f"Missing variable: {name}. "\n            "Run the earlier validation stages first."\n        )\n\n# ------------------------------------------------------------\n# 3. EXTRACT REFERENCE EVENT EPOCHS\n# ------------------------------------------------------------\n\nif not isinstance(alignment_df, pd.DataFrame):\n    raise TypeError(\n        "alignment_df must be a pandas DataFrame."\n    )\n\nrequired_alignment_columns = [\n    "reference",\n    "candidate_transition_indices"\n]\n\nmissing = [\n    col for col in required_alignment_columns\n    if col not in alignment_df.columns\n]\n\nif missing:\n    raise ValueError(\n        f"Missing alignment columns: {missing}"\n    )\n\ndef normalize_epoch_list(value):\n    """\n    Convert candidate transition indices into\n    a sorted list of integer epochs.\n    """\n\n    if isinstance(value, (list, tuple, np.ndarray)):\n        values = value\n\n    elif isinstance(value, (int, np.integer)):\n        values = [value]\n\n    else:\n        raise TypeError(\n            "Unexpected transition-index format: "\n            f"{type(value)}. "\n            "Inspect alignment_df before continuing."\n        )\n\n    return sorted(set(\n        int(v) for v in values\n    ))\n\n\nreference_epochs = {}\n\nfor reference in REFERENCE_ORDER:\n\n    matches = alignment_df.loc[\n        alignment_df["reference"] == reference\n    ]\n\n    if len(matches) != 1:\n        raise ValueError(\n            f"Expected one alignment row for "\n            f"{reference}; found {len(matches)}."\n        )\n\n    epochs = normalize_epoch_list(\n        matches.iloc[0][\n            "candidate_transition_indices"\n        ]\n    )\n\n    if not epochs:\n        raise ValueError(\n            f"No reference epochs for {reference}"\n        )\n\n    reference_epochs[reference] = epochs\n\nprint("=" * 75)\nprint("REFERENCE EVENT EPOCHS")\nprint("=" * 75)\n\nfor reference in REFERENCE_ORDER:\n    print(\n        f"{reference}: "\n        f"{reference_epochs[reference]}"\n    )\n\n# ------------------------------------------------------------\n# 4. BUILD OBJECT LOOKUPS\n# ------------------------------------------------------------\n\nobject_lookup_12a = {}\n\nfor method in METHOD_ORDER:\n\n    objects = refined_catalogues[method]\n\n    object_lookup_12a[method] = {}\n\n    for obj in objects:\n\n        row = int(obj["object_row"])\n\n        if row in object_lookup_12a[method]:\n            raise ValueError(\n                f"Duplicate object row {row} "\n                f"for method {method}"\n            )\n\n        object_lookup_12a[method][row] = obj\n\n# ------------------------------------------------------------\n# 5. IDENTIFY ELIGIBLE OBJECTS\n# ------------------------------------------------------------\n\nrecords = []\nsummary_records = []\n\nfor reference in REFERENCE_ORDER:\n\n    ref_idx = np.unique(\n        np.asarray(\n            reference_masks[reference],\n            dtype=np.int64\n        )\n    )\n\n    sign = REFERENCE_SIGNS[reference]\n    epochs = set(reference_epochs[reference])\n\n    for method in METHOD_ORDER:\n\n        candidates = []\n\n        for obj in refined_catalogues[method]:\n\n            event_epoch = int(obj["event_epoch"])\n            event_sign = int(obj["event_sign"])\n\n            # Require correct change direction.\n            if event_sign != sign:\n                continue\n\n            # Require matching reference transition.\n            if event_epoch not in epochs:\n                continue\n\n            idx = np.unique(\n                np.asarray(\n                    obj["indices"],\n                    dtype=np.int64\n                )\n            )\n\n            intersection = np.intersect1d(\n                idx,\n                ref_idx,\n                assume_unique=True\n            ).size\n\n            if intersection < MIN_REFERENCE_OVERLAP:\n                continue\n\n            union_size = (\n                len(idx)\n                + len(ref_idx)\n                - intersection\n            )\n\n            spatial_iou = (\n                intersection / union_size\n                if union_size else 0.0\n            )\n\n            recall = (\n                intersection / len(ref_idx)\n                if len(ref_idx) else 0.0\n            )\n\n            row = int(obj["object_row"])\n\n            metadata = metadata_by_method[method]\n\n            if row < 0 or row >= len(metadata):\n                raise IndexError(\n                    f"Metadata row {row} invalid "\n                    f"for {method}"\n                )\n\n            meta = metadata[row]\n\n            if method == "original":\n\n                start_epoch = meta.get(\n                    "start_epoch"\n                )\n\n                end_epoch = meta.get(\n                    "end_epoch"\n                )\n\n            else:\n\n                start_epoch = meta.get(\n                    "growth_start_epoch"\n                )\n\n                end_epoch = meta.get(\n                    "growth_end_epoch"\n                )\n\n            if start_epoch is None or end_epoch is None:\n                raise ValueError(\n                    f"Missing interval for "\n                    f"{method}, object row {row}"\n                )\n\n            start_epoch = int(start_epoch)\n            end_epoch = int(end_epoch)\n\n            if end_epoch <= start_epoch:\n                raise ValueError(\n                    f"Invalid interval for "\n                    f"{method}, object row {row}: "\n                    f"[{start_epoch}, {end_epoch})"\n                )\n\n            record = {\n                "method": method,\n                "reference": reference,\n                "object_row": row,\n                "object_id": row + 1,\n                "event_epoch": event_epoch,\n                "event_sign": event_sign,\n                "start_epoch": start_epoch,\n                "end_epoch": end_epoch,\n                "corepoints": len(idx),\n                "reference_overlap": intersection,\n                "spatial_iou": spatial_iou,\n                "reference_recall": recall\n            }\n\n            records.append(record)\n            candidates.append(record)\n\n        # --------------------------------------------\n        # Summarize the candidate set\n        # --------------------------------------------\n\n        if candidates:\n\n            candidate_rows = [\n                item["object_row"]\n                for item in candidates\n            ]\n\n            candidate_objects = [\n                object_lookup_12a[method][row]\n                for row in candidate_rows\n            ]\n\n            union_idx = np.unique(\n                np.concatenate([\n                    np.asarray(\n                        obj["indices"],\n                        dtype=np.int64\n                    )\n                    for obj in candidate_objects\n                ])\n            )\n\n            union_intersection = np.intersect1d(\n                union_idx,\n                ref_idx,\n                assume_unique=True\n            ).size\n\n            union_size = (\n                len(union_idx)\n                + len(ref_idx)\n                - union_intersection\n            )\n\n            union_iou = (\n                union_intersection / union_size\n                if union_size else 0.0\n            )\n\n            union_recall = (\n                union_intersection / len(ref_idx)\n                if len(ref_idx) else 0.0\n            )\n\n            best_single = max(\n                item["spatial_iou"]\n                for item in candidates\n            )\n\n        else:\n\n            union_idx = np.array([], dtype=np.int64)\n            union_iou = 0.0\n            union_recall = 0.0\n            best_single = 0.0\n\n        summary_records.append({\n            "method": method,\n            "reference": reference,\n            "reference_epochs": ",".join(\n                map(str, sorted(epochs))\n            ),\n            "eligible_objects": len(candidates),\n            "reference_corepoints": len(ref_idx),\n            "union_corepoints": len(union_idx),\n            "best_single_spatial_iou": best_single,\n            "candidate_union_spatial_iou": union_iou,\n            "candidate_union_recall": union_recall\n        })\n\n# ------------------------------------------------------------\n# 6. CREATE DATAFRAMES\n# ------------------------------------------------------------\n\nobjects_4h12a = pd.DataFrame(\n    records,\n    columns=[\n        "method",\n        "reference",\n        "object_row",\n        "object_id",\n        "event_epoch",\n        "event_sign",\n        "start_epoch",\n        "end_epoch",\n        "corepoints",\n        "reference_overlap",\n        "spatial_iou",\n        "reference_recall"\n    ]\n)\n\nsummary_4h12a = pd.DataFrame(\n    summary_records\n)\n\n# Sort in intended order\nmethod_rank = {\n    method: i\n    for i, method in enumerate(METHOD_ORDER)\n}\n\nreference_rank = {\n    reference: i\n    for i, reference in enumerate(REFERENCE_ORDER)\n}\n\nif not objects_4h12a.empty:\n\n    objects_4h12a["_method_rank"] = (\n        objects_4h12a["method"].map(method_rank)\n    )\n\n    objects_4h12a["_reference_rank"] = (\n        objects_4h12a["reference"].map(reference_rank)\n    )\n\n    objects_4h12a = (\n        objects_4h12a.sort_values(\n            [\n                "_reference_rank",\n                "_method_rank",\n                "spatial_iou"\n            ],\n            ascending=[True, True, False]\n        )\n        .drop(\n            columns=[\n                "_method_rank",\n                "_reference_rank"\n            ]\n        )\n        .reset_index(drop=True)\n    )\n\nsummary_4h12a["_method_rank"] = (\n    summary_4h12a["method"].map(method_rank)\n)\n\nsummary_4h12a["_reference_rank"] = (\n    summary_4h12a["reference"].map(reference_rank)\n)\n\nsummary_4h12a = (\n    summary_4h12a.sort_values(\n        ["_reference_rank", "_method_rank"]\n    )\n    .drop(\n        columns=["_method_rank", "_reference_rank"]\n    )\n    .reset_index(drop=True)\n)\n\n# ------------------------------------------------------------\n# 7. DISPLAY RESULTS\n# ------------------------------------------------------------\n\nprint("\\n" + "=" * 75)\nprint("STAGE 4H.12A — SIX-REFERENCE CANDIDATE SUMMARY")\nprint("=" * 75)\n\ndisplay(\n    summary_4h12a.round(4)\n)\n\nprint("\\n" + "=" * 75)\nprint("ELIGIBLE OBJECTS AND TEMPORAL INTERVALS")\nprint("=" * 75)\n\ndisplay(\n    objects_4h12a.round(4)\n)\n\n# ------------------------------------------------------------\n# 8. SAVE DIAGNOSTICS\n# ------------------------------------------------------------\n\nsummary_path = (\n    OUT_DIR\n    / "six_reference_candidate_summary.csv"\n)\n\nobjects_path = (\n    OUT_DIR\n    / "six_reference_object_intervals.csv"\n)\n\nsummary_4h12a.to_csv(\n    summary_path,\n    index=False\n)\n\nobjects_4h12a.to_csv(\n    objects_path,\n    index=False\n)\n\nprint("\\n" + "=" * 75)\nprint("STAGE 4H.12A COMPLETE")\nprint("=" * 75)\n\nprint("\\nSaved summary:")\nprint(summary_path)\n\nprint("\\nSaved object catalogue:")\nprint(objects_path)\n\nprint("\\nNo segmentation or suppression was rerun.")\n', 47: '\n# ============================================================\n# STAGE 4H.12B — SIX-REFERENCE EVENT-LEVEL VALIDATION\n# ============================================================\n#\n# References:\n#   A4_D, A4_E, A5_D, A5_E, A6_D, A6_E\n#\n# Methods:\n#   Original 4D-OBC\n#   KF magnitude\n#   KF rate\n#   KF magnitude + NMS\n#   KF rate + NMS\n#\n# Metrics:\n#   1. Best-single-object spatial IoU\n#   2. Candidate-union spatial IoU\n#   3. Temporal IoU\n#   4. Spatiotemporal IoU\n#\n# Reference temporal support:\n#   Each candidate transition k represents [k, k+1).\n#   Multiple candidate transitions are treated as a\n#   union of transition intervals, not necessarily a\n#   single continuous event.\n#\n# IMPORTANT:\n#   Results are provisional until candidate membership\n#   is reviewed.\n#\n# NO segmentation rerun\n# NO duplicate suppression rerun\n# NO checkpoint modifications\n# ============================================================\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom pathlib import Path\nfrom IPython.display import display\n\n# ------------------------------------------------------------\n# 1. VERIFY REQUIRED INPUTS\n# ------------------------------------------------------------\n\nrequired = [\n    "objects_4h12a",\n    "summary_4h12a",\n    "reference_masks",\n    "reference_epochs",\n    "object_lookup_12a"\n]\n\nfor name in required:\n    if name not in globals():\n        raise RuntimeError(\n            f"Missing variable: {name}. "\n            "Run Stage 4H.12A first."\n        )\n\n# ------------------------------------------------------------\n# 2. CONFIGURATION\n# ------------------------------------------------------------\n\nPROJECT_DIR = Path(\n    r"C:\\Users\\tenbi\\py4dgeo-kalmanfiltering"\n)\n\nOUT_DIR = (\n    PROJECT_DIR\n    / "schneeferner"\n    / "stage4h_validation"\n    / "stage4h12_six_reference_validation"\n    / "publication_ready"\n)\n\nOUT_DIR.mkdir(parents=True, exist_ok=True)\n\nMETHOD_ORDER = [\n    "original",\n    "kfmag",\n    "kfrate",\n    "kfmag_nms",\n    "kfrate_nms"\n]\n\nMETHOD_LABELS = [\n    "Original\\n4D-OBC",\n    "KF\\nmagnitude",\n    "KF\\nrate",\n    "KF magnitude\\n+ NMS",\n    "KF rate\\n+ NMS"\n]\n\nREFERENCE_ORDER = [\n    "A4_D",\n    "A4_E",\n    "A5_D",\n    "A5_E",\n    "A6_D",\n    "A6_E"\n]\n\nREFERENCE_TITLES = {\n    "A4_D": "A4_D — Deposition (+)",\n    "A4_E": "A4_E — Erosion (−)",\n    "A5_D": "A5_D — Deposition (+)",\n    "A5_E": "A5_E — Erosion (−)",\n    "A6_D": "A6_D — Deposition (+)",\n    "A6_E": "A6_E — Erosion (−)"\n}\n\nMETRIC_COLORS = {\n    "spatial": "#0072B2",\n    "temporal": "#009E73",\n    "spatiotemporal": "#E69F00"\n}\n\n# ------------------------------------------------------------\n# 3. METRIC FUNCTIONS\n# ------------------------------------------------------------\n\ndef iou_sets(a, b):\n\n    a = set(a)\n    b = set(b)\n\n    union = len(a | b)\n\n    if union == 0:\n        return 0.0\n\n    return len(a & b) / union\n\n\ndef epoch_support(start, end):\n\n    start = int(start)\n    end = int(end)\n\n    if end <= start:\n        raise ValueError(\n            f"Invalid interval [{start}, {end})"\n        )\n\n    return set(range(start, end))\n\n\ndef event_space_time_iou(\n    objects,\n    reference_indices,\n    reference_transitions\n):\n    """\n    Construct predicted spatial support per epoch.\n\n    Reference:\n        Fixed spatial footprint during each candidate\n        reference transition.\n\n    Prediction:\n        Object footprint constant throughout its\n        reported growth interval.\n\n    This is a space-time support approximation.\n    """\n\n    predicted_by_epoch = {}\n\n    for obj in objects:\n\n        indices = set(\n            np.asarray(\n                obj["indices"],\n                dtype=np.int64\n            ).tolist()\n        )\n\n        active_epochs = epoch_support(\n            obj["start_epoch"],\n            obj["end_epoch"]\n        )\n\n        for epoch in active_epochs:\n\n            predicted_by_epoch.setdefault(\n                epoch,\n                set()\n            ).update(indices)\n\n    reference_set = set(\n        np.asarray(\n            reference_indices,\n            dtype=np.int64\n        ).tolist()\n    )\n\n    all_epochs = (\n        set(predicted_by_epoch)\n        | set(reference_transitions)\n    )\n\n    intersection_total = 0\n    union_total = 0\n\n    for epoch in all_epochs:\n\n        predicted = predicted_by_epoch.get(\n            epoch,\n            set()\n        )\n\n        reference = (\n            reference_set\n            if epoch in reference_transitions\n            else set()\n        )\n\n        intersection_total += len(\n            predicted & reference\n        )\n\n        union_total += len(\n            predicted | reference\n        )\n\n    return (\n        intersection_total / union_total\n        if union_total\n        else 0.0\n    )\n\n# ------------------------------------------------------------\n# 4. CALCULATE SIX-REFERENCE METRICS\n# ------------------------------------------------------------\n\nrecords = []\ncandidate_diagnostics = []\n\nfor reference in REFERENCE_ORDER:\n\n    ref_idx = np.unique(\n        np.asarray(\n            reference_masks[reference],\n            dtype=np.int64\n        )\n    )\n\n    ref_set = set(ref_idx.tolist())\n\n    ref_transitions = set(\n        int(e)\n        for e in reference_epochs[reference]\n    )\n\n    for method in METHOD_ORDER:\n\n        group = objects_4h12a.loc[\n            (objects_4h12a["method"] == method)\n            &\n            (objects_4h12a["reference"] == reference)\n        ]\n\n        selected_objects = []\n\n        for _, row in group.iterrows():\n\n            object_row = int(\n                row["object_row"]\n            )\n\n            if object_row not in object_lookup_12a[method]:\n                raise KeyError(\n                    f"Object row {object_row} "\n                    f"not found for {method}"\n                )\n\n            obj = object_lookup_12a[method][object_row]\n\n            idx = np.unique(\n                np.asarray(\n                    obj["indices"],\n                    dtype=np.int64\n                )\n            )\n\n            selected_objects.append({\n                "object_row": object_row,\n                "object_id": int(row["object_id"]),\n                "indices": idx,\n                "start_epoch": int(row["start_epoch"]),\n                "end_epoch": int(row["end_epoch"]),\n                "event_epoch": int(row["event_epoch"])\n            })\n\n            overlap = len(\n                set(idx.tolist()) & ref_set\n            )\n\n            candidate_diagnostics.append({\n                "reference": reference,\n                "method": method,\n                "object_id": int(row["object_id"]),\n                "object_row": object_row,\n                "corepoints": len(idx),\n                "reference_overlap": overlap,\n                "reference_recall": (\n                    overlap / len(ref_set)\n                    if ref_set else 0.0\n                ),\n                "object_precision": (\n                    overlap / len(idx)\n                    if len(idx) else 0.0\n                ),\n                "spatial_iou": iou_sets(\n                    idx,\n                    ref_idx\n                ),\n                "start_epoch": int(row["start_epoch"]),\n                "end_epoch": int(row["end_epoch"])\n            })\n\n        # --------------------------------------------\n        # Spatial union\n        # --------------------------------------------\n\n        if selected_objects:\n\n            union_idx = np.unique(\n                np.concatenate([\n                    obj["indices"]\n                    for obj in selected_objects\n                ])\n            )\n\n        else:\n\n            union_idx = np.array(\n                [],\n                dtype=np.int64\n            )\n\n        union_spatial_iou = iou_sets(\n            union_idx,\n            ref_idx\n        )\n\n        best_single_iou = max(\n            [\n                iou_sets(\n                    obj["indices"],\n                    ref_idx\n                )\n                for obj in selected_objects\n            ],\n            default=0.0\n        )\n\n        # --------------------------------------------\n        # Temporal support\n        # --------------------------------------------\n\n        predicted_epochs = set()\n\n        for obj in selected_objects:\n\n            predicted_epochs.update(\n                epoch_support(\n                    obj["start_epoch"],\n                    obj["end_epoch"]\n                )\n            )\n\n        temporal_iou = iou_sets(\n            predicted_epochs,\n            ref_transitions\n        )\n\n        # --------------------------------------------\n        # Spatiotemporal IoU\n        # --------------------------------------------\n\n        st_iou = event_space_time_iou(\n            selected_objects,\n            ref_idx,\n            ref_transitions\n        )\n\n        # --------------------------------------------\n        # Store metrics\n        # --------------------------------------------\n\n        records.append({\n            "reference": reference,\n            "method": method,\n            "eligible_objects": len(selected_objects),\n            "reference_corepoints": len(ref_idx),\n            "union_corepoints": len(union_idx),\n            "best_single_spatial_iou": best_single_iou,\n            "union_spatial_iou": union_spatial_iou,\n            "temporal_iou": temporal_iou,\n            "spatiotemporal_iou": st_iou,\n            "spatial_iou_gain": (\n                union_spatial_iou\n                - best_single_iou\n            ),\n            "reference_transitions": ",".join(\n                map(str, sorted(ref_transitions))\n            ),\n            "predicted_epochs": ",".join(\n                map(str, sorted(predicted_epochs))\n            )\n        })\n\n# ------------------------------------------------------------\n# 5. RESULTS DATAFRAMES\n# ------------------------------------------------------------\n\nresults_4h12b = pd.DataFrame(records)\n\ncandidate_diagnostics_4h12b = pd.DataFrame(\n    candidate_diagnostics\n)\n\nprint("=" * 75)\nprint("STAGE 4H.12B — SIX-REFERENCE EVENT-LEVEL METRICS")\nprint("=" * 75)\n\ndisplay(\n    results_4h12b.round(4)\n)\n\nprint("\\nCANDIDATE OBJECT DIAGNOSTICS")\n\ndisplay(\n    candidate_diagnostics_4h12b.round(4)\n)\n\n# ------------------------------------------------------------\n# 6. SIX-PANEL THREE-METRIC FIGURE\n# ------------------------------------------------------------\n\nplt.rcParams.update({\n    "font.family": "DejaVu Sans",\n    "pdf.fonttype": 42,\n    "ps.fonttype": 42\n})\n\nfig, axes = plt.subplots(\n    3,\n    2,\n    figsize=(16, 13),\n    sharey=True\n)\n\nx = np.arange(\n    len(METHOD_ORDER)\n)\n\nwidth = 0.25\n\nmetrics = [\n    (\n        "union_spatial_iou",\n        "Spatial IoU",\n        METRIC_COLORS["spatial"]\n    ),\n    (\n        "temporal_iou",\n        "Temporal IoU",\n        METRIC_COLORS["temporal"]\n    ),\n    (\n        "spatiotemporal_iou",\n        "Spatiotemporal IoU",\n        METRIC_COLORS["spatiotemporal"]\n    )\n]\n\nfor ax, reference in zip(\n    axes.flat,\n    REFERENCE_ORDER\n):\n\n    subset = (\n        results_4h12b.loc[\n            results_4h12b["reference"] == reference\n        ]\n        .set_index("method")\n        .loc[METHOD_ORDER]\n    )\n\n    for j, (column, label, color) in enumerate(\n        metrics\n    ):\n\n        values = subset[\n            column\n        ].to_numpy(dtype=float)\n\n        offset = (\n            j - 1\n        ) * width\n\n        bars = ax.bar(\n            x + offset,\n            values,\n            width=width * 0.92,\n            color=color,\n            edgecolor="white",\n            linewidth=0.4,\n            label=label,\n            zorder=3\n        )\n\n        for bar, value in zip(\n            bars,\n            values\n        ):\n\n            if value <= 0:\n                continue\n\n            ax.annotate(\n                f"{value:.2f}",\n                xy=(\n                    bar.get_x()\n                    + bar.get_width() / 2,\n                    value\n                ),\n                xytext=(0, 3),\n                textcoords="offset points",\n                ha="center",\n                va="bottom",\n                fontsize=7\n            )\n\n    ax.set_title(\n        REFERENCE_TITLES[reference],\n        fontsize=12,\n        fontweight="bold",\n        pad=12\n    )\n\n    ax.set_xticks(x)\n\n    ax.set_xticklabels(\n        METHOD_LABELS,\n        fontsize=8.5\n    )\n\n    ax.set_ylim(0, 1.12)\n\n    ax.set_yticks(\n        np.arange(0, 1.01, 0.2)\n    )\n\n    ax.grid(\n        axis="y",\n        linestyle="--",\n        alpha=0.25,\n        linewidth=0.6,\n        zorder=0\n    )\n\n    ax.spines["top"].set_visible(False)\n    ax.spines["right"].set_visible(False)\n\nfor ax in axes[:, 0]:\n\n    ax.set_ylabel(\n        "Intersection over Union (IoU)",\n        fontsize=10\n    )\n\nfig.suptitle(\n    "Schneeferner — Six-Reference Event-Level "\n    "Spatial, Temporal and Spatiotemporal IoU",\n    fontsize=16,\n    fontweight="bold",\n    y=0.985\n)\n\nhandles, labels = (\n    axes.flat[0].get_legend_handles_labels()\n)\n\nfig.legend(\n    handles,\n    labels,\n    loc="lower center",\n    ncol=3,\n    frameon=False,\n    fontsize=11,\n    bbox_to_anchor=(0.5, 0.025)\n)\n\nfig.subplots_adjust(\n    left=0.07,\n    right=0.98,\n    top=0.93,\n    bottom=0.10,\n    hspace=0.43,\n    wspace=0.16\n)\n\npdf_path = (\n    OUT_DIR\n    / "six_reference_three_metric_comparison.pdf"\n)\n\npng_path = (\n    OUT_DIR\n    / "six_reference_three_metric_comparison.png"\n)\n\nfig.savefig(\n    pdf_path,\n    format="pdf",\n    bbox_inches="tight",\n    facecolor="white"\n)\n\nfig.savefig(\n    png_path,\n    format="png",\n    dpi=300,\n    bbox_inches="tight",\n    facecolor="white"\n)\n\nplt.show()\n\n# ------------------------------------------------------------\n# 7. METHOD-LEVEL SUMMARY\n# ------------------------------------------------------------\n\nmethod_summary_4h12b = (\n    results_4h12b.groupby(\n        "method",\n        as_index=False\n    )\n    .agg(\n        mean_spatial_iou=(\n            "union_spatial_iou",\n            "mean"\n        ),\n        mean_temporal_iou=(\n            "temporal_iou",\n            "mean"\n        ),\n        mean_spatiotemporal_iou=(\n            "spatiotemporal_iou",\n            "mean"\n        ),\n        references_with_objects=(\n            "eligible_objects",\n            lambda x: int(\n                (x > 0).sum()\n            )\n        )\n    )\n)\n\nmethod_summary_4h12b["method_order"] = (\n    method_summary_4h12b["method"].map({\n        method: i\n        for i, method in enumerate(METHOD_ORDER)\n    })\n)\n\nmethod_summary_4h12b = (\n    method_summary_4h12b.sort_values(\n        "method_order"\n    )\n    .drop(columns="method_order")\n    .reset_index(drop=True)\n)\n\nprint("\\n" + "=" * 75)\nprint("METHOD-LEVEL SUMMARY ACROSS SIX REFERENCES")\nprint("=" * 75)\n\ndisplay(\n    method_summary_4h12b.round(4)\n)\n\n# ------------------------------------------------------------\n# 8. SAVE RESULTS\n# ------------------------------------------------------------\n\nresults_path = (\n    OUT_DIR\n    / "six_reference_event_level_metrics.csv"\n)\n\ndiagnostics_path = (\n    OUT_DIR\n    / "six_reference_candidate_diagnostics.csv"\n)\n\nsummary_path = (\n    OUT_DIR\n    / "six_reference_method_summary.csv"\n)\n\nresults_4h12b.to_csv(\n    results_path,\n    index=False\n)\n\ncandidate_diagnostics_4h12b.to_csv(\n    diagnostics_path,\n    index=False\n)\n\nmethod_summary_4h12b.to_csv(\n    summary_path,\n    index=False\n)\n\nprint("\\n" + "=" * 75)\nprint("STAGE 4H.12B COMPLETE")\nprint("=" * 75)\n\nprint("\\nSaved PDF:")\nprint(pdf_path)\n\nprint("\\nSaved PNG:")\nprint(png_path)\n\nprint("\\nSaved metrics:")\nprint(results_path)\n\nprint("\\nSaved candidate diagnostics:")\nprint(diagnostics_path)\n\nprint("\\nSaved method summary:")\nprint(summary_path)\n\nprint("\\nOriginal checkpoints remain unchanged.")\n'}


def _execute_validated_cell(cell_number, inputs, project, out_dir):
    # Evaluate the validated code from this module; no additional .py files required.
    source = _VALIDATED_STAGE4H12_SOURCES[cell_number]
    tree = ast.parse(source, filename=f"embedded_stage4h12_cell{cell_number}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PROJECT_DIR", "OUT_DIR"):
                node.value = ast.Name(
                    id="_project_override" if name == "PROJECT_DIR" else "_out_override",
                    ctx=ast.Load(),
                )
    ast.fix_missing_locations(tree)
    ns = dict(inputs, _project_override=Path(project), _out_override=Path(out_dir))
    exec(compile(tree, f"embedded_stage4h12_cell{cell_number}", "exec"), ns)
    return ns


def validate_six_references(project,run_id,mask_npz,alignment_csv,analysis=None,apply_duplicate_suppression=False):
    root,run=project_paths(project,run_id)
    if analysis is None: analysis=load_analysis(project,run_id)

    if apply_duplicate_suppression:
        refined,metadata=refine_objects(project,run_id,analysis)
        catalogues=refined['refined_catalogues']
    else:
        objects,metadata=load_object_checkpoints(run)
        catalogues={m:[dict(object_row=i,object_id=i+1,
            indices=np.unique(o['indices']),event_epoch=int(o['event_epoch']),
            event_sign=int(o['event_sign']),size=len(o['indices']))
            for i,o in enumerate(v)] for m,v in objects.items()}

    masks=load_reference_masks(mask_npz)
    alignment=load_alignment_csv(alignment_csv)
    if set(alignment['reference'])!=set(REFERENCES):
        raise ValueError('Expected exactly six reference events')

    inputs=dict(reference_masks=masks,refined_catalogues=catalogues,
                metadata_by_method=metadata,alignment_df=alignment)
    mode='with_suppression' if apply_duplicate_suppression else 'without_suppression'
    out=run/'validation'/mode
    a=_execute_validated_cell(46,inputs,root,out/'stage4h12_candidates')
    b=_execute_validated_cell(47,a,root,out/'publication_ready')

    print('Duplicate suppression:',apply_duplicate_suppression)
    print('Objects:',{m:len(v) for m,v in catalogues.items()})
    return dict(event_metrics=b['results_4h12b'],
                method_summary=b['method_summary_4h12b'],
                candidate_diagnostics=b['candidate_diagnostics_4h12b'],
                output_dir=out/'publication_ready')
