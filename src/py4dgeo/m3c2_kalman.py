# ============================================================
# m3c2_kalman.py
# KALMAN FILTERING FOR M3C2 TIME SERIES
# ============================================================

import numpy as np
from filterpy.kalman import KalmanFilter


def _timedeltas_to_days(timedeltas):
    """
    Convert py4dgeo timedeltas to elapsed days and derive
    epoch-to-epoch time steps.

    Parameters
    ----------
    timedeltas : sequence
        py4dgeo timedelta objects.

    Returns
    -------
    dt_series : np.ndarray
        Time-step lengths in days.
    """

    times_days = np.array([
        td.total_seconds() / (60.0 * 60.0 * 24.0)
        for td in timedeltas
    ], dtype=float)

    if len(times_days) == 0:
        raise ValueError("timedeltas must not be empty.")

    dt_series = np.diff(
        times_days,
        prepend=times_days[0]
    )

    if len(dt_series) > 1:
        dt_series[0] = dt_series[1]
    else:
        dt_series[0] = 1.0

    return dt_series


def kalman_smooth_m3c2_series(
    z,
    sigma_obs,
    dt_series,
    process_sigma=0.01
):
    """
    Apply Kalman filtering and RTS smoothing to one M3C2
    change time series.

    State vector
    ------------
    x = [change, change_rate]

    Parameters
    ----------
    z : array-like
        M3C2 distance/change time series for one corepoint.

    sigma_obs : array-like
        Observation standard deviation for each epoch.

    dt_series : array-like
        Time-step lengths in days.

    process_sigma : float, default=0.01
        Process-noise standard deviation controlling the
        smoothness/responsiveness of the model.

    Returns
    -------
    smooth_change : np.ndarray
        Kalman-smoothed change magnitude.

    smooth_rate : np.ndarray
        Kalman-estimated change rate.

    smooth_sigma_change : np.ndarray
        Standard deviation of smoothed change.

    smooth_sigma_rate : np.ndarray
        Standard deviation of estimated rate.
    """

    z = np.asarray(z, dtype=float)
    sigma_obs = np.asarray(
        sigma_obs,
        dtype=float
    )
    dt_series = np.asarray(
        dt_series,
        dtype=float
    )

    if z.ndim != 1:
        raise ValueError(
            "z must be a one-dimensional time series."
        )

    if sigma_obs.shape != z.shape:
        raise ValueError(
            "sigma_obs must have the same shape as z."
        )

    if dt_series.shape != z.shape:
        raise ValueError(
            "dt_series must have the same length as z."
        )

    n = len(z)

    xs = np.full(
        (n, 2),
        np.nan,
        dtype=float
    )

    Ps = np.full(
        (n, 2, 2),
        np.nan,
        dtype=float
    )

    valid = np.isfinite(z)

    if np.sum(valid) < 2:
        empty = np.full(
            n,
            np.nan,
            dtype=float
        )

        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy()
        )

    first_valid = np.where(valid)[0][0]

    kf = KalmanFilter(
        dim_x=2,
        dim_z=1
    )

    kf.x = np.array([
        z[first_valid],
        0.0
    ])

    kf.P = np.eye(2) * 1.0

    kf.H = np.array([
        [1.0, 0.0]
    ])

    # --------------------------------------------------------
    # Forward Kalman filter
    # --------------------------------------------------------

    for t in range(n):

        dt = float(
            dt_series[t]
        )

        kf.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        q = process_sigma ** 2

        kf.Q = np.array([
            [
                0.25 * dt**4,
                0.5 * dt**3
            ],
            [
                0.5 * dt**3,
                dt**2
            ]
        ]) * q

        kf.predict()

        if np.isfinite(z[t]):

            kf.R = np.array([
                [
                    sigma_obs[t] ** 2
                ]
            ])

            kf.update(
                np.array([
                    z[t]
                ])
            )

        xs[t] = kf.x
        Ps[t] = kf.P

    # --------------------------------------------------------
    # Rauch-Tung-Striebel smoother
    # --------------------------------------------------------

    x_smooth = xs.copy()
    P_smooth = Ps.copy()

    for t in range(
        n - 2,
        -1,
        -1
    ):

        dt = float(
            dt_series[t + 1]
        )

        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        q = process_sigma ** 2

        Q = np.array([
            [
                0.25 * dt**4,
                0.5 * dt**3
            ],
            [
                0.5 * dt**3,
                dt**2
            ]
        ]) * q

        P_pred = (
            F
            @ Ps[t]
            @ F.T
            + Q
        )

        try:

            C = (
                Ps[t]
                @ F.T
                @ np.linalg.inv(
                    P_pred
                )
            )

            x_smooth[t] = (
                xs[t]
                + C
                @ (
                    x_smooth[t + 1]
                    - F @ xs[t]
                )
            )

            P_smooth[t] = (
                Ps[t]
                + C
                @ (
                    P_smooth[t + 1]
                    - P_pred
                )
                @ C.T
            )

        except np.linalg.LinAlgError:
            continue

    smooth_change = (
        x_smooth[:, 0]
    )

    smooth_rate = (
        x_smooth[:, 1]
    )

    smooth_sigma_change = np.sqrt(
        np.clip(
            P_smooth[:, 0, 0],
            0,
            None
        )
    )

    smooth_sigma_rate = np.sqrt(
        np.clip(
            P_smooth[:, 1, 1],
            0,
            None
        )
    )

    return (
        smooth_change,
        smooth_rate,
        smooth_sigma_change,
        smooth_sigma_rate
    )


def kalman_filter_m3c2(
    distances,
    lodetection,
    timedeltas,
    process_sigma=0.01,
    min_sigma_obs=0.005,
    corepoint_indices=None,
    progress_callback=None
):
    """
    Apply the Kalman + RTS smoother to multiple M3C2
    corepoint time series.

    Parameters
    ----------
    distances : np.ndarray
        Shape:
            n_corepoints x n_epochs

        Raw M3C2 distance/change time series.

    lodetection : np.ndarray
        Detection-limit array associated with the M3C2 results.

    timedeltas : sequence
        py4dgeo time differences.

    process_sigma : float, default=0.01
        Process-noise standard deviation.

    min_sigma_obs : float, default=0.005
        Minimum observation standard deviation.

    corepoint_indices : iterable or None
        Corepoints to process.

        If None, all corepoints are processed.

    progress_callback : callable or None
        Optional callback:

            progress_callback(
                processed,
                total
            )

        This keeps printing/progress behaviour outside the
        scientific implementation.

    Returns
    -------
    dict
        {
            "change": ...,
            "rate": ...,
            "sigma_change": ...,
            "sigma_rate": ...,
            "observation_sigma": ...,
            "dt_days": ...,
            "processed_indices": ...
        }
    """

    distances = np.asarray(
        distances,
        dtype=float
    )

    lodetection = np.asarray(
        lodetection,
        dtype=float
    )

    if distances.ndim != 2:
        raise ValueError(
            "distances must have shape "
            "(n_corepoints, n_epochs)."
        )

    n_corepoints, n_epochs = (
        distances.shape
    )

    # --------------------------------------------------------
    # Observation uncertainty
    # --------------------------------------------------------

    sigma_obs = np.maximum(
        lodetection / 1.96,
        min_sigma_obs
    )

    try:
        sigma_obs = np.broadcast_to(
            sigma_obs,
            distances.shape
        ).copy()

    except ValueError as exc:
        raise ValueError(
            "lodetection must be broadcastable to the "
            "shape of distances."
        ) from exc

    # --------------------------------------------------------
    # Time intervals
    # --------------------------------------------------------

    dt_series = _timedeltas_to_days(
        timedeltas
    )

    if len(dt_series) != n_epochs:
        raise ValueError(
            "Number of timedeltas does not match "
            "the number of epochs."
        )

    # --------------------------------------------------------
    # Corepoint selection
    # --------------------------------------------------------

    if corepoint_indices is None:

        processed_indices = np.arange(
            n_corepoints,
            dtype=int
        )

    else:

        processed_indices = np.asarray(
            list(corepoint_indices),
            dtype=int
        )

        if processed_indices.ndim != 1:
            raise ValueError(
                "corepoint_indices must be one-dimensional."
            )

        if np.any(
            processed_indices < 0
        ) or np.any(
            processed_indices >= n_corepoints
        ):
            raise IndexError(
                "corepoint_indices contains an index "
                "outside the available corepoint range."
            )

    # --------------------------------------------------------
    # Allocate complete output arrays
    # --------------------------------------------------------

    kalman_change = np.full_like(
        distances,
        np.nan,
        dtype=float
    )

    kalman_rate = np.full_like(
        distances,
        np.nan,
        dtype=float
    )

    kalman_sigma_change = np.full_like(
        distances,
        np.nan,
        dtype=float
    )

    kalman_sigma_rate = np.full_like(
        distances,
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # Process selected corepoints
    # --------------------------------------------------------

    total = len(
        processed_indices
    )

    for counter, cp_idx in enumerate(
        processed_indices,
        start=1
    ):

        (
            kalman_change[cp_idx],
            kalman_rate[cp_idx],
            kalman_sigma_change[cp_idx],
            kalman_sigma_rate[cp_idx]
        ) = kalman_smooth_m3c2_series(
            z=distances[cp_idx],
            sigma_obs=sigma_obs[cp_idx],
            dt_series=dt_series,
            process_sigma=process_sigma
        )

        if progress_callback is not None:

            progress_callback(
                counter,
                total
            )

    return {
        "change": kalman_change,
        "rate": kalman_rate,
        "sigma_change": (
            kalman_sigma_change
        ),
        "sigma_rate": (
            kalman_sigma_rate
        ),
        "observation_sigma": (
            sigma_obs
        ),
        "dt_days": (
            dt_series
        ),
        "processed_indices": (
            processed_indices
        )
    }


# ============================================================
# KALMAN-BASED SEED DETECTION
# ============================================================

import pandas as pd


def _extract_true_intervals(boolean_series):
    """
    Extract continuous True intervals from a boolean sequence.

    Parameters
    ----------
    boolean_series : array-like of bool
        Boolean activity mask.

    Returns
    -------
    list of tuple
        Each tuple is:
            (start_epoch, end_epoch)
    """

    boolean_series = np.asarray(
        boolean_series,
        dtype=bool
    )

    active = np.where(
        boolean_series
    )[0]

    if len(active) == 0:
        return []

    intervals = []

    start = int(
        active[0]
    )

    previous = int(
        active[0]
    )

    for t in active[1:]:

        t = int(t)

        if t == previous + 1:

            previous = t

        else:

            intervals.append(
                (
                    start,
                    previous
                )
            )

            start = t
            previous = t

    intervals.append(
        (
            start,
            previous
        )
    )

    return intervals


def _find_formation_start(
    change,
    change_sigma,
    raw_start_epoch,
    direction,
    earliest_allowed_epoch,
    baseline_tolerance=0.02,
    baseline_sigma_factor=1.0,
    max_extension_epochs=45,
    min_extension_slope=0.001
):
    """
    Estimate the formation-start epoch before the first
    statistically significant magnitude epoch.

    The method searches backwards from raw_start_epoch for the
    latest near-baseline epoch while preventing extension into
    an unrelated opposite-sign event.

    Parameters
    ----------
    change : np.ndarray
        Kalman-smoothed change series.

    change_sigma : np.ndarray
        Standard deviation of Kalman-smoothed change.

    raw_start_epoch : int
        First statistically significant epoch.

    direction : str
        "positive" or "negative".

    earliest_allowed_epoch : int
        Earliest epoch allowed for backward extension.

    baseline_tolerance : float
        Absolute near-zero baseline tolerance.

    baseline_sigma_factor : float
        Multiplier applied to change uncertainty when determining
        the baseline envelope.

    max_extension_epochs : int
        Maximum number of epochs to search backwards.

    min_extension_slope : float
        Small tolerance used by the trend-consistency guard.

    Returns
    -------
    int
        Refined formation-start epoch.
    """

    change = np.asarray(
        change,
        dtype=float
    )

    change_sigma = np.asarray(
        change_sigma,
        dtype=float
    )

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
        - int(max_extension_epochs)
    )

    search_end = raw_start_epoch

    candidate_epochs = np.arange(
        search_start,
        search_end + 1,
        dtype=int
    )

    if len(candidate_epochs) == 0:
        return raw_start_epoch

    signal_segment = change[
        candidate_epochs
    ]

    sigma_segment = change_sigma[
        candidate_epochs
    ]

    baseline_threshold = np.maximum(
        baseline_tolerance,
        baseline_sigma_factor
        * sigma_segment
    )

    if direction == "positive":

        same_sign_mask = (
            signal_segment
            >= -baseline_tolerance
        )

        baseline_mask = (
            np.abs(signal_segment)
            <= baseline_threshold
        )

    elif direction == "negative":

        same_sign_mask = (
            signal_segment
            <= baseline_tolerance
        )

        baseline_mask = (
            np.abs(signal_segment)
            <= baseline_threshold
        )

    else:

        return raw_start_epoch

    valid_baseline_epochs = (
        candidate_epochs[
            baseline_mask
            & same_sign_mask
        ]
    )

    if len(valid_baseline_epochs) > 0:

        extended_start = int(
            valid_baseline_epochs[-1]
        )

    else:

        finite_mask = np.isfinite(
            signal_segment
        )

        if not np.any(finite_mask):
            return raw_start_epoch

        finite_epochs = (
            candidate_epochs[
                finite_mask
            ]
        )

        finite_signal = (
            signal_segment[
                finite_mask
            ]
        )

        extended_start = int(
            finite_epochs[
                np.nanargmin(
                    np.abs(
                        finite_signal
                    )
                )
            ]
        )

    # --------------------------------------------------------
    # Trend-consistency guard
    # --------------------------------------------------------

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

        finite_slopes = slopes[
            np.isfinite(slopes)
        ]

        if len(finite_slopes) == 0:

            return raw_start_epoch

        if direction == "positive":

            trend_fraction = np.mean(
                finite_slopes
                >= -min_extension_slope
            )

            if trend_fraction < 0.50:
                extended_start = (
                    raw_start_epoch
                )

        elif direction == "negative":

            trend_fraction = np.mean(
                finite_slopes
                <= min_extension_slope
            )

            if trend_fraction < 0.50:
                extended_start = (
                    raw_start_epoch
                )

    return int(
        max(
            earliest_allowed_epoch,
            min(
                extended_start,
                raw_start_epoch
            )
        )
    )


def _build_seed_masks(
    signal,
    sigma,
    direction,
    z_threshold
):
    """
    Build statistically significant positive/negative masks.
    """

    lod = (
        float(z_threshold)
        * sigma
    )

    selected_masks = []

    if direction in [
        "positive",
        "both"
    ]:

        selected_masks.append(
            (
                "positive",
                signal > lod
            )
        )

    if direction in [
        "negative",
        "both"
    ]:

        selected_masks.append(
            (
                "negative",
                signal < -lod
            )
        )

    if direction not in [
        "positive",
        "negative",
        "both"
    ]:

        raise ValueError(
            "direction must be "
            "'positive', 'negative', or 'both'."
        )

    return selected_masks


def _detect_seeds(
    detection_signal,
    change,
    rate,
    sigma_change,
    sigma_rate,
    timestamps,
    corepoint_indices=None,
    direction="both",
    z_threshold=1.96,
    min_duration=10,
    min_magnitude=0.05,
    extend_to_formation_start=True,
    formation_baseline_tolerance=0.02,
    formation_baseline_sigma_factor=1.0,
    max_formation_extension_epochs=45,
    min_extension_slope=0.001
):
    """
    Internal generic implementation for KF-Mag and KF-Rate
    seed detection.
    """

    change = np.asarray(
        change,
        dtype=float
    )

    rate = np.asarray(
        rate,
        dtype=float
    )

    sigma_change = np.asarray(
        sigma_change,
        dtype=float
    )

    sigma_rate = np.asarray(
        sigma_rate,
        dtype=float
    )

    if change.ndim != 2:
        raise ValueError(
            "change must have shape "
            "(n_corepoints, n_epochs)."
        )

    if rate.shape != change.shape:
        raise ValueError(
            "rate must have the same shape as change."
        )

    if sigma_change.shape != change.shape:
        raise ValueError(
            "sigma_change must have the same shape "
            "as change."
        )

    if sigma_rate.shape != change.shape:
        raise ValueError(
            "sigma_rate must have the same shape "
            "as change."
        )

    n_corepoints, n_epochs = (
        change.shape
    )

    if len(timestamps) != n_epochs:

        raise ValueError(
            "timestamps length must match "
            "the number of epochs."
        )

    if corepoint_indices is None:

        indices = np.arange(
            n_corepoints,
            dtype=int
        )

    else:

        indices = np.asarray(
            list(corepoint_indices),
            dtype=int
        )

    if np.any(
        indices < 0
    ) or np.any(
        indices >= n_corepoints
    ):

        raise IndexError(
            "corepoint_indices contains indices outside "
            "the available corepoint range."
        )

    records = []

    for cp_idx in indices:

        cp_idx = int(
            cp_idx
        )

        cp_change = change[
            cp_idx
        ]

        cp_rate = rate[
            cp_idx
        ]

        cp_change_sigma = (
            sigma_change[
                cp_idx
            ]
        )

        cp_rate_sigma = (
            sigma_rate[
                cp_idx
            ]
        )

        if np.all(
            np.isnan(
                cp_change
            )
        ):
            continue

        if detection_signal == "magnitude":

            signal = cp_change
            sigma = cp_change_sigma

        elif detection_signal == "rate":

            signal = cp_rate
            sigma = cp_rate_sigma

        else:

            raise ValueError(
                "detection_signal must be "
                "'magnitude' or 'rate'."
            )

        seed_masks = _build_seed_masks(
            signal=signal,
            sigma=sigma,
            direction=direction,
            z_threshold=z_threshold
        )

        for (
            seed_direction,
            seed_mask
        ) in seed_masks:

            raw_intervals = (
                _extract_true_intervals(
                    seed_mask
                )
            )

            for (
                interval_id,
                (
                    raw_start_epoch,
                    raw_end_epoch
                )
            ) in enumerate(
                raw_intervals
            ):

                raw_start_epoch = int(
                    raw_start_epoch
                )

                raw_end_epoch = int(
                    raw_end_epoch
                )

                if interval_id == 0:

                    earliest_allowed_epoch = 0

                else:

                    previous_raw_end = int(
                        raw_intervals[
                            interval_id - 1
                        ][1]
                    )

                    earliest_allowed_epoch = (
                        previous_raw_end + 1
                    )

                # ------------------------------------------------
                # KF-Mag formation-start refinement
                # ------------------------------------------------

                if (
                    detection_signal
                    == "magnitude"
                    and extend_to_formation_start
                ):

                    start_epoch = (
                        _find_formation_start(
                            change=cp_change,
                            change_sigma=cp_change_sigma,
                            raw_start_epoch=raw_start_epoch,
                            direction=seed_direction,
                            earliest_allowed_epoch=(
                                earliest_allowed_epoch
                            ),
                            baseline_tolerance=(
                                formation_baseline_tolerance
                            ),
                            baseline_sigma_factor=(
                                formation_baseline_sigma_factor
                            ),
                            max_extension_epochs=(
                                max_formation_extension_epochs
                            ),
                            min_extension_slope=(
                                min_extension_slope
                            )
                        )
                    )

                else:

                    start_epoch = (
                        raw_start_epoch
                    )

                end_epoch = (
                    raw_end_epoch
                )

                duration = (
                    end_epoch
                    - start_epoch
                    + 1
                )

                if duration < min_duration:
                    continue

                interval_change = (
                    cp_change[
                        start_epoch:
                        end_epoch + 1
                    ]
                )

                interval_rate = (
                    cp_rate[
                        start_epoch:
                        end_epoch + 1
                    ]
                )

                interval_change_sigma = (
                    cp_change_sigma[
                        start_epoch:
                        end_epoch + 1
                    ]
                )

                interval_rate_sigma = (
                    cp_rate_sigma[
                        start_epoch:
                        end_epoch + 1
                    ]
                )

                raw_interval_change = (
                    cp_change[
                        raw_start_epoch:
                        raw_end_epoch + 1
                    ]
                )

                raw_interval_rate = (
                    cp_rate[
                        raw_start_epoch:
                        raw_end_epoch + 1
                    ]
                )

                if seed_direction == "positive":

                    if np.all(
                        np.isnan(
                            interval_change
                        )
                    ):
                        continue

                    event_magnitude = float(
                        np.nanmax(
                            interval_change
                        )
                    )

                    peak_epoch_local = int(
                        np.nanargmax(
                            interval_change
                        )
                    )

                elif seed_direction == "negative":

                    if np.all(
                        np.isnan(
                            interval_change
                        )
                    ):
                        continue

                    event_magnitude = float(
                        abs(
                            np.nanmin(
                                interval_change
                            )
                        )
                    )

                    peak_epoch_local = int(
                        np.nanargmin(
                            interval_change
                        )
                    )

                else:

                    continue

                if (
                    event_magnitude
                    < min_magnitude
                ):
                    continue

                peak_epoch = (
                    start_epoch
                    + peak_epoch_local
                )

                net_change = (
                    cp_change[end_epoch]
                    - cp_change[start_epoch]
                )

                abs_net_change = abs(
                    net_change
                )

                mean_change_lod = (
                    float(z_threshold)
                    * np.nanmean(
                        interval_change_sigma
                    )
                )

                mean_rate_lod = (
                    float(z_threshold)
                    * np.nanmean(
                        interval_rate_sigma
                    )
                )

                records.append({
                    "corepoint_index_python": (
                        cp_idx
                    ),

                    "corepoint_number_user": (
                        cp_idx + 1
                    ),

                    "start_epoch": (
                        int(start_epoch)
                    ),

                    "end_epoch": (
                        int(end_epoch)
                    ),

                    "duration_epochs": (
                        int(duration)
                    ),

                    "start_time": (
                        timestamps[
                            start_epoch
                        ]
                    ),

                    "end_time": (
                        timestamps[
                            end_epoch
                        ]
                    ),

                    "raw_significant_start_epoch": (
                        raw_start_epoch
                    ),

                    "raw_significant_end_epoch": (
                        raw_end_epoch
                    ),

                    "raw_significant_duration_epochs": (
                        raw_end_epoch
                        - raw_start_epoch
                        + 1
                    ),

                    "raw_significant_start_time": (
                        timestamps[
                            raw_start_epoch
                        ]
                    ),

                    "raw_significant_end_time": (
                        timestamps[
                            raw_end_epoch
                        ]
                    ),

                    "formation_start_extension_applied": (
                        bool(
                            start_epoch
                            != raw_start_epoch
                        )
                    ),

                    "formation_extension_epochs": (
                        int(
                            raw_start_epoch
                            - start_epoch
                        )
                    ),

                    "formation_start_extension_enabled": (
                        bool(
                            detection_signal
                            == "magnitude"
                            and extend_to_formation_start
                        )
                    ),

                    "direction": (
                        seed_direction
                    ),

                    "peak_epoch": (
                        int(
                            peak_epoch
                        )
                    ),

                    "peak_time": (
                        timestamps[
                            peak_epoch
                        ]
                    ),

                    "event_magnitude": (
                        event_magnitude
                    ),

                    "net_change": float(
                        net_change
                    ),

                    "abs_net_change": float(
                        abs_net_change
                    ),

                    "mean_change": float(
                        np.nanmean(
                            interval_change
                        )
                    ),

                    "max_abs_change": float(
                        np.nanmax(
                            np.abs(
                                interval_change
                            )
                        )
                    ),

                    "mean_rate": float(
                        np.nanmean(
                            interval_rate
                        )
                    ),

                    "max_abs_rate": float(
                        np.nanmax(
                            np.abs(
                                interval_rate
                            )
                        )
                    ),

                    "raw_mean_change": float(
                        np.nanmean(
                            raw_interval_change
                        )
                    ),

                    "raw_max_abs_change": float(
                        np.nanmax(
                            np.abs(
                                raw_interval_change
                            )
                        )
                    ),

                    "raw_mean_rate": float(
                        np.nanmean(
                            raw_interval_rate
                        )
                    ),

                    "raw_max_abs_rate": float(
                        np.nanmax(
                            np.abs(
                                raw_interval_rate
                            )
                        )
                    ),

                    "mean_change_sigma": float(
                        np.nanmean(
                            interval_change_sigma
                        )
                    ),

                    "mean_rate_sigma": float(
                        np.nanmean(
                            interval_rate_sigma
                        )
                    ),

                    "mean_change_lod": float(
                        mean_change_lod
                    ),

                    "mean_rate_lod": float(
                        mean_rate_lod
                    ),

                    "seed_detection_method": (
                        detection_signal
                    ),

                    "seed_direction_setting": (
                        direction
                    ),

                    "formation_baseline_tolerance": (
                        formation_baseline_tolerance
                    ),

                    "formation_baseline_sigma_factor": (
                        formation_baseline_sigma_factor
                    ),

                    "max_formation_extension_epochs": (
                        max_formation_extension_epochs
                    ),

                    "min_extension_slope": (
                        min_extension_slope
                        if detection_signal
                        == "magnitude"
                        else np.nan
                    )
                })

    seed_table = pd.DataFrame(
        records
    )

    if len(seed_table) > 0:

        seed_table = (
            seed_table
            .sort_values(
                [
                    "event_magnitude",
                    "duration_epochs"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )

        seed_table[
            "seed_rank"
        ] = np.arange(
            1,
            len(seed_table) + 1
        )

    return seed_table


def detect_magnitude_seeds(
    change,
    rate,
    sigma_change,
    sigma_rate,
    timestamps,
    corepoint_indices=None,
    direction="both",
    z_threshold=1.96,
    min_duration=10,
    min_magnitude=0.05,
    extend_to_formation_start=True,
    formation_baseline_tolerance=0.02,
    formation_baseline_sigma_factor=1.0,
    max_formation_extension_epochs=45,
    min_extension_slope=0.001
):
    """
    Detect KF-Mag seed intervals using statistically significant
    Kalman-smoothed change magnitude.
    """

    return _detect_seeds(
        detection_signal="magnitude",
        change=change,
        rate=rate,
        sigma_change=sigma_change,
        sigma_rate=sigma_rate,
        timestamps=timestamps,
        corepoint_indices=corepoint_indices,
        direction=direction,
        z_threshold=z_threshold,
        min_duration=min_duration,
        min_magnitude=min_magnitude,
        extend_to_formation_start=(
            extend_to_formation_start
        ),
        formation_baseline_tolerance=(
            formation_baseline_tolerance
        ),
        formation_baseline_sigma_factor=(
            formation_baseline_sigma_factor
        ),
        max_formation_extension_epochs=(
            max_formation_extension_epochs
        ),
        min_extension_slope=(
            min_extension_slope
        )
    )


def detect_rate_seeds(
    change,
    rate,
    sigma_change,
    sigma_rate,
    timestamps,
    corepoint_indices=None,
    direction="both",
    z_threshold=1.96,
    min_duration=10,
    min_magnitude=0.05
):
    """
    Detect KF-Rate seed intervals using statistically significant
    Kalman-estimated change rate.

    Event magnitude is still evaluated from the Kalman-smoothed
    change series, matching the current thesis implementation.
    """

    return _detect_seeds(
        detection_signal="rate",
        change=change,
        rate=rate,
        sigma_change=sigma_change,
        sigma_rate=sigma_rate,
        timestamps=timestamps,
        corepoint_indices=corepoint_indices,
        direction=direction,
        z_threshold=z_threshold,
        min_duration=min_duration,
        min_magnitude=min_magnitude,
        extend_to_formation_start=False
    )


# ============================================================
# SPATIAL SUPPORT VALIDATION
# ============================================================

from scipy.spatial import cKDTree


def validate_spatial_support(
    seed_table,
    corepoints,
    kalman_change,
    kalman_rate,
    timestamps,
    spatial_support_radius=1.5,
    min_neighbour_support=3,
    min_support_fraction=0.30,
    support_time_tolerance=5,
    neighbour_magnitude_ratio=0.5,
):
    """
    Validate Kalman-derived seed intervals using neighbouring
    corepoints with consistent temporal and directional support.

    Parameters
    ----------
    seed_table : pandas.DataFrame
        KF-Mag or KF-Rate seed table.

    corepoints : np.ndarray
        Corepoint coordinates, shape (n_corepoints, >=2).

    kalman_change : np.ndarray
        Kalman-smoothed change magnitude array.

    kalman_rate : np.ndarray
        Kalman change-rate array.

    timestamps : sequence
        Analysis timestamps.

    spatial_support_radius : float
        Spatial search radius around each seed corepoint.

    min_neighbour_support : int
        Minimum number of neighbouring corepoints required.

    min_support_fraction : float
        Minimum fraction of spatial neighbours that must support
        the event.

    support_time_tolerance : int
        Number of epochs by which the seed support window is
        expanded in both temporal directions.

    neighbour_magnitude_ratio : float
        Minimum neighbouring event magnitude relative to the
        candidate seed magnitude.

    Returns
    -------
    pandas.DataFrame
        Spatially validated seed table.
    """

    if seed_table is None or len(seed_table) == 0:
        raise ValueError(
            "seed_table is empty."
        )

    required_columns = [
        "corepoint_index_python",
        "start_epoch",
        "end_epoch",
        "direction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in seed_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from seed_table: "
            + ", ".join(missing_columns)
        )

    corepoints = np.asarray(
        corepoints,
        dtype=float
    )

    kalman_change = np.asarray(
        kalman_change,
        dtype=float
    )

    kalman_rate = np.asarray(
        kalman_rate,
        dtype=float
    )

    if corepoints.ndim != 2 or corepoints.shape[1] < 2:
        raise ValueError(
            "corepoints must have shape (n_corepoints, >=2)."
        )

    if kalman_change.ndim != 2:
        raise ValueError(
            "kalman_change must have shape "
            "(n_corepoints, n_epochs)."
        )

    if kalman_rate.shape != kalman_change.shape:
        raise ValueError(
            "kalman_rate must have the same shape "
            "as kalman_change."
        )

    n_corepoints, n_epochs = (
        kalman_change.shape
    )

    if corepoints.shape[0] != n_corepoints:
        raise ValueError(
            "corepoints and kalman_change must contain "
            "the same number of corepoints."
        )

    if len(timestamps) != n_epochs:
        raise ValueError(
            "timestamps length must match the number of epochs."
        )

    seed_table = seed_table.copy()

    # --------------------------------------------------------
    # Preserve support-aware intervals when already available
    # --------------------------------------------------------

    if "support_start_epoch" not in seed_table.columns:
        seed_table[
            "support_start_epoch"
        ] = seed_table[
            "start_epoch"
        ]

    if "support_end_epoch" not in seed_table.columns:
        seed_table[
            "support_end_epoch"
        ] = seed_table[
            "end_epoch"
        ]

    seed_table[
        "support_start_epoch"
    ] = seed_table[
        "support_start_epoch"
    ].fillna(
        seed_table[
            "start_epoch"
        ]
    ).astype(int)

    seed_table[
        "support_end_epoch"
    ] = seed_table[
        "support_end_epoch"
    ].fillna(
        seed_table[
            "end_epoch"
        ]
    ).astype(int)

    tree_xy = cKDTree(
        corepoints[:, :2]
    )

    validated_records = []

    for _, seed_row in seed_table.iterrows():

        cp_idx = int(
            seed_row[
                "corepoint_index_python"
            ]
        )

        start_epoch = int(
            seed_row[
                "start_epoch"
            ]
        )

        end_epoch = int(
            seed_row[
                "end_epoch"
            ]
        )

        support_start_epoch = int(
            seed_row[
                "support_start_epoch"
            ]
        )

        support_end_epoch = int(
            seed_row[
                "support_end_epoch"
            ]
        )

        direction = (
            seed_row[
                "direction"
            ]
        )

        if (
            cp_idx < 0
            or cp_idx >= n_corepoints
        ):
            continue

        if (
            support_start_epoch < 0
            or support_end_epoch >= n_epochs
            or support_end_epoch < support_start_epoch
        ):
            continue

        change = (
            kalman_change[
                cp_idx
            ]
        )

        rate = (
            kalman_rate[
                cp_idx
            ]
        )

        if np.all(
            np.isnan(
                change
            )
        ):
            continue

        interval_change = (
            change[
                support_start_epoch:
                support_end_epoch + 1
            ]
        )

        interval_rate = (
            rate[
                support_start_epoch:
                support_end_epoch + 1
            ]
        )

        if np.all(
            np.isnan(
                interval_change
            )
        ):
            continue

        if direction == "positive":

            event_magnitude = float(
                np.nanmax(
                    interval_change
                )
            )

            peak_epoch_local = int(
                np.nanargmax(
                    interval_change
                )
            )

        elif direction == "negative":

            event_magnitude = float(
                abs(
                    np.nanmin(
                        interval_change
                    )
                )
            )

            peak_epoch_local = int(
                np.nanargmin(
                    interval_change
                )
            )

        else:
            continue

        peak_epoch = (
            support_start_epoch
            + peak_epoch_local
        )

        net_change = (
            change[
                support_end_epoch
            ]
            - change[
                support_start_epoch
            ]
        )

        abs_net_change = abs(
            net_change
        )

        neighbours = tree_xy.query_ball_point(
            corepoints[
                cp_idx,
                :2
            ],
            r=float(
                spatial_support_radius
            )
        )

        neighbours = [
            int(n)
            for n in neighbours
            if int(n) != cp_idx
        ]

        if len(neighbours) == 0:
            continue

        spatial_support_start = max(
            0,
            support_start_epoch
            - int(
                support_time_tolerance
            )
        )

        spatial_support_end = min(
            n_epochs - 1,
            support_end_epoch
            + int(
                support_time_tolerance
            )
        )

        neighbour_support_count = 0

        for nb in neighbours:

            nb_change = (
                kalman_change[
                    nb
                ]
            )

            if np.all(
                np.isnan(
                    nb_change
                )
            ):
                continue

            nb_interval = (
                nb_change[
                    spatial_support_start:
                    spatial_support_end + 1
                ]
            )

            if np.all(
                np.isnan(
                    nb_interval
                )
            ):
                continue

            if direction == "positive":

                nb_event_magnitude = float(
                    np.nanmax(
                        nb_interval
                    )
                )

                same_direction = (
                    nb_event_magnitude > 0
                )

                meaningful = (
                    nb_event_magnitude
                    >= (
                        neighbour_magnitude_ratio
                        * event_magnitude
                    )
                )

            elif direction == "negative":

                nb_min = float(
                    np.nanmin(
                        nb_interval
                    )
                )

                nb_event_magnitude = abs(
                    nb_min
                )

                same_direction = (
                    nb_min < 0
                )

                meaningful = (
                    nb_event_magnitude
                    >= (
                        neighbour_magnitude_ratio
                        * event_magnitude
                    )
                )

            else:

                same_direction = False
                meaningful = False

            if (
                same_direction
                and meaningful
            ):
                neighbour_support_count += 1

        support_fraction = (
            neighbour_support_count
            / max(
                len(neighbours),
                1
            )
        )

        if (
            neighbour_support_count
            < min_neighbour_support
        ):
            continue

        if (
            support_fraction
            < min_support_fraction
        ):
            continue

        record = (
            seed_row.to_dict()
        )

        record.update({
            "support_start_epoch":
                support_start_epoch,

            "support_end_epoch":
                support_end_epoch,

            "support_duration_epochs":
                (
                    support_end_epoch
                    - support_start_epoch
                    + 1
                ),

            "support_start_time":
                timestamps[
                    support_start_epoch
                ],

            "support_end_time":
                timestamps[
                    support_end_epoch
                ],

            "peak_epoch":
                int(
                    peak_epoch
                ),

            "peak_time":
                timestamps[
                    peak_epoch
                ],

            "event_magnitude":
                float(
                    event_magnitude
                ),

            "net_change":
                float(
                    net_change
                ),

            "abs_net_change":
                float(
                    abs_net_change
                ),

            "mean_change":
                float(
                    np.nanmean(
                        interval_change
                    )
                ),

            "max_abs_change":
                float(
                    np.nanmax(
                        np.abs(
                            interval_change
                        )
                    )
                ),

            "mean_rate":
                float(
                    np.nanmean(
                        interval_rate
                    )
                ),

            "max_abs_rate":
                float(
                    np.nanmax(
                        np.abs(
                            interval_rate
                        )
                    )
                ),

            "n_neighbours":
                int(
                    len(
                        neighbours
                    )
                ),

            "neighbour_support_count":
                int(
                    neighbour_support_count
                ),

            "support_fraction":
                float(
                    support_fraction
                ),

            "spatial_support_used":
                True,

            "spatial_support_radius":
                float(
                    spatial_support_radius
                ),

            "min_neighbour_support":
                int(
                    min_neighbour_support
                ),

            "min_support_fraction":
                float(
                    min_support_fraction
                ),

            "support_time_tolerance":
                int(
                    support_time_tolerance
                ),

            "neighbour_magnitude_ratio":
                float(
                    neighbour_magnitude_ratio
                ),
        })

        validated_records.append(
            record
        )

    validated_table = pd.DataFrame(
        validated_records
    )

    if len(validated_table) > 0:

        validated_table = (
            validated_table
            .sort_values(
                [
                    "event_magnitude",
                    "support_fraction",
                    "support_duration_epochs"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
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
            dtype=int
        )

    return validated_table


# ============================================================
# NON-MAXIMUM SUPPRESSION OF KALMAN SEEDS
# ============================================================

def non_maximum_suppression(
    seed_table,
    corepoints,
    nms_radius=3.0,
    min_support_count=3,
    min_support_fraction=0.30,
):
    """
    Apply spatial-temporal non-maximum suppression to ranked
    Kalman-derived seed intervals.

    Seeds are ranked primarily by absolute net change. For seeds
    with similar strength, available spatial-support information
    and seed duration provide secondary ranking.

    A candidate seed is retained when:
        1. sufficient nearby seed support exists,
        2. supporting seeds have the same change direction,
        3. their temporal intervals overlap.

    Once a seed is retained, nearby lower-ranked seeds describing
    the same directional and temporally overlapping event are
    suppressed.

    Parameters
    ----------
    seed_table : pandas.DataFrame
        KF-Mag or KF-Rate seed table, optionally already spatially
        validated.

    corepoints : np.ndarray
        Corepoint coordinates, shape (n_corepoints, >=2).

    nms_radius : float
        Spatial NMS search radius.

    min_support_count : int
        Minimum number of neighbouring supporting seeds.

    min_support_fraction : float
        Minimum fraction of neighbours supporting the seed.

    Returns
    -------
    pandas.DataFrame
        NMS-retained seed table.
    """

    if seed_table is None or len(seed_table) == 0:
        raise ValueError(
            "seed_table is empty."
        )

    required_columns = [
        "corepoint_index_python",
        "start_epoch",
        "end_epoch",
        "direction",
    ]

    missing = [
        col
        for col in required_columns
        if col not in seed_table.columns
    ]

    if missing:
        raise ValueError(
            "Missing required seed columns: "
            + ", ".join(missing)
        )

    corepoints = np.asarray(
        corepoints,
        dtype=float
    )

    if (
        corepoints.ndim != 2
        or corepoints.shape[1] < 2
    ):
        raise ValueError(
            "corepoints must have shape "
            "(n_corepoints, >=2)."
        )

    table = seed_table.copy()

    # --------------------------------------------------------
    # Support-aware temporal intervals
    # --------------------------------------------------------

    if "support_start_epoch" not in table.columns:
        table["support_start_epoch"] = (
            table["start_epoch"]
        )

    if "support_end_epoch" not in table.columns:
        table["support_end_epoch"] = (
            table["end_epoch"]
        )

    table["support_start_epoch"] = (
        table["support_start_epoch"]
        .fillna(table["start_epoch"])
        .astype(int)
    )

    table["support_end_epoch"] = (
        table["support_end_epoch"]
        .fillna(table["end_epoch"])
        .astype(int)
    )

    table["support_duration_epochs"] = (
        table["support_end_epoch"]
        - table["support_start_epoch"]
        + 1
    )

    # --------------------------------------------------------
    # Ranking variables
    # --------------------------------------------------------

    if "abs_net_change" not in table.columns:

        if "net_change" in table.columns:

            table["abs_net_change"] = (
                table["net_change"].abs()
            )

        elif "event_magnitude" in table.columns:

            table["abs_net_change"] = (
                table["event_magnitude"].abs()
            )

        else:

            raise ValueError(
                "Seed table must contain abs_net_change, "
                "net_change, or event_magnitude."
            )

    if "support_fraction" not in table.columns:
        table["support_fraction"] = np.nan

    table[
        "support_fraction_for_ranking"
    ] = (
        table["support_fraction"]
        .fillna(0.0)
    )

    # Preserve current ranking logic.
    table = table.sort_values(
        [
            "abs_net_change",
            "support_fraction_for_ranking",
            "support_duration_epochs",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    table["pre_nms_rank"] = np.arange(
        1,
        len(table) + 1,
        dtype=int,
    )

    # --------------------------------------------------------
    # Spatial index
    # --------------------------------------------------------

    tree = cKDTree(
        corepoints[:, :2]
    )

    kept_rows = []

    suppressed = set()

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    for row_index, seed_row in table.iterrows():

        if row_index in suppressed:
            continue

        cp_idx = int(
            seed_row[
                "corepoint_index_python"
            ]
        )

        if (
            cp_idx < 0
            or cp_idx >= len(corepoints)
        ):
            continue

        seed_start = int(
            seed_row[
                "support_start_epoch"
            ]
        )

        seed_end = int(
            seed_row[
                "support_end_epoch"
            ]
        )

        direction = (
            seed_row[
                "direction"
            ]
        )

        neighbours = tree.query_ball_point(
            corepoints[
                cp_idx,
                :2
            ],
            r=float(
                nms_radius
            ),
        )

        neighbours = [
            int(n)
            for n in neighbours
            if int(n) != cp_idx
        ]

        nearby_seeds = table[
            table[
                "corepoint_index_python"
            ].isin(
                neighbours
            )
        ]

        support_count = 0

        for _, neighbour_seed in (
            nearby_seeds.iterrows()
        ):

            if (
                neighbour_seed[
                    "direction"
                ]
                != direction
            ):
                continue

            neighbour_start = int(
                neighbour_seed[
                    "support_start_epoch"
                ]
            )

            neighbour_end = int(
                neighbour_seed[
                    "support_end_epoch"
                ]
            )

            overlap = max(
                0,
                min(
                    seed_end,
                    neighbour_end,
                )
                - max(
                    seed_start,
                    neighbour_start,
                )
                + 1,
            )

            if overlap > 0:
                support_count += 1

        support_fraction = (
            support_count
            / max(
                len(neighbours),
                1,
            )
        )

        if (
            support_count
            < min_support_count
        ):
            continue

        if (
            support_fraction
            < min_support_fraction
        ):
            continue

        retained_seed = (
            seed_row.copy()
        )

        retained_seed[
            "nms_support_count"
        ] = int(
            support_count
        )

        retained_seed[
            "nms_support_fraction"
        ] = float(
            support_fraction
        )

        retained_seed[
            "nms_neighbour_count"
        ] = int(
            len(neighbours)
        )

        kept_rows.append(
            retained_seed
        )

        # ----------------------------------------------------
        # Suppress lower-ranked nearby representations
        # of the same event.
        # ----------------------------------------------------

        for (
            other_index,
            other_seed
        ) in table.iterrows():

            if (
                other_index == row_index
                or other_index in suppressed
            ):
                continue

            other_cp = int(
                other_seed[
                    "corepoint_index_python"
                ]
            )

            if other_cp not in neighbours:
                continue

            if (
                other_seed[
                    "direction"
                ]
                != direction
            ):
                continue

            other_start = int(
                other_seed[
                    "support_start_epoch"
                ]
            )

            other_end = int(
                other_seed[
                    "support_end_epoch"
                ]
            )

            overlap = max(
                0,
                min(
                    seed_end,
                    other_end,
                )
                - max(
                    seed_start,
                    other_start,
                )
                + 1,
            )

            if overlap > 0:

                suppressed.add(
                    other_index
                )

    nms_table = pd.DataFrame(
        kept_rows
    )

    if len(nms_table) == 0:
        return nms_table

    nms_table = nms_table.sort_values(
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
    ).reset_index(
        drop=True
    )

    nms_table["nms_rank"] = np.arange(
        1,
        len(nms_table) + 1,
        dtype=int,
    )

    nms_table[
        "nms_radius"
    ] = float(
        nms_radius
    )

    nms_table[
        "nms_min_support_count"
    ] = int(
        min_support_count
    )

    nms_table[
        "nms_min_support_fraction"
    ] = float(
        min_support_fraction
    )

    return nms_table
