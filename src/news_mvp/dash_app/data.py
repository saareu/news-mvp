"""Generates mock data for the Dash application."""

from __future__ import annotations

from datetime import datetime, timedelta
import functools

import numpy as np
import polars as pl


@functools.lru_cache(maxsize=10)
def create_mock_heatmap_data(days: int = 30) -> pl.DataFrame:
    """
    Generates a mock Polars DataFrame with hourly article counts.
    Results are cached to prevent regenerating the same data repeatedly.

    Args:
        days: The number of past days to generate data for.

    Returns:
        A Polars DataFrame with 'publish_dt' (hourly timestamps) and 'count' columns.
    """
    # Use a fixed date to ensure consistent results
    # This ensures we don't generate new data due to time changes
    # October 1, 2025 as reference point
    end_date = datetime(2025, 10, 1, 0, 0, 0)
    start_date = end_date - timedelta(days=days)

    # Control data density based on time range
    # Fewer data points for better performance
    if days <= 1:
        interval = "30m"  # 30-minute intervals for 1 day
        max_value = 30
    elif days <= 7:
        interval = "2h"  # 2-hour intervals for 7 days
        max_value = 40
    else:
        interval = "6h"  # 6-hour intervals for longer periods
        max_value = 50

    # Generate timestamps with fixed intervals
    timestamps = pl.datetime_range(
        start=start_date, end=end_date, interval=interval, eager=True
    )

    # Generate deterministic counts with pattern
    # Use a fixed seed for reproducible results
    np.random.seed(42)
    n_points = len(timestamps)

    # Create more stable pattern
    t = np.linspace(0, days, n_points)
    counts = (
        np.random.randint(5, 15, n_points)  # Base noise
        + 10 * np.sin(t * np.pi / 4)  # Multi-day cycle
        + 5 * np.sin(t * 2 * np.pi)  # Daily cycle
    )
    counts = np.maximum(0, counts).astype(int)
    counts = np.minimum(counts, max_value).astype(int)  # Cap maximum values

    df = pl.DataFrame({"publish_dt": timestamps, "count": counts})

    return df


@functools.lru_cache(maxsize=10)
def create_mock_cluster_data(days: int = 30, num_clusters: int = 5) -> pl.DataFrame:
    """
    Generates mock cluster data with article counts per cluster over time.

    Args:
        days: The number of past days to generate data for.
        num_clusters: Number of clusters to generate.

    Returns:
        A Polars DataFrame with cluster data.
    """
    # Use a fixed date to ensure consistent results
    end_date = datetime(2025, 10, 1, 0, 0, 0)
    start_date = end_date - timedelta(days=days)

    # Determine interval based on days
    if days <= 1:
        interval = "30m"
    elif days <= 7:
        interval = "2h"
    else:
        interval = "6h"

    # Generate timestamps
    timestamps = pl.datetime_range(
        start=start_date, end=end_date, interval=interval, eager=True
    )

    # Create dataframes for each cluster
    np.random.seed(42)
    n_points = len(timestamps)
    dfs = []

    cluster_topics = [
        "Politics",
        "Sports",
        "Technology",
        "Entertainment",
        "Business",
        "Health",
        "Science",
        "Environment",
    ]

    for cluster_id in range(1, num_clusters + 1):
        # Create different pattern for each cluster
        phase_shift = cluster_id * np.pi / num_clusters
        # amplitude not used but defined for consistency with pattern generation
        # amplitude = np.random.uniform(5, 15)
        base_count = np.random.randint(3, 10)

        t = np.linspace(0, days, n_points)

        # Regular pattern
        counts = np.random.randint(1, 5, n_points) + base_count * np.sin(  # Base noise
            t * np.pi / 4 + phase_shift
        )  # Multi-day cycle

        # Add some bursts (sudden spikes)
        if np.random.random() > 0.3:  # 70% chance of having bursts
            burst_points = np.random.choice(
                range(n_points),
                size=np.random.randint(1, 3),  # 1-2 bursts
                replace=False,
            )

            for point in burst_points:
                # Create a burst over several time points
                burst_length = np.random.randint(3, 8)
                burst_start = max(0, point - burst_length // 2)
                burst_end = min(n_points, point + burst_length // 2)

                # Create spike pattern
                spike = np.random.randint(15, 30)
                decay = np.linspace(spike, spike / 3, burst_end - burst_start)
                counts[burst_start:burst_end] += decay

        counts = np.maximum(0, counts).astype(int)

        # Select a topic for this cluster
        if cluster_id <= len(cluster_topics):
            topic = cluster_topics[cluster_id - 1]
        else:
            topic = f"Topic {cluster_id}"

        # Create dataframe for this cluster
        cluster_df = pl.DataFrame(
            {
                "publish_dt": timestamps,
                "count": counts,
                "cluster_id": cluster_id,
                "topic": topic,
            }
        )

        dfs.append(cluster_df)

    # Combine all cluster dataframes
    if dfs:
        result = pl.concat(dfs)
        return result

    # Return empty dataframe with correct schema if no clusters
    return pl.DataFrame({"publish_dt": [], "count": [], "cluster_id": [], "topic": []})


def detect_bursts(
    df: pl.DataFrame, window: int = 5, zscore_threshold: float = 3.0
) -> pl.DataFrame:
    """
    Detect bursts in time series data using rolling z-score.

    Args:
        df: Polars DataFrame with time series data.
        window: Rolling window size for z-score calculation.
        zscore_threshold: Threshold for burst detection.

    Returns:
        DataFrame with burst information.
    """
    # Convert to pandas for rolling window operations (easier than Polars for this)
    pdf = df.to_pandas()

    # Group by cluster and calculate rolling statistics
    bursts = []

    for cluster_id, group in pdf.groupby("cluster_id"):
        # Sort by time
        group = group.sort_values("publish_dt")

        # Calculate rolling mean and std for z-score
        rolling_mean = group["count"].rolling(window=window, min_periods=1).mean()
        rolling_std = group["count"].rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero

        # Calculate z-scores
        zscores = (group["count"] - rolling_mean) / rolling_std

        # Identify bursts where z-score exceeds threshold
        burst_mask = zscores >= zscore_threshold

        if burst_mask.any():
            # Get consecutive burst periods
            burst_periods = []
            burst_start = None

            for i, (dt, is_burst) in enumerate(zip(group["publish_dt"], burst_mask)):
                if is_burst and burst_start is None:
                    burst_start = dt
                elif not is_burst and burst_start is not None:
                    burst_periods.append(
                        {
                            "cluster_id": cluster_id,
                            "topic": group["topic"].iloc[0],
                            "start_time": burst_start,
                            "end_time": group["publish_dt"].iloc[i - 1],
                            "max_count": group["count"]
                            .iloc[max(0, i - window) : i]
                            .max(),
                            "zscore": zscores.iloc[i - 1],
                        }
                    )
                    burst_start = None

            # Handle case where burst is at the end of the data
            if burst_start is not None:
                burst_periods.append(
                    {
                        "cluster_id": cluster_id,
                        "topic": group["topic"].iloc[0],
                        "start_time": burst_start,
                        "end_time": group["publish_dt"].iloc[-1],
                        "max_count": group["count"].iloc[-window:].max(),
                        "zscore": zscores.iloc[-1],
                    }
                )

            bursts.extend(burst_periods)

    # Convert to Polars DataFrame
    if bursts:
        burst_df = pl.DataFrame(bursts)
        return burst_df

    # Return empty dataframe with correct schema
    return pl.DataFrame(
        {
            "cluster_id": [],
            "topic": [],
            "start_time": [],
            "end_time": [],
            "max_count": [],
            "zscore": [],
        }
    )


@functools.lru_cache(maxsize=10)
def create_mock_cluster_purity_data(num_clusters: int = 5) -> pl.DataFrame:
    """
    Generate mock cluster purity metrics for admin view.

    Args:
        num_clusters: Number of clusters to generate data for.

    Returns:
        DataFrame with cluster purity metrics.
    """
    np.random.seed(42)  # For reproducibility

    data = []
    for cluster_id in range(1, num_clusters + 1):
        # Generate random purity score between 0.6 and 1.0
        purity_score = np.random.uniform(0.6, 1.0)

        # Generate source distribution
        ynet_pct = np.random.uniform(0.1, 0.8)
        haaretz_pct = np.random.uniform(0.1, 0.8 - ynet_pct)
        hayom_pct = 1.0 - ynet_pct - haaretz_pct

        # Size is number of articles in cluster
        cluster_size = np.random.randint(5, 100)

        # Generate top keywords
        keywords = [
            "politics",
            "economy",
            "technology",
            "health",
            "sports",
            "culture",
            "security",
            "education",
            "environment",
            "transportation",
        ]
        np.random.shuffle(keywords)
        top_keywords = keywords[:3]

        data.append(
            {
                "cluster_id": cluster_id,
                "purity_score": purity_score,
                "size": cluster_size,
                "ynet_pct": ynet_pct,
                "haaretz_pct": haaretz_pct,
                "hayom_pct": hayom_pct,
                "top_keywords": ", ".join(top_keywords),
            }
        )

    return pl.DataFrame(data)


@functools.lru_cache(maxsize=10)
def create_mock_source_specific_data(
    days: int = 30, num_clusters: int = 5
) -> pl.DataFrame:
    """
    Generate source-specific article count data for each cluster.

    Args:
        days: Number of days to generate data for.
        num_clusters: Number of clusters to include.

    Returns:
        DataFrame with source-specific counts per cluster over time.
    """
    # Use a fixed date to ensure consistent results
    end_date = datetime(2025, 10, 1, 0, 0, 0)
    start_date = end_date - timedelta(days=days)

    # Determine interval based on days
    if days <= 1:
        interval = "1h"
    elif days <= 7:
        interval = "4h"
    else:
        interval = "12h"

    # Generate timestamps
    timestamps = pl.datetime_range(
        start=start_date, end=end_date, interval=interval, eager=True
    )

    # Create dataframes for each source and cluster
    np.random.seed(43)  # Different seed for variety
    sources = ["ynet", "haaretz", "hayom"]
    n_points = len(timestamps)
    dfs = []

    for cluster_id in range(1, num_clusters + 1):
        for source in sources:
            # Different pattern for each source
            phase_shift = (
                (sources.index(source) + cluster_id)
                * np.pi
                / (len(sources) * num_clusters)
            )
            # amplitude not used but defined for consistency with pattern generation
            # amplitude = np.random.uniform(2, 8)
            base_count = np.random.randint(1, 5)

            t = np.linspace(0, days, n_points)

            # Regular pattern with source-specific characteristics
            counts = np.random.randint(
                0, 3, n_points
            ) + base_count * np.sin(  # Base noise
                t * np.pi / 4 + phase_shift
            )  # Multi-day cycle

            # Source-specific patterns
            if source == "ynet":
                # More spiky
                counts += np.random.binomial(1, 0.2, n_points) * np.random.randint(
                    3, 8, n_points
                )
            elif source == "haaretz":
                # More even
                counts = np.convolve(counts, np.ones(3) / 3, mode="same")

            counts = np.maximum(0, counts).astype(int)

            source_df = pl.DataFrame(
                {
                    "publish_dt": timestamps,
                    "count": counts,
                    "cluster_id": cluster_id,
                    "source": source,
                }
            )

            dfs.append(source_df)

    # Combine all dataframes
    if dfs:
        result = pl.concat(dfs)
        return result

    # Return empty dataframe with correct schema
    return pl.DataFrame({"publish_dt": [], "count": [], "cluster_id": [], "source": []})


@functools.lru_cache(maxsize=1)
def create_mock_umap_data(
    num_clusters: int = 5, points_per_cluster: int = 100
) -> pl.DataFrame:
    """
    Generate mock UMAP embedding data for visualization.

    Args:
        num_clusters: Number of clusters to generate.
        points_per_cluster: Points per cluster.

    Returns:
        DataFrame with x, y coordinates and cluster assignments.
    """
    np.random.seed(44)

    data = []
    for cluster_id in range(1, num_clusters + 1):
        # Generate cluster center
        center_x = np.random.uniform(-5, 5)
        center_y = np.random.uniform(-5, 5)

        # Generate points around center with some noise
        points_x = center_x + np.random.normal(0, 1, points_per_cluster)
        points_y = center_y + np.random.normal(0, 1, points_per_cluster)

        # Source distribution
        sources = np.random.choice(
            ["ynet", "haaretz", "hayom"], size=points_per_cluster, p=[0.4, 0.3, 0.3]
        )

        for i in range(points_per_cluster):
            data.append(
                {
                    "x": float(points_x[i]),
                    "y": float(points_y[i]),
                    "cluster_id": cluster_id,
                    "source": sources[i],
                }
            )

    return pl.DataFrame(data)
