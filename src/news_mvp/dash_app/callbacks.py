"""Dash callbacks for the Articles Timeline view."""

from __future__ import annotations

import time
from functools import wraps
from typing import Dict, List

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, callback_context, html, dcc, ALL
from dash.exceptions import PreventUpdate
from news_mvp.logging_setup import get_logger
from .data import (
    create_mock_heatmap_data,
    create_mock_cluster_data,
    detect_bursts,
    create_mock_cluster_purity_data,
    create_mock_source_specific_data,
    create_mock_umap_data,
)

logger = get_logger(__name__)

# Cache for throttling callbacks
_last_callback_time = {}
_throttle_cache = {}


def throttle(interval=1.0):
    """
    Decorator to throttle callback execution to prevent excessive updates.
    Enhanced with context-aware debouncing for smoother UI rendering.

    Args:
        interval: Minimum time (in seconds) between callback executions.
    """

    def decorator(func):
        # Track active callbacks to prevent overlapping executions
        active_executions = set()

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            func_id = func.__name__
            context_id = str(args) + str(kwargs)  # Context-aware caching
            cache_key = f"{func_id}:{context_id}"

            # Avoid executing a function that's already running
            if func_id in active_executions:
                return _throttle_cache.get(cache_key)

            # If this is the first call or enough time has passed
            if (
                cache_key not in _last_callback_time
                or now - _last_callback_time[cache_key] > interval
            ):

                try:
                    # Mark as active to prevent overlapping execution
                    active_executions.add(func_id)

                    # Update the timestamp
                    _last_callback_time[cache_key] = now

                    # Execute the function and cache result with context-specific key
                    result = func(*args, **kwargs)
                    _throttle_cache[cache_key] = result
                    return result

                finally:
                    # Always remove from active set when done
                    active_executions.discard(func_id)

            # Return cached result
            return _throttle_cache.get(cache_key)

        return wrapper

    return decorator


def register_callbacks(app: Dash) -> None:
    """
    Register all callbacks for the Dash app.

    Args:
        app: The Dash application instance.
    """

    # Store the initial figure to avoid regenerating on every callback
    _initial_figures = {}

    @app.callback(
        [
            Output("view_mode", "data"),
            Output("customer-view-btn", "color"),
            Output("customer-view-btn", "outline"),
            Output("admin-view-btn", "color"),
            Output("admin-view-btn", "outline"),
            Output("admin-filters", "style"),
            Output("embedding-container", "style"),
        ],
        [
            Input("customer-view-btn", "n_clicks"),
            Input("admin-view-btn", "n_clicks"),
        ],
        [
            State("view_mode", "data"),
        ],
        prevent_initial_call=True,
    )
    def toggle_view_mode(customer_clicks, admin_clicks, current_mode):
        """
        Toggle between Customer and Admin view modes.

        Args:
            customer_clicks: Number of clicks on customer button.
            admin_clicks: Number of clicks on admin button.
            current_mode: Current view mode.

        Returns:
            Updated view mode and button states.
        """
        ctx = callback_context
        if not ctx.triggered:
            return (
                current_mode,
                "primary" if current_mode == "customer" else "secondary",
                False if current_mode == "customer" else True,
                "secondary" if current_mode == "customer" else "primary",
                True if current_mode == "customer" else False,
                (
                    {"display": "none"}
                    if current_mode == "customer"
                    else {"display": "block"}
                ),
                (
                    {"display": "none"}
                    if current_mode == "customer"
                    else {"display": "block"}
                ),
            )

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "customer-view-btn":
            new_mode = "customer"
        else:
            new_mode = "admin"

        return (
            new_mode,
            "primary" if new_mode == "customer" else "secondary",
            False if new_mode == "customer" else True,
            "secondary" if new_mode == "customer" else "primary",
            True if new_mode == "customer" else False,
            {"display": "none"} if new_mode == "customer" else {"display": "block"},
            {"display": "none"} if new_mode == "customer" else {"display": "block"},
        )

    @app.callback(
        Output("density_strip", "figure"),
        Input("date_range_picker", "value"),
        prevent_initial_call=False,
    )
    @throttle(interval=1.0)  # Throttle to max once per second
    def update_density_heatmap(date_range_value: str) -> go.Figure:
        """
        Create or update the density strip heatmap based on the selected date range.
        Throttled to prevent excessive updates.
        """
        # Check if we already have this figure cached
        cache_key = f"heatmap_{date_range_value}"
        if cache_key in _initial_figures:
            return _initial_figures[cache_key]

        # Generate data only if needed
        days_map = {"today": 1, "7d": 7, "30d": 30}
        days = days_map.get(date_range_value, 7)
        df = create_mock_heatmap_data(days=days)

        # Performance optimized figure creation
        # Avoid expensive operations like string formatting in loops
        counts = df["count"].to_list()

        # Use simple bar chart instead of scattergl for more stable rendering
        fig = go.Figure(
            data=go.Bar(
                x=df["publish_dt"],
                y=counts,
                marker=dict(
                    color=counts,
                    colorscale="Viridis",
                    showscale=False,
                ),
                hoverinfo="x+text",
                text=[f"Count: {c}" for c in counts],
                width=3600000,  # Width in milliseconds (1 hour)
            )
        )

        fig.update_layout(
            title=f"Article Density - Last {days} Day(s)",
            margin=dict(t=40, b=20, l=40, r=20),  # Increased left margin for y-axis
            autosize=True,
            yaxis=dict(
                title="Count",
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                fixedrange=True,  # Prevent y-axis zooming
                range=[0, max(counts) * 1.1],  # Fixed y-axis range with 10% headroom
            ),
            xaxis=dict(
                title="Time",
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                fixedrange=False,  # Allow x-axis zooming
                rangeslider=dict(visible=False),  # No range slider for cleaner look
            ),
            height=120,  # Fixed height
            dragmode="select",  # Enable brushing
            plot_bgcolor="rgba(240,240,240,0.3)",  # Light background
            # Set a fixed uirevision to maintain view state and prevent redraws
            uirevision="fixed_density_strip",
            font=dict(size=10),  # Smaller font
        )

        # Cache the figure for future use
        _initial_figures[f"heatmap_{date_range_value}"] = fig

        return fig

    @app.callback(
        Output("umap-embedding", "figure"),
        Input("view_mode", "data"),
        prevent_initial_call=False,
    )
    @throttle(interval=2.0)  # Throttle to prevent excessive updates
    def update_umap_embedding(view_mode):
        """
        Update the UMAP embedding visualization (Admin view only).

        Args:
            view_mode: Current view mode ("customer" or "admin").

        Returns:
            Updated UMAP figure.
        """
        # Only generate if in admin mode
        if view_mode != "admin":
            # Return minimal figure to save resources
            return {
                "data": [],
                "layout": {
                    "title": "UMAP Projection of Article Embeddings",
                    "showlegend": False,
                    "height": 300,
                },
            }

        # Generate mock UMAP data
        umap_data = create_mock_umap_data(num_clusters=5)

        # Get unique clusters for coloring
        clusters = umap_data["cluster_id"].unique().to_list()

        # Create a trace for each cluster
        traces = []
        for cluster_id in sorted(clusters):
            cluster_data = umap_data.filter(pl.col("cluster_id") == cluster_id)

            # Create traces by source within each cluster
            for source in ["ynet", "haaretz", "hayom"]:
                source_data = cluster_data.filter(pl.col("source") == source)

                if not source_data.is_empty():
                    source_marker = {
                        "ynet": {"symbol": "circle"},
                        "haaretz": {"symbol": "square"},
                        "hayom": {"symbol": "diamond"},
                    }

                    traces.append(
                        go.Scatter(
                            x=source_data["x"].to_list(),
                            y=source_data["y"].to_list(),
                            mode="markers",
                            marker={
                                "size": 8,
                                "opacity": 0.7,
                                "line": {"width": 0.5, "color": "white"},
                                "symbol": source_marker[source]["symbol"],
                            },
                            name=f"Cluster {cluster_id} - {source}",
                            legendgroup=f"Cluster {cluster_id}",
                        )
                    )

        # Create the figure
        fig = go.Figure(data=traces)

        # Update layout
        fig.update_layout(
            title="UMAP Projection of Article Embeddings",
            xaxis={"title": "UMAP Dimension 1", "zeroline": False},
            yaxis={"title": "UMAP Dimension 2", "zeroline": False},
            legend={"title": "Cluster - Source"},
            height=300,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
            hovermode="closest",
            plot_bgcolor="rgba(240,240,240,0.3)",
        )

        return fig

    @app.callback(
        Output("selected_time_range", "data"),
        Input("density_strip", "selectedData"),
        Input({"type": "burst_marker", "index": ALL}, "n_clicks"),
        State({"type": "burst_marker", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    @throttle(interval=0.5)  # Throttle selection updates
    def store_selected_time_range(
        selected_data: dict | None,
        burst_clicks: List[int | None],
        burst_ids: List[dict | None],
    ) -> dict | None:
        """
        Store the brushed time range from the heatmap or burst click.
        Handles both regular time range selection and burst click events.

        Args:
            selected_data: The data selected by brushing on the heatmap.
            burst_clicks: Click count on burst markers.
            burst_id: ID of the clicked burst marker (contains burst range info).

        Returns:
            A dictionary with 'start' and 'end' timestamps or None.
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Check which input triggered the callback
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle burst marker clicks
        if "burst_marker" in trigger_id:
            # Find which burst marker was clicked
            clicked_index = None
            for i, clicks in enumerate(burst_clicks):
                if clicks is not None and clicks > 0:
                    clicked_index = i
                    break

            if clicked_index is not None and clicked_index < len(burst_ids):
                # Get the burst ID that was clicked
                clicked_burst = burst_ids[clicked_index]
                if not clicked_burst:
                    raise PreventUpdate

                burst_index = clicked_burst.get("index", "")

                try:
                    # Extract start and end times from the marker ID
                    # Format is "burst-{cluster_id}-{start_time}-{end_time}"
                    parts = burst_index.split("-")
                    if len(parts) >= 4:
                        _, cluster_id, start_time, end_time = (
                            parts[0],
                            parts[1],
                            "-".join(parts[2:-1]),
                            parts[-1],
                        )
                        return {
                            "start": start_time,
                            "end": end_time,
                            "source": "burst",
                            "cluster_id": cluster_id,
                        }
                except (ValueError, IndexError, AttributeError):
                    # If there's an error extracting burst info, prevent update
                    raise PreventUpdate

        # Handle regular time range selection from the density strip
        elif (
            trigger_id == "density_strip"
            and selected_data
            and selected_data.get("points")
        ):
            points = selected_data["points"]
            if not points:
                return None

            # For a scatter plot, selectedData gives us the x-values of the selected points
            # Use a try/except to guard against invalid data
            try:
                selected_x = [point["x"] for point in points]
                start_time = min(selected_x)
                end_time = max(selected_x)
                return {"start": start_time, "end": end_time, "source": "brush"}
            except (KeyError, ValueError):
                # If there's an error in processing, prevent update
                raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output("cluster_bands", "children"),
        Input("date_range_picker", "value"),
        Input("selected_time_range", "data"),
        Input("view_mode", "data"),
        Input("purity-filter", "value"),
        Input("min-size-filter", "value"),
        prevent_initial_call=False,
    )
    @throttle(interval=1.0)  # Throttle updates to once per second
    def update_cluster_bands(
        date_range_value: str,
        selected_time_range: Dict | None,
        view_mode: str,
        purity_filter: float,
        min_size_filter: int,
    ) -> List[html.Div]:
        """
        Create or update the cluster bands visualization with burst detection overlays.

        Args:
            date_range_value: Selected date range.
            selected_time_range: Selected time range from brushing or burst click.

        Returns:
            List of HTML components representing cluster bands with burst overlays.
        """
        # Map date range to number of days
        days_map = {"today": 1, "7d": 7, "30d": 30}
        days = days_map.get(date_range_value, 7)

        # Get cluster data and detect bursts
        cluster_df = create_mock_cluster_data(days=days)
        burst_df = detect_bursts(cluster_df)

        # Get source-specific data for Admin view
        source_df = (
            create_mock_source_specific_data(days=days)
            if view_mode == "admin"
            else None
        )

        # Get cluster purity data for Admin view
        purity_df = create_mock_cluster_purity_data() if view_mode == "admin" else None

        # Apply Admin filters if needed
        if (
            view_mode == "admin"
            and purity_df is not None
            and purity_filter is not None
            and min_size_filter is not None
        ):
            # Filter by purity score and cluster size
            filtered_clusters = purity_df.filter(
                (pl.col("purity_score") >= purity_filter)
                & (pl.col("size") >= min_size_filter)
            )

            # Get filtered cluster IDs
            valid_cluster_ids = filtered_clusters["cluster_id"].to_list()

            # Apply filtering to cluster_df
            if valid_cluster_ids:
                cluster_df = cluster_df.filter(
                    pl.col("cluster_id").is_in(valid_cluster_ids)
                )
                if source_df is not None:
                    source_df = source_df.filter(
                        pl.col("cluster_id").is_in(valid_cluster_ids)
                    )
                if not burst_df.is_empty():
                    burst_df = burst_df.filter(
                        pl.col("cluster_id").is_in(valid_cluster_ids)
                    )

        # Group by cluster
        cluster_list = []

        # Get unique clusters
        clusters = cluster_df.group_by("cluster_id").agg(
            pl.col("topic").first().alias("topic")
        )

        # Calculate max count for scaling
        max_count_value = 30

        # For admin view, try to get actual max count
        if view_mode == "admin" and source_df is not None and not source_df.is_empty():
            try:
                source_max = source_df["count"].max()
                if source_max is not None and isinstance(source_max, (int, float)):
                    max_count_value = max(max_count_value, int(source_max))
            except Exception as e:
                logger.warning(f"Could not determine max count from source data: {e}")

        # Generate a component for each cluster
        for row in clusters.iter_rows(named=True):
            cluster_id = row["cluster_id"]
            topic = row["topic"]

            # Filter data for this cluster
            cluster_data = cluster_df.filter(pl.col("cluster_id") == cluster_id)

            # Find bursts for this cluster
            cluster_bursts = (
                burst_df.filter(pl.col("cluster_id") == cluster_id).to_dicts()
                if not burst_df.is_empty()
                else []
            )

            # Create spark line trace
            timestamps = [
                dt.strftime("%Y-%m-%d %H:%M:%S")
                for dt in cluster_data["publish_dt"].to_list()
            ]
            counts = cluster_data["count"].to_list()

            # Create a mini sparkline graph
            sparkline = dcc.Graph(
                id=f"sparkline-{cluster_id}",
                figure={
                    "data": [
                        {
                            "x": timestamps,
                            "y": counts,
                            "type": "line",
                            "line": {"width": 1.5, "color": "#007bff"},
                            "mode": "lines",
                            "name": topic,
                        }
                    ],
                    "layout": {
                        "margin": {"l": 30, "r": 10, "t": 10, "b": 20},
                        "height": 60,
                        "xaxis": {
                            "showgrid": False,
                            "zeroline": False,
                            "showticklabels": False,
                            "fixedrange": True,
                        },
                        "yaxis": {
                            "showgrid": False,
                            "zeroline": False,
                            "showticklabels": False,
                            "fixedrange": True,
                            "range": [0, max_count_value * 1.1],
                        },
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "hovermode": False,
                        "uirevision": f"sparkline-{topic}",
                    },
                },
                config={"displayModeBar": False},
                className="mb-0 mt-0",
                style={"height": "60px"},
            )

            # Create burst overlays for this cluster
            burst_overlays = []
            for burst in cluster_bursts:
                # Calculate position and width based on time range
                try:
                    start_dt = burst["start_time"]
                    end_dt = burst["end_time"]
                    # max_count not used but kept as comment for future reference
                    # max_count = burst["max_count"]
                    zscore = burst["zscore"]

                    # Create burst ID with start and end time for click handling
                    burst_id = f"burst-{cluster_id}-{start_dt}-{end_dt}"  # Create translucent overlay for burst
                    overlay = html.Div(
                        className="burst-overlay",
                        style={
                            "position": "absolute",
                            "backgroundColor": "rgba(255, 165, 0, 0.3)",
                            "border": "1px dashed orange",
                            "height": "100%",
                            "zIndex": 10,
                            # Position will be set with JS in a callback
                        },
                        id=f"overlay-{burst_id}",
                    )

                    # Create lightning marker for burst
                    marker = html.Button(
                        "⚡",
                        id={"type": "burst_marker", "index": burst_id},
                        className="burst-marker",
                        style={
                            "position": "absolute",
                            "top": "-12px",
                            "cursor": "pointer",
                            "backgroundColor": "transparent",
                            "border": "none",
                            "fontSize": "16px",
                            "zIndex": 20,
                            # Position will be set with JS in a callback
                        },
                        title=f"Burst in '{topic}': Z-score {zscore:.1f}. Click to zoom.",
                    )

                    burst_overlays.extend([overlay, marker])
                except Exception:
                    # Skip burst if there's an error
                    continue

            # For admin view, add source breakdown data
            source_breakdown = []
            if view_mode == "admin" and source_df is not None:
                # Get source data for this cluster
                cluster_sources = source_df.filter(pl.col("cluster_id") == cluster_id)

                if not cluster_sources.is_empty():
                    # Calculate source breakdowns (for the entire period)
                    sources_agg = cluster_sources.group_by("source").agg(
                        pl.sum("count").alias("total")
                    )

                    # Calculate percentages
                    total_count = sources_agg["total"].sum()
                    if total_count > 0:
                        # Create a horizontal bar chart with percentages
                        source_labels = sources_agg["source"].to_list()
                        source_counts = sources_agg["total"].to_list()
                        source_percentages = [
                            count / total_count * 100 for count in source_counts
                        ]

                        # Look up purity data if available
                        purity_info = ""
                        if purity_df is not None:
                            cluster_purity = purity_df.filter(
                                pl.col("cluster_id") == cluster_id
                            )
                            if not cluster_purity.is_empty():
                                purity = cluster_purity["purity_score"][0]
                                size = cluster_purity["size"][0]
                                keywords = cluster_purity["top_keywords"][0]
                                purity_info = html.Div(
                                    [
                                        html.Small(
                                            f"Purity: {purity:.2f} | Size: {size}",
                                            className="text-muted",
                                        ),
                                        html.Br(),
                                        html.Small(
                                            f"Keywords: {keywords}",
                                            className="text-muted",
                                        ),
                                    ]
                                )

                        # Create source breakdown chart
                        source_chart = dcc.Graph(
                            id=f"sources-{cluster_id}",
                            figure={
                                "data": [
                                    {
                                        "x": source_percentages,
                                        "y": source_labels,
                                        "type": "bar",
                                        "orientation": "h",
                                        "marker": {
                                            "color": ["#1E88E5", "#D81B60", "#FFC107"],
                                        },
                                    }
                                ],
                                "layout": {
                                    "margin": {"l": 50, "r": 10, "t": 10, "b": 10},
                                    "height": 80,
                                    "xaxis": {
                                        "showgrid": False,
                                        "zeroline": False,
                                        "showticklabels": False,
                                        "fixedrange": True,
                                    },
                                    "yaxis": {
                                        "showgrid": False,
                                        "zeroline": False,
                                        "fixedrange": True,
                                    },
                                    "plot_bgcolor": "rgba(0,0,0,0)",
                                    "paper_bgcolor": "rgba(0,0,0,0)",
                                    "hovermode": "closest",
                                    "hoverinfo": "x+y",
                                    "barmode": "stack",
                                },
                            },
                            config={"displayModeBar": False},
                            className="mt-1 mb-0",
                            style={"height": "80px"},
                        )

                        # Add percentages as labels next to the source name
                        source_labels_with_pct = html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(f"{source}: ", className="me-1"),
                                        html.Span(
                                            f"{pct:.1f}%",
                                            className="badge bg-secondary",
                                        ),
                                    ],
                                    className="me-2 d-inline-block",
                                )
                                for source, pct in zip(
                                    source_labels, source_percentages
                                )
                            ],
                            className="d-flex flex-wrap small",
                        )

                        # Add source breakdown to the list
                        source_breakdown = [
                            html.Div(
                                [purity_info, source_labels_with_pct],
                                className="mt-1 mb-1",
                            ),
                            source_chart,
                        ]

            # Combine cluster info with sparkline and (optionally) source breakdown
            cluster_component = html.Div(
                [
                    html.Div(
                        [
                            html.H6(topic, className="mb-0 mt-0"),
                            html.Small(f"Cluster {cluster_id}", className="text-muted"),
                        ],
                        className="mb-1",
                    ),
                    html.Div(
                        [sparkline, *burst_overlays],
                        className="position-relative",
                        id=f"cluster-container-{cluster_id}",
                    ),
                    # Add source breakdown for admin view
                    *source_breakdown,
                ],
                className="mb-4",
                id=f"cluster-band-{cluster_id}",
            )

            cluster_list.append(cluster_component)

        # If no clusters, show a message
        if not cluster_list:
            return [
                html.Div(
                    "No cluster data available", className="text-center text-muted py-5"
                )
            ]

        return cluster_list
