"""Dash layout scaffolding for the Articles Timeline view."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def build_header() -> dbc.Navbar:
    """Construct the header bar with navigation controls."""
    environment_badge = dbc.Badge("Dev", color="secondary", className="ms-2")

    # View mode toggle (Customer/Admin)
    view_mode_toggle = dbc.ButtonGroup(
        [
            dbc.Button(
                "Customer",
                id="customer-view-btn",
                color="primary",
                outline=False,
                className="view-toggle-btn",
                n_clicks=0,
            ),
            dbc.Button(
                "Admin",
                id="admin-view-btn",
                color="secondary",
                outline=True,
                className="view-toggle-btn",
                n_clicks=0,
            ),
        ],
        id="view-mode-toggle",
        className="me-3",
    )

    date_range_picker = dcc.RadioItems(
        id="date_range_picker",
        options=[
            {"label": "Today", "value": "today"},
            {"label": "7d", "value": "7d"},
            {"label": "30d", "value": "30d"},
        ],
        value="today",
        inline=True,
        className="ms-3",
        inputClassName="me-2",
        labelClassName="me-3",
    )

    language_toggle = dcc.RadioItems(
        id="language_toggle",
        options=[
            {"label": "HE", "value": "he"},
            {"label": "EN", "value": "en"},
        ],
        value="en",
        inline=True,
        className="ms-3",
        inputClassName="me-2",
        labelClassName="me-1",
    )

    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Articles Timeline"),
                environment_badge,
                # Add view toggle group to the left of the other controls
                html.Div(
                    [view_mode_toggle],
                    className="d-flex align-items-center ms-auto me-3",
                ),
                html.Div(
                    [date_range_picker, language_toggle],
                    className="d-flex align-items-center",
                ),
            ],
            fluid=True,
            className="align-items-center",
        ),
        color="light",
        dark=False,
        sticky="top",
        className="mb-3",
    )

    return header


def build_filters_panel() -> dbc.Collapse:
    """Construct the collapsible filters panel with placeholders."""
    # Common filters for both Customer and Admin views
    common_filters = [
        html.Div(
            [
                html.H6("Search"),
                dbc.Input(id="search-filter", placeholder="Search articles"),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                html.H6("Source"),
                dbc.Checklist(
                    id="source-filter",
                    options=[
                        {"label": "Ynet", "value": "ynet"},
                        {"label": "Haaretz", "value": "haaretz"},
                        {"label": "Israel Hayom", "value": "hayom"},
                    ],
                    value=["ynet", "haaretz", "hayom"],
                    inline=True,
                ),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                html.H6("Cluster"),
                dbc.Placeholder(className="w-100", style={"height": "40px"}),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                html.H6("Language"),
                dbc.Checklist(
                    id="language-filter",
                    options=[
                        {"label": "Hebrew", "value": "he"},
                        {"label": "English", "value": "en"},
                    ],
                    value=["he", "en"],
                    inline=True,
                ),
            ],
            className="mb-3",
        ),
    ]

    # Admin-specific filters (hidden in Customer view)
    admin_filters = html.Div(
        [
            html.Hr(),
            html.H6("Admin Filters", className="text-danger"),
            html.Div(
                [
                    html.Label("Cluster Purity", className="form-label"),
                    dcc.Slider(
                        id="purity-filter",
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.5,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.Label("Min Cluster Size", className="form-label"),
                    dcc.Slider(
                        id="min-size-filter",
                        min=5,
                        max=50,
                        step=5,
                        value=10,
                        marks={5: "5", 25: "25", 50: "50"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="mb-3",
            ),
        ],
        id="admin-filters",
        style={"display": "none"},  # Hidden by default (Customer view)
    )

    filter_content = dbc.Card(
        [
            dbc.CardHeader("Filters"),
            dbc.CardBody(common_filters + [admin_filters]),
        ],
        className="h-100",
    )

    collapse = dbc.Collapse(
        filter_content,
        id="filters_panel",
        is_open=True,
        className="me-3",
    )

    return collapse


def build_main_canvas() -> html.Div:
    """Construct the main visualization placeholders."""
    # The density strip will be a dcc.Graph component with fixed height constraints
    density_strip_graph = html.Div(
        dcc.Graph(
            id="density_strip",
            config={"displayModeBar": False},  # Minimal UI for the heatmap
            style={"height": "120px"},  # Fixed height
            responsive=True,  # Responsive width
        ),
        className="mb-4",
        style={
            "height": "120px",
            "max-height": "120px",
            "overflow": "hidden",
        },  # Container constraints
    )

    # UMAP embedding visualization (Admin view only)
    embedding_viz = html.Div(
        [
            html.H5("Embedding Visualization (UMAP)", className="mt-3 mb-2"),
            html.Div(
                dcc.Graph(
                    id="umap-embedding",
                    config={"displayModeBar": True},
                    style={"height": "300px"},
                    figure={
                        "layout": {"title": "UMAP Projection of Article Embeddings"}
                    },
                ),
                className="border rounded p-3 mb-4 bg-white",
            ),
        ],
        id="embedding-container",
        style={"display": "none"},  # Hidden by default (Customer view)
    )

    return html.Div(
        [
            density_strip_graph,
            embedding_viz,
            html.Div(id="cluster_bands", className="border rounded p-4 bg-light"),
        ],
        className="flex-grow-1",
    )


def build_detail_rail() -> html.Div:
    """Construct the detail rail container (hidden by default)."""
    return html.Div(
        id="detail_rail",
        className="border-start ps-3",
        style={"minWidth": "280px", "display": "none"},
    )


def build_footer() -> html.Footer:
    """Construct the footer with article counts."""
    return html.Footer(
        dbc.Container(
            html.Small("Displaying X of Y articles"),
            fluid=True,
            className="py-2 text-muted",
        ),
        className="mt-4 border-top bg-white",
    )


def layout() -> html.Div:
    """Return the root layout for the Articles Timeline view."""
    header = build_header()
    filters_panel = build_filters_panel()
    main_canvas = build_main_canvas()
    detail_rail = build_detail_rail()
    footer = build_footer()

    content = dbc.Container(
        dbc.Row(
            [
                dbc.Col(filters_panel, width=3, lg=2),
                dbc.Col(main_canvas, width=9, lg=8, className="d-flex flex-column"),
                dbc.Col(detail_rail, width=12, lg=2),
            ],
            className="g-0",
        ),
        fluid=True,
    )

    return html.Div(
        [
            header,
            content,
            footer,
            # Store components for app state
            dcc.Store(id="selected_time_range"),
            dcc.Store(id="view_mode", data="customer"),  # Default to customer view
        ],
        className="vh-100 d-flex flex-column",
    )
