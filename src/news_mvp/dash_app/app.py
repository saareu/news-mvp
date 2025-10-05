"""Dash application entry point for the Articles Timeline view."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from . import callbacks, layout


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,  # Prevent callback validation errors
    update_title="",  # Empty string instead of default "Updating..." title
    assets_folder="assets",  # For custom CSS
)
app.layout = layout.layout()
callbacks.register_callbacks(app)

server = app.server


if __name__ == "__main__":
    # Disable the Dash reloader to avoid duplicate processes opening multiple windows on Windows.
    try:
        app.run(debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        # Clean termination ensures processes are properly closed
        import sys

        sys.exit(0)
