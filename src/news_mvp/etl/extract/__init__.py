#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified etl.extract package init.

We intentionally export only `fetch_rss_bytes` to keep the package import fast and
avoid importing CLI modules when importing the package.
"""

from .rss_fetcher import fetch_rss_bytes

__all__ = ["fetch_rss_bytes"]