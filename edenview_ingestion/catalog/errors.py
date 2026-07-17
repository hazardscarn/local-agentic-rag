"""Typed exceptions for the catalog package -- mirrors docling_parsing/errors.py's
approach so callers never need to know DuckDB's own exception types."""

from __future__ import annotations


class CatalogError(Exception):
    """Base class for every error raised by the catalog package."""


class NotFoundError(CatalogError):
    """A lookup by name/id found no matching row."""


class DuplicateNameError(CatalogError):
    """A DB name or Qdrant collection name that must be unique already exists --
    surfaced as a typed error instead of a raw DuckDB constraint-violation traceback."""
