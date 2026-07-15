"""Typed exceptions so callers never see a raw Docling traceback or need to know
Docling's own exception hierarchy."""

from __future__ import annotations

from pathlib import Path


class DoclingParsingError(Exception):
    """Base class for every error raised by docling_parsing."""

    def __init__(self, message: str, source: str | Path | None = None):
        self.source = source
        super().__init__(message)


class UnsupportedFormatError(DoclingParsingError):
    """The input file's format isn't one Docling can parse (or wasn't recognized at all)."""


class ExtractionFailedError(DoclingParsingError):
    """Docling recognized the format but conversion failed outright (status FAILURE),
    as opposed to PARTIAL_SUCCESS, which is not an error -- see DocumentMetadata.status."""
