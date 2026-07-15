"""Must be imported before any other Docling module in this package -- see `__init__.py`.

Docling resolves certain model-stage presets (e.g. picture classification) at first
import of `docling.datamodel.pipeline_options`, baking in whatever
`settings.inference.compile_torch_models` is at that exact moment; setting it afterwards
has no effect on presets that already resolved. This module exists solely to flip that
setting before anything else in `docling_parsing` can trigger that import.

Windows without an MSVC compiler (`cl.exe`) on PATH -- a very common local dev setup, not
an edge case -- makes `torch.compile` crash outright rather than just run slower, so it's
disabled unconditionally rather than only on machines where it happens to fail.
"""

from docling.datamodel.settings import settings

settings.inference.compile_torch_models = False
