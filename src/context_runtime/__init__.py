"""Compatibility package for Context Runtime <=0.1.x.

New code should import :mod:`infinitum`. This namespace is retained so an
in-place upgrade does not immediately break integrations that imported the old
package name.
"""

from infinitum import __version__

__all__ = ["__version__"]
