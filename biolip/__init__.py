"""BioLip-style lip-motion video branch."""

try:
    from .config import Config, get_config
except (ImportError, ValueError):
    from config import Config, get_config

__version__ = "0.1.0"
__all__ = ["Config", "get_config", "__version__"]
