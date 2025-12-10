try:  # Python 3.12+
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

__all__ = ["override"]