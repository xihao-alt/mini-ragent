"""Common interface implemented by every MiniRAGent tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Tool(ABC):
    """A discoverable operation that can be exposed through a Tool Registry."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool using arguments matching ``parameters``."""
