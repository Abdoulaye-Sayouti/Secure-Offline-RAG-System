from abc import ABC, abstractmethod
from typing import List, Any, Union
from pathlib import Path

class BaseLoader(ABC):
    """Abstract base class for all document loaders."""
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Any]:
        """Load document(s) from source."""
        pass