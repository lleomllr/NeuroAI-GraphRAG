from abc import ABC, abstractmethod
from src.extract.schema import GraphExtraction

class BaseExtractor(ABC):
    @abstractmethod
    def extract_graph_from_text(self, text: str) -> GraphExtraction | None:
        pass