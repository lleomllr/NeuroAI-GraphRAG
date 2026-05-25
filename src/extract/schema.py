from pydantic import BaseModel, Field 
from typing import List, Literal

class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: Literal["Paper", "Author", "Model", "Dataset", "Concept", "Metric"] = Field(description="Type of the node")
    name: str = Field(description="Name of the node")

class Edge(BaseModel):
    source_id: str = Field(description="ID of the source node")
    target_id: str = Field(description="ID of the target node")
    type: Literal[
        "AUTHORED_BY",   # Paper -> Author
        "EVALUATED_ON",  # Model -> Dataset
        "USES_METHOD",   # Model/Paper -> Concept
        "ACHIEVES",      # Model -> Metric
        "RELATED_TO"     # Concept -> Concept
    ] = Field(description="Type of the edge")
    description: str = Field(description="Description of the relationship based on the text")

class GraphExtraction(BaseModel):
    nodes: List[Node] 
    edges: List[Edge] 