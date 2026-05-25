import pytest
import sys
from pydantic import ValidationError
from src.extract.schema import GraphExtraction, Node, Edge

def test_valid_graph_extraction():
    data = {
        "nodes": [
            {"id": "n1", "label": "Concept", "name": "Backpropagation"}
        ],
        "edges": [
            {"source_id": "n1", "target_id": "n2", "type": "RELATED_TO", "description": "test"}
        ]
    }
    graph = GraphExtraction(**data)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].name == "Backpropagation"

def test_invalid_graph_extraction_missing_fields():
    data = {
        "nodes": [
            {"id": "n1", "name": "Backpropagation"} 
        ],
        "edges": []
    }
    
    with pytest.raises(ValidationError):
         GraphExtraction(**data)

if __name__ == "__main__":
    print("Manual test: Running pytest for test_schema.py")
    sys.exit(pytest.main([__file__]))