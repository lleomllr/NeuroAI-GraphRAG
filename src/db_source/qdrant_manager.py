from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import config 

class QdrantManager:
    def __init__(self, collection_name: str = "arxiv_papers"):
        self.client = QdrantClient(url=config.QDRANT_URL) 
        self.collection_name = collection_name
        self.vector_size = 384  
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection_name}' create in Qdrant")

    def insert_document(self, doc_id: str, text: str, vector: list[float], metadata: dict = None):
        payload = {"text": text}
        if metadata:
            payload.update(metadata)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id, 
                    vector=vector,
                    payload=payload
                )
            ]
        )