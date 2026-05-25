from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from src.extract.schema import GraphExtraction
from src.config import config

class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j_uri
        )
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)

    def close(self):
        self.driver.close()

    def ingest_graph(self, doc_id: str, paper_title: str, graph: GraphExtraction):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {id: $doc_id})
                SET p.name = $title, p.label = 'Paper'
                """,
                doc_id=doc_id, title=paper_title
            )

            for node in graph.nodes:
                vector = self.encoder.encode(node.name).tolist()

                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.name = $name, n.label = $label, n.embedding = $embedding
                    WITH n
                    MATCH (p:Paper {id: $doc_id})
                    MERGE (p)-[:MENTIONS]->(n)
                    """,
                    id=node.id, name=node.name, label=node.label, 
                    embedding=vector, doc_id=doc_id
                )
            
            for edge in graph.edges:
                session.run(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATION {type: $type}]->(target)
                    SET r.description = $desc
                    """,
                    source_id=edge.source_id, 
                    target_id=edge.target_id,
                    type=edge.type, 
                    desc=edge.description
                )