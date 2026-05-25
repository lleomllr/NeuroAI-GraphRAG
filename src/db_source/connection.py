import os 
from neo4j import GraphDatabase
from src.config import config

class Neo4jConnection:
    def __init__(self):
        uri = config.neo4j_uri
        self.driver = GraphDatabase.driver(config.neo4j_uri)

    def close(self):
        self.driver.close()
    
    def test_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 'Connection successful!' AS msg")
            return result.single()["msg"] 
        
    def init_vector_index(self):
        query = """
        CREATE VECTOR INDEX article_embedding_index IF NOT EXISTS 
        FOR (a:paper) ON (a.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536, 
            `vector.similarity`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)
            print("Vector index created or already exists")

if __name__ == "__main__":
    manager = Neo4jConnection()
    try: 
        print(manager.test_connection())
        manager.init_vector_index()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        manager.close()