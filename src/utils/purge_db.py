from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from src.config import config

def purge_all():
    try:
        qdrant_client = QdrantClient(url=config.QDRANT_URL)
        collection_name = "arxiv_papers"
        if qdrant_client.collection_exists(collection_name):
            print(f"Deleting Qdrant collection: '{collection_name}'...")
            qdrant_client.delete_collection(collection_name=collection_name)
            print("-> Qdrant collection purged successfully.")
        else:
            print(f"The Qdrant collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Error occurred while purging Qdrant: {e}")
        
    try:
        print("Purge complete of the Neo4j graph (MATCH (n) DETACH DELETE n)...")
        driver = GraphDatabase.driver(
            config.neo4j_uri
        )
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("-> Neo4j purged successfully.")
    except Exception as e:
        print(f"Error occurred while purging Neo4j: {e}")
        
    print("=== PURGE COMPLETE ===")

if __name__ == "__main__":
    confirm = input("Are you sure you want to purge Qdrant and Neo4j? (y/N) : ")
    if confirm.lower() == 'y':
        purge_all()
    else:
        print("Purge cancelled.")