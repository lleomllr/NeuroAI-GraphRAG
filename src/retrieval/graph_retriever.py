from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.config import config

class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j_uri
        )
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)

    def close(self):
        self.driver.close()

    def get_subgraph_context(self, entity_name: str) -> str: 
        query = """
        MATCH (source:Entity)
        WHERE toLower(source.name) CONTAINS toLower($entity_name)
        MATCH (source)-[r:RELATION]-(target:Entity)
        RETURN source.name AS source_name, 
                type(r) AS relation_type,
                r.description AS description,
                target.name AS target_name
        LIMIT 20
        """

        with self.driver.session() as session:
            res = session.run(query, entity_name=entity_name)
            records = list(res)

        if not records:
            return "No relevant context found in the graph database."
        
        context_lines = [f"Context extracted from graph for entity '{entity_name}':"]
        for record in records:
            line = (
                f"- {record['source_name']} "
                f"-[{record['relation_type']}] "
                f"{record['target_name']} "
                f"(Description: {record['description']})"
            )
            context_lines.append(line)

        return "\n".join(context_lines)
    
    def get_hybrid_context(self, user_question: str, top_k: int = 3) -> str:
        question_vector = self.encoder.encode(user_question).tolist()

        search_res = self.qdrant_client.query_points(
            collection_name="arxiv_papers",
            query=question_vector,
            limit=top_k
        ).points

        if not search_res: 
            return "No relevant documents found in vector database"
        
        doc_ids = [str(hit.id) for hit in search_res]

        vec_context = []
        for hit in search_res:
            title = hit.payload.get("title", "Unknown Title")
            text = hit.payload.get("text", "")
            vec_context.append(f"--- Paper: {title} (ID: {hit.id}, Similarity Score: {hit.score:.2f}) ---\n{text}")

        query = """
        MATCH (p:Paper) WHERE p.id IN $doc_ids
        MATCH (p)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (e)-[r:RELATION]->(neighbor:Entity)
        RETURN p.name AS paper_title, 
               e.name AS entity_name, 
               e.label AS entity_label, 
               r.type AS relation_type, 
               neighbor.name AS neighbor_name,
               r.description AS relation_description
        LIMIT 30
        """
        
        with self.driver.session() as session:
            result = session.run(query, doc_ids=doc_ids)
            records = list(result)
        
        graph_context = []
        seen_rel = set()
            
        if records:
            graph_context.append("Context extracted from graph database:")
            for record in records:
                paper = record["paper_title"]
                entity = f"{record['entity_name']} ({record['entity_label']})"
                mention_line = f"- Paper: {paper} mentions Entity: {entity}"

                if mention_line not in graph_context:
                    graph_context.append(mention_line)
                if record["relation_type"] and record["neighbor_name"]:
                    rel_key = (record["entity_name"], record["relation_type"], record["neighbor_name"])
                    if rel_key not in seen_rel: 
                        seen_rel.add(rel_key)
                        graph_context.append(
                            f" * {record['entity_name']} --[{record['relation_type']}]--> {record['neighbor_name']} ({record['relation_description']})"
                        )
        else:
            graph_context.append("No relevant context found in the graph database")
        
        full_context = (
            "=== RAW TEXTUAL CONTEXT FROM VECTOR DB ===\n" +
            "\n\n".join(vec_context) +
            "\n\n=== STRUCTURED CONTEXT FROM GRAPH DB ===\n" +
            "\n".join(graph_context)
        )
        return full_context
    
if __name__ == "__main__":
    retriever = GraphRetriever()
    try:
        context = retriever.get_hybrid_context("large language models neurosymbolic", top_k=2)
        print(context)
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
    finally:
        retriever.close()