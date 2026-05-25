import arxiv
import logging
import uuid 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import config
from src.extract.ollama_extract import LocalLLMExtractor
from src.extract.hf_extract import HFExtractor
from src.db_source.qdrant_manager import QdrantManager
from src.db_source.neo4j_writer import Neo4jWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_extractor():
    if config.EXTRACTION_BACKEND == "huggingface":
        return HFExtractor()
    
    return LocalLLMExtractor()

def fetch_arxiv_papers(query="cat:cs.AI", max_results=5):
    logger.info(f"Searching {max_results} papers on arXiv with the query: {query}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return list(client.results(search))

def main():
    logger.info("Starting the GraphRAG pipeline")
    
    extractor = get_extractor()
    db_writer = Neo4jWriter()
    qdrant_manager = QdrantManager()

    encoder = SentenceTransformer(config.EMBEDDING_MODEL)
    
    papers = fetch_arxiv_papers(query="ti:\"large language models\" OR ti:\"neurosymbolic\"", max_results=5)
    
    success_count = 0
    
    for paper in tqdm(papers, desc="Processing papers"):
        raw_arxiv_id = paper.entry_id.split('/')[-1] if hasattr(paper, 'entry_id') else getattr(paper, 'arxiv_id', '')
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_arxiv_id))
        text_to_analyze = f"Title: {paper.title}\n\nAbstract: {paper.summary}"

        paper_vec = encoder.encode(text_to_analyze).tolist()
        qdrant_manager.insert_document(
            doc_id=doc_id,
            text=text_to_analyze,
            vector=paper_vec,
            metadata={"title": paper.title}
        )

        graph_data = extractor.extract_graph_from_text(text_to_analyze)
        
        if graph_data:
            try:
                db_writer.ingest_graph(doc_id, paper.title, graph_data)
                success_count += 1
            except Exception as e:
                logger.error(f"Error inserting into Neo4j for '{paper.title}': {e}")
        else:
            logger.warning(f"Failed to extract graph for paper: {paper.title}")
            
    db_writer.close()
    logger.info(f"Pipeline completed. {success_count}/{len(papers)} graphs inserted successfully.")

if __name__ == "__main__":
    main()