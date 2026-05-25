import urllib.request
import xml.etree.ElementTree as ET
import json 
from pathlib import Path
from pydantic import BaseModel
from typing import List

from src.config import config

class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    categories: List[str]

class ArxivClient: 
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.namespace = {"atom": "http://www.w3.org/2005/Atom"}

    def fetch_papers(self, max_res: int = config.ARXIV_MAX_RES) -> List[ArxivPaper]:
        search_query = "%28cat:cs.AI%29+OR+%28cat:cs.CL%29+OR+%28cat:cs.LG%29+AND+q-bio.NC"
        url = f"{self.base_url}?search_query={search_query}&max_results={max_res}"

        print(f"Fetching papers from arXiv with URL: {url}")
        try: 
            with urllib.request.urlopen(url) as response: 
                xml_data = response.read()
        except Exception as e: 
            print(f"Error fetching data from arXiv: {e}")
            return []
        return self._parse_xml(xml_data)
    
    def _parse_xml(self, xml_data: bytes) -> List[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", self.namespace):
            arxiv_id = entry.find("atom:id", self.namespace).text.split("/abs/")[-1]
            title = entry.find("atom:title", self.namespace).text.strip().replace("\n", " ")
            authors = [author.find("atom:name", self.namespace).text for author in entry.findall("atom:author", self.namespace)]
            abstract = entry.find("atom:summary", self.namespace).text.strip()
            published_date = entry.find("atom:published", self.namespace).text
            categories = [cat.attrib["term"] for cat in entry.findall("atom:category", self.namespace)]
            
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                categories=categories
            )
            papers.append(paper)
        return papers
    
    def save_to_json(self, papers: List[ArxivPaper], file_path: str = "raw_arxiv_papers.json"):
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / file_path
        with open(output_path, "w", encoding="utf-8") as f: 
            json_data = [paper.model_dump() for paper in papers]
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"{len(papers)} papers saved to {output_path}")

if __name__ == "__main__":
    client = ArxivClient()
    data = client.fetch_papers(max_res=50)
    client.save_to_json(data)