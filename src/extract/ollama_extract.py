import instructor
from openai import OpenAI
from pydantic import ValidationError

from src.config import config
from src.extract.schema import GraphExtraction

class LocalLLMExtractor:
    def __init__(self):
        client = OpenAI(
            base_url=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY
        )
        self.client = instructor.from_openai(
            client,
            mode=instructor.Mode.MD_JSON 
        )
        self.model = config.LLM_MODEL

    def extract_graph_from_text(self, text: str) -> GraphExtraction | None:
        system_prompt = """
        You are a strict, robotic data extractor. 
        Your ONLY purpose is to output a strictly valid, parsable JSON object representing a knowledge graph.
        
        CRITICAL JSON SYNTAX RULES:
        1. NO COMMENTS: Never use // or /* */.
        2. NO TRUNCATION: Never use ellipses (...) or skip items. Output the complete JSON.
        3. NO PLACEHOLDERS: Never use tags like <author_name>. If a value is missing in the text, use "Unknown".
        4. NO CONVERSATION: Never output introductory or explanatory text (e.g., do not say "Here is the JSON"). 
        Output ONLY the raw JSON object.

        CRITICAL SCHEMA RULES:
        - Node 'label' MUST be exactly one of: "Paper", "Author", "Model", "Dataset", "Concept", "Metric".
        - Edge 'type' MUST be exactly one of: "AUTHORED_BY", "EVALUATED_ON", "USES_METHOD", "ACHIEVES", "RELATED_TO".
        """

        try:
            graph = self.client.chat.completions.create(
                model=self.model,
                response_model=GraphExtraction,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyse ce texte:\n{text}"}
                ],
                temperature=0.1
            )
            return graph

        except Exception as e:
            print(f"Critical failure in extraction (even after attempts): {e}")
            return None

if __name__ == "__main__":
    extractor = LocalLLMExtractor()
    sample_text = "The Transformer architecture, introduced by Vaswani et al. in 2017, revolutionized natural language processing. It was inspired by the limitations of recurrent neural networks and convolutional neural networks in handling long-range dependencies in text. The key innovation was the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence regardless of their position. This architecture has been applied to various tasks, including machine translation, text summarization, and question answering, leading to significant advancements in the field."
    
    print("Extracting graph from sample text...")
    result = extractor.extract_graph_from_text(sample_text)
    
    if result:
        print("\n--- Nodes ---")
        for n in result.nodes:
            print(f"[{n.label}] {n.name} (ID: {n.id})")
        print("\n--- Edges ---")
        for e in result.edges:
            print(f"{e.source} --({e.type})--> {e.target} | {e.description}")