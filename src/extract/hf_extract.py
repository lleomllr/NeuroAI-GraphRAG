import json 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import ValidationError 

from src.extract.base_extractor import BaseExtractor
from src.extract.schema import GraphExtraction
from src.config import config

class HFExtractor(BaseExtractor):
    def __init__(self):
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"Loading {model_id} with 4-bit quantization...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config,
            device_map="auto"
        )

    def extract_graph_from_text(self, text: str) -> GraphExtraction | None:
        prompt = f"""
        You are an expert in AI and computational neuroscience. 
        Your aim is to extract a knowledge graph from the text provided.
        You must identify the Nodes (Concepts, Models, Authors) and the Edges (Relationships) between them.
        
        Rules:
        1. Return ONLY a valid JSON object. No text before or after.
        2. The JSON must have two keys: 'nodes' and 'edges'.
        3. Possible node labels are: Concept, Model, Author.
        4. Possible relationship (edge) types are: INSPIRED_BY, RELATED_TO, AUTHORED, APPLIED_TO.        
        """

        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Texte :\n{text}\n\nJSON strict :"}
        ]

        txt_input = self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([txt_input], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_txt = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        try:
            start_idx = response_txt.find('{')
            end_idx = response_txt.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in the response")
            
            clean_json = response_txt[start_idx:end_idx]
            parsed_data = json.loads(clean_json)
            return GraphExtraction(**parsed_data)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Error extracting HF: {e}\nResponse was: {response_txt}")
            return None