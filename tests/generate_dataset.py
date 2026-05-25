import json
import os
from langchain_ollama import ChatOllama

llm_gen = ChatOllama(model="llama3", temperature=0.7)

def generate_golden_dataset(documents, output_file="tests/data/dataset_large.json"):
    full_dataset = []
    
    print(f"Starting question generation on {len(documents)} documents...")

    for i, doc in enumerate(documents):
        print(f"Document {i+1}/{len(documents)}...")
        
        prompt = f"""
        Tu es un ingénieur QA spécialisé en RAG (Retrieval Augmented Generation).
        Ta mission est de créer un jeu de test pour évaluer un système GraphRAG.
        
        Voici un document :
        ---
        {doc}
        ---
        
        Génère 3 paires "Question / Réponse" basées UNIQUEMENT sur ce texte.
        Consignes strictes :
        1. Les questions doivent être complexes (multi-sauts) si possible : elles doivent demander de lier deux concepts ou deux entités mentionnées dans le texte.
        2. La réponse doit être précise et complète.
        3. Réponds UNIQUEMENT avec un JSON valide sous la forme suivante, sans préambule :
        [
            {{"question": "...", "ground_truth": "..."}},
            ...
        ]
        """
        
        try:
            response = llm_gen.invoke(prompt)
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            json_part = content[start:end]
            questions = json.loads(json_part)
            full_dataset.extend(questions)
        except Exception as e:
            print(f"Error on document {i+1}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"\nDataset generated with success in {output_file} ({len(full_dataset)} questions).")

if __name__ == "__main__":
    my_doc = [
        """SciCore-Mol: Augmenting Large Language Models with Pluggable Molecular Cognition Modules. 
        Large Language Models (LLMs) face challenges with molecular data. We propose SciCore-Mol, 
        a framework with three modules: a topology-aware perception module, a latent diffusion-based 
        molecular generation module, and a reaction-aware reasoning module. It bridges discrete symbols 
        and continuous data, improving performance on chemical tasks.""",
        
        """Evaluating Large Language Models as Live Strategic Agents: Provider Performance, 
        Hybrid Decomposition, and Operational Gaps in Timed Risk Play. We study LLMs in a 
        timed multi-phase Risk environment. In a replicated 32-game cross-provider championship, 
        gemini-3.1-pro-preview won 20 of 32 games against gpt-5.1, claude-opus-4-7, and kimi-k2.6. 
        We show that live-agent performance depends on objective tracking, execution conversion, 
        cost, and runtime reliability.""",
        
        """A Multi-Source Framework for Relational Validation of Large Language Models 
        Using Expert-Curated Encyclopedic Sources. This paper introduces a multi-source framework 
        for the relational validation of LLMs. By comparing LLM-generated knowledge graphs to 
        expert-curated encyclopedias, we reveal a consistent 'relational deficit': LLMs recognize 
        domain-specific concepts but consistently fail to reproduce their relational structure."""
    ]
    
    generate_golden_dataset(my_doc)