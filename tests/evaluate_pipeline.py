import json
from langchain_ollama import ChatOllama
from src.retrieval.graph_retriever import GraphRetriever
from src.generation.answer_gen import AnswerGenerator

judge_llm = ChatOllama(model="llama3", temperature=0)

def evaluate_with_judge(question, context, answer):
    prompt = f"""
    Tu es un expert en évaluation de systèmes RAG. Ta tâche est de noter la réponse générée sur une échelle de 1 à 5.
    
    Question: {question}
    Contexte fourni: {context}
    Réponse générée: {answer}
    
    Critères :
    - Fidélité (La réponse est-elle basée uniquement sur le contexte ?): 1 à 5
    - Pertinence (La réponse répond-elle précisément à la question ?): 1 à 5
    
    Réponds EXCLUSIVEMENT au format JSON suivant :
    {{"fidelity": <note>, "relevance": <note>, "reason": "<courte explication>"}}
    """
    
    response = judge_llm.invoke(prompt)
    content = response.content
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        return json.loads(content[start:end])
    except:
        return {"fidelity": 0, "relevance": 0, "reason": "Format error"}

def run_evaluation():
    retriever = GraphRetriever()
    generator = AnswerGenerator()
    
    with open("tests/data/dataset_large.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = []
    
    print(f"Starting evaluation on {len(dataset)} questions...\n")
    
    for item in dataset:
        q = item["question"]
        context = retriever.get_hybrid_context(q, top_k=3)
        answer = generator.generate_answer(q, context)
        score = evaluate_with_judge(q, context, answer)
        
        results.append({"question": q, "score": score})
        print(f"Question : {q[:50]}...")
        print(f"Score : {score['fidelity']}/5 (Fidelity), {score['relevance']}/5 (relevance)")
        print(f"Juge : {score['reason']}\n")

    retriever.close()
    return results

if __name__ == "__main__":
    run_evaluation()