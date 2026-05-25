import logging
from src.retrieval.graph_retriever import GraphRetriever
from src.generation.answer_gen import AnswerGenerator

logging.basicConfig(level=logging.WARNING)

def main():
    print("GraphRAG pipeline initialisation...")
    retriever = GraphRetriever()
    generator = AnswerGenerator()
    
    print("\n" + "="*50)
    print("GraphRAG Assistant ready! (type 'exit' to quit)")
    print("="*50)

    try:
        while True:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            if not question.strip():
                continue

            print("\n1. Search in the vector and graph databases (Qdrant + Neo4j)...")
            context = retriever.get_hybrid_context(question, top_k=3)
            
            print("2. Generate the answer (Ollama)...")
            answer = generator.generate_answer(question, context)
            
            print("\n" + "="*50)
            print("ANSWER:")
            print(answer)
            print("="*50)
            
            print("\n[DEBUG] Used Context:")
            print(context)
            
    except KeyboardInterrupt:
        print("\nAssistant stopped by user")
    finally:
        retriever.close()

if __name__ == "__main__":
    main()