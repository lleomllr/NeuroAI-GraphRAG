from openai import OpenAI
from src.config import config

class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY
        )
        self.model = config.LLM_MODEL

    def generate_answer(self, question: str, context: str) -> str:
        system_prompt = """
        You are a research assistant specialising in artificial intelligence.
        Your task is to answer the user's question based solely on the context provided.
        
        Strict rules:
        1. Use the information from the “RAW TEXTUAL CONTEXT” and the "STRUCTURED GRAPH CONTEXT".
        2. If the answer is not found in the context, state clearly: 'I do not have enough information in my knowledge base to answer.' DO NOT BE CREATIVE.
        3. Explicitly cite the names of papers or entities (Concepts/Models) when you use them to formulate your answer.
        4. Be clear, concise, and structure your answer with bullet points if necessary.
        """

        user_prompt = f"=== GIVEN CONTEXT ===\n{context}\n\n=== USER QUESTION ===\n{question}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error occurred while generating the answer: {e}"