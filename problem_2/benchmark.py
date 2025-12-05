from handle_query import rag_query, ping_llm
import config

def vanilla_query(query):
    system_prompt = """You are a helpful research assistant. concisely answer 
    questions based on what you know about research papers CellFM, scGPT, and 
    Geneformer."""
    vanilla_response = ping_llm(system_prompt, query)
    return vanilla_response

def compare_rag_vanilla(query):
    vanilla_response = vanilla_query(query)
    rag_response = rag_query(query)

    evaluation_query = f"{query}; Answers: A - {vanilla_response.content}, B - {rag_response.content}"
    evaluation_response = rag_query(evaluation_query, evaluation=True)
    return {
        "question": query,
        "vanilla_response": vanilla_response.content,
        "rag_response": rag_response.content,
        "evaluation": evaluation_response.content,
    }

def main():
    questions = config.TEST_QUESTIONS
    results = []

    for question in questions:
        result = compare_rag_vanilla(question)
        results.append(result)

    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            f.write(f"### Question {i}:\n{r['question']}\n\n")
            f.write(f"**Vanilla Response:**\n{r['vanilla_response']}\n\n")
            f.write(f"**RAG Response:**\n{r['rag_response']}\n\n")
            f.write(f"**Evaluation:**\n{r['evaluation']}\n\n")
            f.write("---\n\n")


if __name__ == "__main__":
    main()