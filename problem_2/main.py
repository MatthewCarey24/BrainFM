import argparse
from handle_query import rag_query
from index_papers import index_papers
import config
import os

def main():
    parser = argparse.ArgumentParser(description="RAG Workflow for Papers QA")
    parser.add_argument('--query', type=str, 
                        help="The query string")
    parser.add_argument('--index', action='store_true',
                        help="Index papers before querying (creates new index)")
    parser.add_argument('--k', type=int,
                        help="Number of chunks to retrieve (default: 5)")

    args = parser.parse_args()
    
    # Index papers if requested
    if args.index:
        print("Indexing papers...")
        index_papers()
        print(f"Indexing complete. Saved to {config.INDEX_PATH}")
    
    # Check if index exists
    if not os.path.exists(config.INDEX_PATH):
        parser.error(f"Index not found at {config.INDEX_PATH}. \
                     Run with --index first.")
    
    # Handle query
    if args.query:
        print(f"\nYou: {args.query}")
        response = rag_query(args.query, config.INDEX_PATH, k=args.k)
        print(f"\nRAG Response:\n{response}")


if __name__ == "__main__":
    main()