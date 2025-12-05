import config
import glob
import os
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter




def extract_text_from_pdf(filename):
    text = ""
    with open(f"{config.PAPERS_DIR}/{filename}", "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
    
        num_pages = len(pdf_reader.pages)
        print(f"{filename} has {num_pages} pages")

        # Extract text from each page
        for page_number, page in enumerate(pdf_reader.pages):
            text += f"\n--- Page {page_number + 1} ---\n"
            text += page.extract_text()
    return text


def check_pages(chunk, curr_page):
        delimiters = re.findall(r'\n?--- Page (\d+) ---\n?', chunk)
        if delimiters:
            pages = [int(m) for m in delimiters]
            min_page = min(pages)
            max_page = max(pages)
            pages_str = f"{min_page}-{max_page}" if min_page != max_page else str(min_page)
            page = max_page  # Update current for next chunks
        else:
            pages_str = str(curr_page)
            page = curr_page
        return page, pages_str
        
def chunk_paper(text, paper):
    chunked_paper = []
    text_chunker = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        length_function = len,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )

    chunks = text_chunker.split_text(text)

    curr_page = 1
    for chunk in chunks:
        curr_page, pages_str = check_pages(chunk, curr_page)
        clean_chunk = re.sub(r'\n--- Page \d+ ---\n', '', chunk).strip()
        chunked_paper.append({
            'text': clean_chunk,
            'metadata': {
                'source': paper,
                'page(s)' : pages_str
            } 
        })

    return chunked_paper

def add_chunks_to_collection(collection, chunks, embedding_model):
    # Prepare data
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Create embeddings ONCE
    embeddings = embedding_model.encode(texts)
    
    # ChromaDB optimized search index
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

def index_papers():
    pdf_paths = glob.glob(os.path.join(config.PAPERS_DIR, "*.pdf"))

    all_chunks = []

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")

        text = extract_text_from_pdf(filename)

        chunks = chunk_paper(text, filename)
        print(f"{len(chunks)} chunks from {filename}")

        all_chunks.extend(chunks)

    
    print(f"    → Created {len(all_chunks)} chunks")

    os.makedirs(config.INDEX_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=config.INDEX_PATH)
    try:
        client.delete_collection(name="papers")
    except:
        pass
    collection = client.create_collection(name="papers")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    add_chunks_to_collection(collection, all_chunks, embedding_model)


