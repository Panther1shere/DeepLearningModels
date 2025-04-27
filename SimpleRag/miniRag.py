from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline
import numpy as np
from file_reader import FileReader
from text_chunker import TextChunker
from sklearn.preprocessing import normalize

# 1. Load the text file
fileReader = FileReader()
stringOfFile = fileReader.load_text_file("./text.txt")
print("Loaded text preview:", stringOfFile[:100])

# 2. Chunk the text
chunker = TextChunker(chunk_size=300, overlap=50)
documents = chunker.chunk_text(stringOfFile)

# 3. Create Embeddings for Documents
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)
document_embeddings = normalize(document_embeddings)

# 4. Create FAISS Index
dimension = document_embeddings.shape[1]
print("Embeddings shape:", document_embeddings.shape)
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

# 5. Load text2text-generation model (Flan-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small", max_length=300)

# 6. Helper Functions
def rerank_chunks(question, retrieved_chunks):
    """Rerank retrieved chunks based on cosine similarity with the question."""
    question_embedding = embedding_model.encode([question])
    chunk_embeddings = embedding_model.encode(retrieved_chunks)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    sorted_indices = np.argsort(-scores)  # Higher score first
    return [retrieved_chunks[i] for i in sorted_indices]

# manual domain detection
def detect_domain(question):
    if "example" in question.lower():
        return "examples"
    elif "usage" in question.lower() or "used" in question.lower():
        return "usage"
    elif "difference" in question.lower():
        return "comparison"
    else:
        return "general"

def rag_qa(user_question, top_k=5):
    """Main RAG flow: retrieve, rerank, answer generation."""
    # Embed and normalize question
    question_embedding = embedding_model.encode([user_question])
    question_embedding = normalize(question_embedding)

    # Search top-k relevant chunks
    distances, indices = index.search(np.array(question_embedding), top_k)
    retrieved_chunks = [documents[i] for i in indices[0]]

    # Rerank the retrieved chunks
    reranked_chunks = rerank_chunks(user_question, retrieved_chunks)

    # Use top 2 reranked chunks for final context
    selected_chunks = reranked_chunks[:2]
    context = "\n".join(selected_chunks)

    # Prepare prompt
    domain = detect_domain(user_question)
    prompt = f"Domain: {domain}\nAnswer the following question based on the context.\n\nContext:\n{context}\n\nQuestion: {user_question}"

    # Generate answer
    answer = qa_model(prompt)[0]['generated_text']

    return answer

# 7. Start the chat
if __name__ == "__main__":
    print("\n Welcome to your custom RAG system!\n")
    while True:
        user_question = input("Enter your question (or 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Thank you for using the RAG system!")
            break
        try:
            response = rag_qa(user_question)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"⚠️ Error: {e}")