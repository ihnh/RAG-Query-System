"""
Retrieval-Augmented Generation (RAG) with Amazon Bedrock
--------------------------------------------------------------------
Prerequisites:
    pip install boto3 numpy chromadb

AWS credentials must be configured:
    aws configure

    - Implement a basic RAG system using Amazon Bedrock
    - Select appropriate Bedrock models for embedding and text generation
    - Build a document indexing system using ChromaDB vector store
    - Develop a retrieval mechanism based on semantic similarity
    - Integrate retrieved context into prompts for improved text generation
    - Compare RAG vs Non-RAG responses to evaluate effectiveness
    
"""

import json
import numpy as np
import boto3
import chromadb

# ──────────────────────────────────────────────
# PART 2: Initialize Bedrock Client & Models
# ──────────────────────────────────────────────

# NOTE: Must use 'bedrock-runtime' for model inference
# This creates a connection to AWS Bedrock

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Nova Embeddings converts texts to vectors, Nova Pro generates human readaable answers

EMBEDDING_MODEL      = "amazon.nova-2-multimodal-embeddings-v1:0"
TEXT_GENERATION_MODEL = "amazon.nova-pro-v1:0"

#Until line 44 - Nova specific API format - Single Embediing : embed one piece of text at a time
#Generic Index - Indexing docs, embeddingDimension 1024 : text becoms a list of 1024 numbers, truncationMode : text too long, cut from end

def get_bedrock_embedding(text: str) -> list[float]:
    body = json.dumps({
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": 1024,
            "text": {"truncationMode": "END", "value": text}
        }
    })
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response["body"].read())
    return response_body["embeddings"][0]["embedding"]


def generate_text(prompt: str) -> str:
    body = json.dumps({
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    })
    response = bedrock.invoke_model(
        modelId=TEXT_GENERATION_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response["body"].read())
    return response_body["output"]["message"]["content"][0]["text"]


# ──────────────────────────────────────────────
# PART 3: Document Indexing with ChromaDB
# ──────────────────────────────────────────────
#ChromaDB runs in memory, no server, no database file, reset every time you run the script
print("=" * 60)
print("PART 3: Setting up ChromaDB Vector Store")
print("=" * 60)

# Initialize ChromaDB (runs in-memory, no server needed)
chroma_client = chromadb.Client()

# Create a collection (we'll add embeddings manually)
collection = chroma_client.create_collection(name="bedrock_docs")

# Sample knowledge base documents
sample_docs = [
    "Amazon Bedrock is a fully managed foundation model service by AWS.",
    "RAG systems combine retrieval and generation for improved responses.",
    "Embeddings are vector representations of text in high-dimensional space.",
    "Chroma is an efficient vector store for building AI applications.",
    "Foundation models can be fine-tuned for specific tasks and domains.",
    "Semantic similarity measures how alike two pieces of text are in meaning.",
    "Vector databases store embeddings and allow fast similarity searches.",
]


def add_documents(docs: list[str]):
    """
    Generate embeddings for each document and store them in ChromaDB.
    """
    print(f"Indexing {len(docs)} documents...")
    embeddings = [get_bedrock_embedding(doc) for doc in docs]
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )
    print("Documents indexed successfully!\n")
# Indexing step >> For each document: Call Bedrock to get its embedding vector, Store both the original text AND the vector in ChromaDB, Give each document a unique ID

add_documents(sample_docs)


# ──────────────────────────────────────────────
# PART 4: RAG System Implementation
# ──────────────────────────────────────────────

print("=" * 60)
print("PART 4: RAG System")
print("=" * 60)

#top_k=2 means retrieve the 2 most relevant documents for any query
def rag_generate(query: str, top_k: int = 2) -> str:
    """
    Full RAG pipeline:
      1. Embed the query
      2. Retrieve top_k most similar documents from ChromaDB
      3. Build a prompt with retrieved context
      4. Generate a response using Claude
    """
    # Step 1: Embed the query
    query_embedding = get_bedrock_embedding(query)

    # Step 2: Retrieve relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]
    # The query gets embedded too, then ChromaDB finds the 2 documents whose vectors are mathematically closest to the query vector.
    # Step 3: Build prompt with context
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer the question accurately.
    # stitch the retrieved documents into the prompt as context. This is what makes it RAG — the model sees your documents before answering
    


Context:
{context}

Question: {query}
   

Answer based on the context provided:"""

    # Step 4: Generate response
    return generate_text(prompt)


def generate_without_rag(query: str) -> str:
    """
    Generate a response using NovaPro WITHOUT any retrieved context.
    Used for comparison against RAG responses.
    """
    prompt = f"Answer this question as best you can: {query}"
    return generate_text(prompt)


# Test single query first
test_query = "How does Amazon Bedrock relate to RAG systems?"
print(f"Query: {test_query}")
print(f"Response: {rag_generate(test_query)}\n")


# ──────────────────────────────────────────────
# PART 5: RAG vs Non-RAG Comparison
# ──────────────────────────────────────────────

print("=" * 60)
print("PART 5: RAG vs Non-RAG Comparison")
print("=" * 60)

test_queries = [
    "What are embeddings used for in AI?",
    "Explain the benefits of using RAG in AI applications.",
    "How does Amazon Bedrock support foundation models?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print(f"\n  RAG Response:\n  {rag_generate(query)}")
    print(f"\n  Non-RAG Response:\n  {generate_without_rag(query)}")
    print("\n" + "=" * 60)

print("\nExercise 4 complete! ✓")
print("""
Key takeaway:
  RAG responses are grounded in YOUR documents — specific, controlled.
  Non-RAG responses come purely from the model's training data — broader
  but potentially less accurate for domain-specific questions.
""")
