"""
Retrieval-Augmented Generation (RAG) with Amazon Bedrock
--------------------------------------------------------------------
Prerequisites:
    pip install boto3 numpy chromadb

AWS credentials must be configured:
    aws configure

Objectives:
    - Implement a basic RAG system using Amazon Bedrock
    - Select appropriate Bedrock models for embedding and text generation
    - Build a document indexing system using ChromaDB vector store
    - Develop a retrieval mechanism based on semantic similarity
    - Integrate retrieved context into prompts for improved text generation
    - Compare RAG vs. non-RAG responses to evaluate effectiveness
"""

import json
import boto3
import chromadb

# ──────────────────────────────────────────────
# PART 2: Initialize Bedrock Client & Models
# ──────────────────────────────────────────────

# NOTE: Must use 'bedrock-runtime' for model inference.
# This creates a connection to Amazon Bedrock.
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Nova Embeddings converts text to vectors; Nova Pro generates human-readable answers.
EMBEDDING_MODEL       = "amazon.nova-2-multimodal-embeddings-v1:0"
TEXT_GENERATION_MODEL = "amazon.nova-pro-v1:0"


def get_bedrock_embedding(text: str) -> list[float]:
    """
    Generate an embedding for one piece of text using Nova Embeddings.

    - SINGLE_EMBEDDING: embed one piece of text at a time
    - GENERIC_INDEX: embedding purpose suited to indexing documents
    - embeddingDimension 1024: the text becomes a list of 1,024 numbers
    - truncationMode END: if the text is too long, cut it from the end
    """
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
    """
    Generate a text response using Amazon Nova Pro.
    """
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

print("=" * 60)
print("PART 3: Setting up ChromaDB Vector Store")
print("=" * 60)

# ChromaDB runs in memory: no server and no database file.
# The collection resets every time the script runs.
chroma_client = chromadb.Client()

# Use cosine similarity so retrieval matches the similarity measure
# used in the embeddings exercise (ChromaDB defaults to L2 distance).
collection = chroma_client.create_collection(
    name="bedrock_docs",
    metadata={"hnsw:space": "cosine"}
)

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
    Indexing step. For each document:
      1. Call Bedrock to get its embedding vector
      2. Store both the original text and the vector in ChromaDB
      3. Give each document a unique ID
    """
    print(f"Indexing {len(docs)} documents...")
    embeddings = [get_bedrock_embedding(doc) for doc in docs]
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )
    print("Documents indexed successfully!\n")


add_documents(sample_docs)


# ──────────────────────────────────────────────
# PART 4: RAG System Implementation
# ──────────────────────────────────────────────

print("=" * 60)
print("PART 4: RAG System")
print("=" * 60)


def rag_generate(query: str, top_k: int = 2) -> str:
    """
    Full RAG pipeline:
      1. Embed the query
      2. Retrieve the top_k most similar documents from ChromaDB
      3. Build a prompt with the retrieved context
      4. Generate a response using Amazon Nova Pro
    """
    # Step 1: Embed the query
    query_embedding = get_bedrock_embedding(query)

    # Step 2: Retrieve relevant documents.
    # ChromaDB finds the top_k documents whose vectors are closest to the query vector.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]

    # Step 3: Stitch the retrieved documents into the prompt as context.
    # This is what makes it RAG: the model sees your documents before answering.
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer the question accurately.

Context:
{context}

Question: {query}

Answer based on the context provided:"""

    # Step 4: Generate response
    return generate_text(prompt)


def generate_without_rag(query: str) -> str:
    """
    Generate a response using Amazon Nova Pro WITHOUT any retrieved context.
    Used for comparison against RAG responses.
    """
    prompt = f"Answer this question as best you can: {query}"
    return generate_text(prompt)


# Test a single query first
test_query = "How does Amazon Bedrock relate to RAG systems?"
print(f"Query: {test_query}")
print(f"Response: {rag_generate(test_query)}\n")


# ──────────────────────────────────────────────
# PART 5: RAG vs. Non-RAG Comparison
# ──────────────────────────────────────────────

print("=" * 60)
print("PART 5: RAG vs. Non-RAG Comparison")
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

print("\nRAG vs. non-RAG comparison complete.")
print("""
Key takeaway:
  RAG responses are grounded in your documents: specific and controlled.
  Non-RAG responses come purely from the model's training data: broader,
  but potentially less accurate for domain-specific questions.
""")
