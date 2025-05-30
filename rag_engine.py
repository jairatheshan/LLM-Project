import fitz
import uuid
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import yaml


yaml_file_path = "API_Cred-copy.yaml"
# Azure OpenAI creds
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)
# openai
API_KEY = config['Open_ai_credentails']['API_KEY']
RESOURCE_ENDPOINT = config['Open_ai_credentails']['RESOURCE_ENDPOINT']
MODEL = config['Open_ai_credentails']['MODEL']
API_VERSION = config['Open_ai_credentails']['API_VERSION']

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def smart_split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(chunks, convert_to_numpy=True)

def store_in_chromadb(chunks, embeddings, db_path="./chroma_store"):
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection("pdf_docs")
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"source": "my_pdf"}] * len(chunks),
        ids=[str(uuid.uuid4()) for _ in chunks]
    )

def answer_query(
    question,
    embedding_model='multi-qa-MiniLM-L6-cos-v1',
    db_path="./chroma_store",
    similarity_threshold=0.85,
    temperature=0.4,
    top_p=0.95,
    chunk_size=500,
    chunk_overlap=50
):
    model = SentenceTransformer(embedding_model)
    question_embedding = model.encode([question])[0].tolist()

    client_db = PersistentClient(path=db_path)
    collection = client_db.get_collection("pdf_docs")

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    if len(distances) == 0 or distances[0] > similarity_threshold:
        return "No idea, this info is not in the documents."

    context = "\n\n".join(documents)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer using the context below. Say 'No idea...' if the answer isn't found.\n"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=RESOURCE_ENDPOINT
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1000
    )

    return response.choices[0].message.content
