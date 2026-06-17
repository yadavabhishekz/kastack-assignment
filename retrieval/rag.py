import json
import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    TOPIC_OUTPUT,
    CHECKPOINT_OUTPUT,
    TOP_K,
    RAW_CHUNKS_OUTPUT
)
from topic_detection.detector import load_all_messages
from config import CSV_PATH


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


TOPIC_COLLECTION = "topic_summaries"
CHUNK_COLLECTION = "message_chunks"
RAW_COLLECTION = "raw_message_chunks"

print("Loading HuggingFace embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)



def index_topic_summaries(topic_file: str = str(TOPIC_OUTPUT)):

    
    store = Chroma(
        collection_name=TOPIC_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    
    if store._collection.count() > 0:
        print(f"Topic summaries already indexed ({store._collection.count()} items). Skipping.")
        return

    
    with open(topic_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    
    docs = []
    for t in topics:
        if not t.get("summary"):
            continue
        text = (
            f"Topic {t['topic_id']} "
            f"(messages {t['start_message']}–{t['end_message']}): "
            f"{t['summary']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "topic_id":      t["topic_id"],
                "start_message": t["start_message"],
                "end_message":   t["end_message"],
            }
        ))

    
    store.add_documents(docs)
    print(f"Indexed {len(docs)} topic summaries into Chroma.")


def index_message_checkpoints(checkpoint_file: str = str(CHECKPOINT_OUTPUT)):

    store = Chroma(
        collection_name=CHUNK_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if store._collection.count() > 0:
        print(f"Message checkpoints already indexed ({store._collection.count()} items). Skipping.")
        return

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoints = json.load(f)

    docs = []
    for cp in checkpoints:
        if not cp.get("summary"):
            continue
        text = (
            f"Messages {cp['start_message']}–{cp['end_message']}: "
            f"{cp['summary']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "checkpoint_id": cp["checkpoint_id"],
                "start_message": cp["start_message"],
                "end_message":   cp["end_message"],
            }
        ))

    store.add_documents(docs)
    print(f"Indexed {len(docs)} message checkpoints into Chroma.")

def index_raw_message_chunks():

    messages = load_all_messages(str(CSV_PATH))

    store = Chroma(
        collection_name=RAW_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if store._collection.count() > 0:
        print("Raw message chunks already indexed.")
        return

    docs = []

    chunk_size = 20

    for start in range(0, len(messages), chunk_size):

        end = min(start + chunk_size, len(messages))

        chunk_text = "\n".join(
            messages[start:end]
        )

        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "start_message": start + 1,
                    "end_message": end,
                }
            )
        )

    store.add_documents(docs)

    print(f"Indexed {len(docs)} raw chunks.")




def retrieve(query: str, top_k: int = TOP_K) -> dict:
    
    
    topic_store = Chroma(
        collection_name=TOPIC_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    chunk_store = Chroma(
        collection_name=CHUNK_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    raw_store = Chroma(
        collection_name=RAW_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    results = {}

    
    if topic_store._collection.count() > 0:
        
        topic_retriever = topic_store.as_retriever(
            search_type="similarity",           
            search_kwargs={"k": top_k},         
        )
        topic_docs = topic_retriever.invoke(query)
        results["topic_summaries"] = [doc.page_content for doc in topic_docs]
    else:
        results["topic_summaries"] = []

    
    if chunk_store._collection.count() > 0:
        chunk_retriever = chunk_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        chunk_docs = chunk_retriever.invoke(query)
        results["message_chunks"] = [doc.page_content for doc in chunk_docs]
    else:
        results["message_chunks"] = []
    
    if raw_store._collection.count() > 0:

        raw_retriever = raw_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        raw_docs = raw_retriever.invoke(query)

        results["raw_chunks"] = [
            doc.page_content
            for doc in raw_docs
        ]

    else:
        results["raw_chunks"] = []

    return results




def build_context_string(retrieval_results: dict) -> str:
    
    parts = []

    if retrieval_results.get("topic_summaries"):
        parts.append("=== Relevant Topic Summaries ===")
        for item in retrieval_results["topic_summaries"]:
            parts.append(f"- {item}")

    if retrieval_results.get("message_chunks"):
        parts.append("\n=== Relevant Message Chunks ===")
        for item in retrieval_results["message_chunks"]:
            parts.append(f"- {item}")
    if retrieval_results.get("raw_chunks"):
        parts.append("\n=== Relevant Raw Conversation Chunks ===")

        for item in retrieval_results["raw_chunks"]:
            parts.append(f"- {item}")

    return "\n".join(parts)



if __name__ == "__main__":
    index_topic_summaries()
    index_message_checkpoints()
    index_raw_message_chunks()

    test_query = "What hobbies does User 1 have?"
    results = retrieve(test_query)
    print("\nRetrieval results for:", test_query)
    print(build_context_string(results))
