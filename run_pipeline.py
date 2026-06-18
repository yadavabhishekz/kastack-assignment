import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

from config import CSV_PATH, TOPIC_OUTPUT, CHECKPOINT_OUTPUT
from topic_detection.detector import load_all_messages, detect_topics
from topic_detection.summarizer import (
    summarize_topics,
    create_100_message_checkpoints,
    save_topics,
    save_checkpoints
)
from persona.persona_builder import build_persona
from retrieval.rag import (
    index_topic_summaries,
    index_message_checkpoints,
    index_raw_message_chunks
)


def main():
   
    print("STEP 1: Load all messages from CSV")
    messages = load_all_messages(str(CSV_PATH))

    print("STEP 2: Detect topic segments")
    topics = detect_topics(messages)


    print("STEP 3: Summarize each topic segment (calls LLM)")
    topics_with_summaries = summarize_topics(topics)
    save_topics(topics_with_summaries, str(TOPIC_OUTPUT))


    print("STEP 4: Create 100-message checkpoints (calls LLM)")
    checkpoints = create_100_message_checkpoints(messages)
    save_checkpoints(checkpoints, str(CHECKPOINT_OUTPUT))


    print("STEP 5: Extract user persona (calls LLM)")
    build_persona(str(CSV_PATH))


    print("STEP 6: Index everything into ChromaDB")
    index_topic_summaries(TOPIC_OUTPUT)
    index_message_checkpoints(CHECKPOINT_OUTPUT)
    index_raw_message_chunks()


    print("!!!!!!Pipeline complete!!!!!!!!!")



if __name__ == "__main__":
    main()
