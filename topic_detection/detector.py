import csv
import re
import json
import os
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SIMILARITY_THRESHOLD,
    CSV_PATH,
    TOPIC_OUTPUT,
    EMBEDDING_MODEL,
    TRIAL_MODE,
    TRIAL_MESSAGE_LIMIT
)
model = SentenceTransformer(EMBEDDING_MODEL)


def load_all_messages(csv_path):

    all_messages = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            conversation_text = row[0]
            lines = conversation_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if re.match(r"^User \d+:", line):
                    all_messages.append(line)

    if TRIAL_MODE:
        all_messages = all_messages[:TRIAL_MESSAGE_LIMIT]
        print(f"TRIAL MODE ENABLED - Using first {len(all_messages)} messages")

    print(f"Total messages loaded: {len(all_messages)}")
    return all_messages


def average_embedding(window_embeddings):
    return np.mean(window_embeddings, axis=0)


def detect_topics(messages):

    print("Generating embeddings for all messages...")
    embeddings = model.encode(messages, show_progress_bar=True, batch_size=64)

    topics = []
    
    WINDOW_SIZE = 5
    MIN_TOPIC_SIZE = 100
    
    current_topic_messages = messages[:WINDOW_SIZE]
    current_start = 1  

    

    print("Detecting topic changes...")

    

    for i in range(WINDOW_SIZE, len(messages) - WINDOW_SIZE):

        prev_window = embeddings[i-WINDOW_SIZE:i]
        next_window = embeddings[i:i+WINDOW_SIZE]

        prev_avg = average_embedding(prev_window)
        next_avg = average_embedding(next_window)

        sim = util.cos_sim(
            prev_avg,
            next_avg
        ).item()

        if i % 100 == 0:
            print(f"i={i}, similarity={sim:.3f}")

        if (
            sim < SIMILARITY_THRESHOLD
            and len(current_topic_messages) >= MIN_TOPIC_SIZE
        ):
            topics.append({
                "topic_id": len(topics) + 1,
                "start_message": current_start,
                "end_message": i,
                "messages": current_topic_messages
            })

            current_topic_messages = [messages[i]]
            current_start = i + 1

        else:
            current_topic_messages.append(messages[i])
    
    for i in range(len(messages) - WINDOW_SIZE, len(messages)):
        current_topic_messages.append(messages[i])

    topics.append({
        "topic_id": len(topics) + 1,
        "start_message": current_start,
        "end_message": len(messages),
        "messages": current_topic_messages
    })

    print(f"Total topics detected: {len(topics)}")
    return topics


if __name__ == "__main__":
    messages = load_all_messages(CSV_PATH)
    topics = detect_topics(messages)
    for t in topics[:3]:
        print(f"\nTopic {t['topic_id']}: messages {t['start_message']}–{t['end_message']}")
        print(f"  First message: {t['messages'][0][:80]}")
