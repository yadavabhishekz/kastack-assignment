import json
import os
import sys
from pathlib import Path
from groq import Groq


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOPIC_OUTPUT,
    CHECKPOINT_OUTPUT,
    CHECKPOINT_SIZE,
)

client = Groq(api_key=GROQ_API_KEY)


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def summarize_messages(messages: list[str]) -> str:
    
    conversation_block = "\n".join(messages)

    prompt = f"""Summarize the following conversation in 2-3 sentences. 
Focus on the main topics discussed. Be concise.

Conversation:
{conversation_block[:3000]}  

Summary:"""

    return call_llm(prompt)


def summarize_topics(topics: list[dict]) -> list[dict]:
    
    print(f"Summarizing {len(topics)} topics...")
    for i, topic in enumerate(topics):
        print(f"  Summarizing topic {i+1}/{len(topics)}...")
        topic["summary"] = summarize_messages(topic["messages"])
    return topics


def create_100_message_checkpoints(all_messages: list[str]) -> list[dict]:
    
    checkpoints = []
    total = len(all_messages)

    print(f"Creating 100-message checkpoints for {total} messages...")
    for start in range(0, total, CHECKPOINT_SIZE):
        end = min(start + CHECKPOINT_SIZE, total)
        chunk = all_messages[start:end]

        
        checkpoint = {
            "checkpoint_id": len(checkpoints) + 1,
            "start_message": start + 1,
            "end_message": end,
            "message_count": len(chunk),
            "summary": summarize_messages(chunk)
        }
        checkpoints.append(checkpoint)
        print(f"  Checkpoint {checkpoint['checkpoint_id']}: messages {start+1}–{end}")

    return checkpoints


def save_topics(topics: list[dict], path: str):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    print(f"Saved topic checkpoints → {path}")


def save_checkpoints(checkpoints: list[dict], path: str):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)
    print(f"Saved 100-message checkpoints → {path}")
