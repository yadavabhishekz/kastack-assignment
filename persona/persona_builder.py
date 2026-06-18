import csv
import re
import json
import os
import sys
from pathlib import Path
from groq import Groq


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL, PERSONA_OUTPUT, CSV_PATH, PERSONA_BATCH_SIZE


OUTPUT_PATH = PERSONA_OUTPUT
BATCH_SIZE  = PERSONA_BATCH_SIZE

client = Groq(api_key=GROQ_API_KEY)


def load_user1_messages(csv_path: str) -> list[str]:
    
    user1_messages = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lines = row[0].strip().split("\n")
            for line in lines:
                line = line.strip()
                
                if line.startswith("User 1:"):
                    
                    text = line[len("User 1:"):].strip()
                    if text:
                        user1_messages.append(text)

    print(f"User 1 messages loaded: {len(user1_messages)}")
    return user1_messages


def extract_persona_from_batch(messages: list[str]) -> dict:
    
    messages_text = "\n".join(messages[:BATCH_SIZE])

    prompt = f"""You are analyzing messages written by a single person (User 1) to extract their persona.

RULES:
- Only include information that is EXPLICITLY mentioned in the messages
- Do NOT guess or hallucinate
- If something is not mentioned, leave the list empty
- Return ONLY valid JSON, no explanation

Extract the following from these messages:
1. habits: recurring activities or routines they mention (e.g. "exercises regularly", "reads books")
2. personal_facts: concrete facts about them (e.g. "student", "lives in Portland", "has a cat")
3. personality_traits: how they come across (e.g. "friendly", "curious", "humorous")
4. communication_style:
   - average_message_length: "short" / "medium" / "long"
   - emoji_usage: "none" / "occasional" / "frequent"
   - tone: "casual" / "formal" / "mixed"
   - response_style: "brief" / "detailed" / "mixed"

Messages:
{messages_text[:4000]}

Return ONLY this JSON structure:
{{
  "habits": [],
  "personal_facts": [],
  "personality_traits": [],
  "communication_style": {{
    "average_message_length": "",
    "emoji_usage": "",
    "tone": "",
    "response_style": ""
  }}
}}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   
        max_tokens=600,
    )
    raw = response.choices[0].message.content.strip()

    
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Warning: Could not parse JSON from batch. Raw response: {raw[:200]}")
        return {
            "habits": [], "personal_facts": [],
            "personality_traits": [],
            "communication_style": {
                "average_message_length": "", "emoji_usage": "",
                "tone": "", "response_style": ""
            }
        }


def merge_personas(persona_list: list[dict]) -> dict:
    
    merged_habits = []
    merged_facts = []
    merged_traits = []
    style_fields = {
        "average_message_length": [],
        "emoji_usage": [],
        "tone": [],
        "response_style": []
    }

    for p in persona_list:
        merged_habits.extend(p.get("habits", []))
        merged_facts.extend(p.get("personal_facts", []))
        merged_traits.extend(p.get("personality_traits", []))

        cs = p.get("communication_style", {})
        for key in style_fields:
            val = cs.get(key, "")
            if val:
                style_fields[key].append(val)

    def most_common(lst):
        if not lst:
            return ""
        return max(set(lst), key=lst.count)

    
    def dedup(lst):
        seen = set()
        result = []
        for item in lst:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    return {
        "habits": dedup(merged_habits),
        "personal_facts": dedup(merged_facts),
        "personality_traits": dedup(merged_traits),
        "communication_style": {k: most_common(v) for k, v in style_fields.items()}
    }


def build_persona(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> dict:
    
    messages = load_user1_messages(csv_path)

    
    persona_batches = []
    for i in range(0, min(len(messages), 2000), BATCH_SIZE):
        batch = messages[i:i + BATCH_SIZE]
        print(f"  Processing batch {i//BATCH_SIZE + 1} ({len(batch)} messages)...")
        persona = extract_persona_from_batch(batch)
        persona_batches.append(persona)

    
    final_persona = merge_personas(persona_batches)

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_persona, f, indent=2, ensure_ascii=False)

    
    return final_persona


if __name__ == "__main__":
    build_persona()
