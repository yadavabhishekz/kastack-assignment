# Conversation Intelligence System

## GitHub Repository

[https://github.com/yadavabhishekz/kastack-assignment]

## Hosted Chatbot URL

[https://kastack-assignment-1.streamlit.app/]

---

# Run Instructions


## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Data Processing Pipeline

```bash
python run_pipeline.py
```

## Start Chatbot

```bash
streamlit run chatbot/app.py
```

---

# How Topic Changes Are Detected

Conversations are processed in chronological order, message by message.

1. A Sentence Transformer model generates embeddings for all messages.
2. A sliding window of 5 messages is used to capture local conversation context.
3. The average embedding of the previous window and next window is calculated.
4. Cosine similarity is computed between the two windows.
5. If the similarity falls below a predefined threshold and the current topic contains at least 200 messages, a new topic checkpoint is created.
6. Each topic checkpoint stores its message range and is later summarized for retrieval.

---

# How Retrieval Works

1. Topic summaries, 100-message checkpoint summaries, and raw conversation chunks are stored separately in ChromaDB using embeddings.
2. When a user asks a question, the query is converted into an embedding.
3. The system searches for the most relevant:
   - Topic summaries
   - Message checkpoint summaries
   - Raw conversation chunks
4. The top matching results from all three sources are retrieved.
5. These results are combined into a single context.
6. The chatbot uses this context to generate the final answer.

This approach helps the chatbot use both high-level conversation summaries and detailed message-level information when answering questions.

---

# How Persona Is Built

1. Messages sent by User 1 are collected from the conversation data.
2. The messages are processed in small batches.
3. Each batch is analyzed to find:
   - Habits
   - Personal facts
   - Personality traits
   - Communication style
4. Only information clearly mentioned in the conversations is included.
5. Information from all batches is combined into a single persona profile.
6. Duplicate information is removed and communication patterns are summarized.
7. The final persona is saved in JSON format and used by the chatbot to answer questions about the user.

The persona is stored in a structured JSON format.

Example:

```json
{
  "habits": [],
  "personal_facts": [],
  "personality_traits": [],
  "communication_style": {}
}
```

---

# Screenshots and Demo


## Topic Checkpoints

![Topic Checkpoints](demo/Topic1.png)
![Topic Checkpoints](demo/Topic2.png)

## 100 Message Checkpoints

![Checkpoints](demo/100msg.png)
![Checkpoints](demo/100msg2.png)

## Persona Extraction

![Persona](demo/persona1.png)
![Persona](demo/persona2.png)

## Chatbot Demo

![Chatbot](demo/image1.png)
![Chatbot](demo/image2.png)


---

# Video Demo (Loom)

