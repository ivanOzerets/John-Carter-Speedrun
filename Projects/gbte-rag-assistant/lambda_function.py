import json
import time
import boto3
import psycopg
from pgvector.psycopg import register_vector

DB_HOST_NAME = "database-1.cluster-cbwe4aco06dj.us-east-2.rds.amazonaws.com"
PORT = 5432
DB_USERNAME = "gbte_app"
REGION = "us-east-2"
ACCEPT = "application/json"
CONTENT_TYPE = "application/json"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
SEARCH_SQL = "SELECT text, title, url, start_time, end_time, embedding <=> %s::vector AS distance FROM chunks ORDER BY embedding <=> %s::vector LIMIT 5"
SIMILARITY_THRESHOLD = 0.7
PROMPT_TEMPLATE = (
    "Answer the question directly and naturally, as if you simply know the answer yourself. "
    "Do not use hedging phrases like \"appears to be\" or \"seems to\", and do not reference "
    "\"the context\" or mention where the information came from. Do not respond in first person "
    "(avoid words like \"I\" or \"my\"); write in a neutral, third-person, informational tone. "
    "Only use the information provided below to answer. Do not add any outside knowledge or guesses. "
    "If the information below does not clearly answer the question, state plainly that this "
    "information is not available, rather than making up an answer. "
    "Avoid using em-dashes; prefer commas or periods instead.\n\n"
    "Information:\n{context}\n\nQuestion: {question}"
)

bedrock = boto3.client(service_name='bedrock-runtime')

# embed the question
def embed_question(question: str) -> list:
    q_json = json.dumps({ "inputText": question })
    response = bedrock.invoke_model(body=q_json, modelId=EMBED_MODEL_ID, accept=ACCEPT, contentType=CONTENT_TYPE)
    response_body = json.loads(response.get('body').read())
    q_embedding = response_body['embedding']

    return q_embedding

# find the closest related chunks
def search_chunks(embedding: list) -> list:
    # generate auth token
    rds_client = boto3.client("rds")
    token = rds_client.generate_db_auth_token(DBHostname=DB_HOST_NAME, Port=PORT, DBUsername=DB_USERNAME, Region=REGION)

    with psycopg.connect(host=DB_HOST_NAME, port=PORT, dbname="postgres", user=DB_USERNAME, password=token, sslmode="require") as conn, \
        conn.cursor() as cur:

        cur.execute(SEARCH_SQL, (embedding, embedding))
        raw_results = cur.fetchall()

    results = []

    for text, title, url, start_time, end_time, distance in raw_results:
        if distance < SIMILARITY_THRESHOLD:
            results.append((text, title, url, start_time, end_time))

    return results

# prompt bedrock for response from constructed query
def generate_answer(question: str, results: list) -> tuple[str, list]:
    context_parts = []
    citations = []

    for text, title, url, start_time, end_time in results:
        context_parts.append(f"[{title}]: {text}")
        citations.append({ "title": title, "url": f"{url}&t={int(start_time)}s" })

    context_block = "\n\n".join(context_parts)

    gen_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            { "role": "user", "content": PROMPT_TEMPLATE.format(context=context_block, question=question) }
        ]
    })

    gen_response = bedrock.invoke_model(body=gen_body, modelId=GENERATION_MODEL_ID, accept=ACCEPT, contentType=CONTENT_TYPE)
    gen_response_body = json.loads(gen_response.get('body').read())
    answer_text = gen_response_body["content"][0]["text"]

    return answer_text, citations

def process_question(question: str) -> dict:
    q_embedding = embed_question(question)
    results = search_chunks(q_embedding)

    if not results:
        return {"answer": "I don't have enough context to answer this question.", "citations": []}

    answer_text, citations = generate_answer(question, results)

    return {"answer": answer_text, "citations": citations}

def handler(event, context):
    body = json.loads(event["body"])
    result = process_question(body["question"])

    return {"statusCode": 200, "body": json.dumps(result)}

if __name__ == "__main__":
    test_event = { "body": json.dumps({ "question": "What is the grip on the forehand?" })}
    print(handler(test_event, None))
