import os
import json
from tqdm import tqdm
import boto3
import psycopg
from pgvector.psycopg import register_vector

DB_HOST_NAME = "database-1.cluster-cbwe4aco06dj.us-east-2.rds.amazonaws.com"
PORT = 5432
DB_USERNAME = "gbte_app"
REGION = "us-east-2"
FILE_DIR = "data/"
FILE_NAME = "output.jsonl"
INSERT_SQL = "INSERT INTO chunks (text, embedding, start_time, end_time, title, url) VALUES (%s, %s, %s, %s, %s, %s)"
LINE_COUNT = 24486

file_path = os.path.join(FILE_DIR, FILE_NAME)

# generate auth token
rds_client = boto3.client("rds")
token = rds_client.generate_db_auth_token(DBHostname=DB_HOST_NAME, Port=PORT, DBUsername=DB_USERNAME, Region=REGION)

# open psycopg3 connection
with psycopg.connect(host=DB_HOST_NAME, port=PORT, dbname="postgres", user=DB_USERNAME, password=token, sslmode="require") as conn, \
    open(file_path, "r", encoding="utf-8") as f, \
    conn.cursor() as cur:

    # register the vector adapter
    register_vector(conn)

    batch_list = []

    # loop through and build each row's values
    for line in tqdm(f, total=LINE_COUNT):
        row = json.loads(line)
        values = (row["text"], row["embedding"], row["start"], row["end"], row["title"], row["url"])
        batch_list.append(values)

        if len(batch_list) >= 500:
            cur.executemany(INSERT_SQL, batch_list)
            batch_list = []

    # batch the inserts
    if batch_list:
        cur.executemany(INSERT_SQL, batch_list)
