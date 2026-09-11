# GBTE RAG Search

A RAG Q&A over GreatBase Tennis Education's course and podcast library, with timestamped citations.

```
Q: "What is the grip on the forehand?"
A: "...third bevel to the right."
   References: Podcast 12: Forehand (link) | TIA Course Lesson 25 (link)
```

A user query in and a contextually accurate answer + reference links out. No chat or conversation memory. A search box embedded on the WordPress site via a `[gbte_search]` shortcode.

## Pipeline

**Ingestion** (`scrape-script.py`) Run once per backlog video/podcast:
```
yt-dlp (audio-only) -> Whisper "medium" (CUDA) -> RecursiveCharacterTextSplitter
(1500 chars / 200 overlap) -> Titan Embed V2 -> appended to data/output.jsonl
```
Each chunk carries its source `title`, `url`, and `start_time`/`end_time` (from mapping the chunk's character offset back to Whisper's segment timestamps), which is what makes the linked citations possible later.

**Loading** (`loader-script.py`) Bulk-loads `output.jsonl` into the `chunks` table in batches of 500.

**Query** (`lambda_function.py`) Live behind API Gateway:
```
question -> embed (Titan V2) -> pgvector cosine search, top 5, distance < 0.7
-> Claude (Sonnet, via Bedrock) answers from retrieved context only
-> {answer, citations} back to the plugin
```
Citations are assembled directly from the retrieved rows (`title` + `url&t=<start_time>s`), not via Bedrock's citations API. The model only ever sees plain text context.

## Project structure

```
scrape-script.py        — download, transcribe, chunk, embed
loader-script.py         — loads output.jsonl into Aurora
lambda_function.py       — query handler (API Gateway -> Lambda)
gbte-rag-search/          — [gbte_search] shortcode + JS + CSS
trust-policy.json         — Lambda IAM assume-role policy
lambda-permissions.json   — bedrock:InvokeModel + rds-db:connect permissions
```

## Infrastructure

- **Aurora PostgreSQL Serverless v2 (pgvector)**, `us-east-2`, 0-4 ACU auto-pause. Public, **IAM-database-auth only**. A `rds generate-db-auth-token` token authenticates the connection. No VPC or RDS Proxy (traffic isn't large enough).
- **API Gateway + Lambda** — `POST /search` takes `{"question"}`, returns `{"answer", "citations"}`. Lambda reaches Aurora over IAM auth and calls Bedrock via an attached role, no stored credentials.
- **Bedrock**: Amazon Titan Embed V2 for embeddings. Claude Sonnet for generation.
- **WordPress plugin**: PHP shortcode + vanilla JS, calls the API Gateway URL directly and renders the answer with clickable, timestamped citation links.

## Live example

A real round trip from the deployed API (`response.json` in this repo):

```json
{
  "answer": "Coach Ilja... was a total package coach connected to the GreatBase Tennis program for years...",
  "citations": [
    { "title": "The GreatBase Tennis Podcast Episode 130 - ILJA SEMJONOVS INTERVIEW",
      "url": "https://www.youtube.com/watch?v=oi9MAxzcvnk&t=3964s" },
    { "title": "The GreatBase Tennis Podcast Episode 280 - GOLDEN GROWTH RECAP",
      "url": "https://www.youtube.com/watch?v=0RHpFHbWqdk&t=171s" }
  ]
}
```

## Status

- End-to-end pipeline is built and deployed: ~24,486 chunks loaded into Aurora, query API live, WordPress plugin built and wired to it.
- **Open**: abuse-protection thresholds (API Gateway throttling is in place).
