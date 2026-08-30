import yt_dlp
import whisper
import os
import boto3
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = whisper.load_model("medium", device="cuda")
model_id = "amazon.titan-embed-text-v2:0"
bedrock = boto3.client(service_name='bedrock-runtime')
accept = "application/json"
content_type = "application/json"

file_directory = "data/"
url_file_name = "urls.md"
out_file_name = 'output.jsonl'

file_path = os.path.join(file_directory, url_file_name)
out_path = os.path.join(file_directory, out_file_name)

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
    'cookiesfrombrowser': ('firefox', os.environ["ZEN_PROFILE_PATH"]),
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
    add_start_index = True,
)

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        url_file = file.read()
else:
    print(f"error: the file at {file_path} does not exist.")

# returns the start and end times for the embedding chunk
def get_chunk_times(seg_list, chunk_char_start, chunk_char_end):
    chunk_start_t = 0.0
    chunk_end_t = 0.0

    for seg in seg_list:
        if seg['seg_char_start'] <= chunk_char_start and seg['seg_char_end'] >= chunk_char_start:
            chunk_start_t = seg['seg_start_t']
            break

    for seg in seg_list:
        if seg['seg_char_start'] <= chunk_char_end and seg['seg_char_end'] >= chunk_char_end:
            chunk_end_t = seg['seg_end_t']
            break

    return chunk_start_t, chunk_end_t

# for each video
for line in url_file.splitlines():
    if '!' in line or not line:
        continue
        
    # using yt_dlp and output path
    with yt_dlp.YoutubeDL(ydl_opts) as ydl, open(out_path, "a", encoding="utf-8") as f:
        # pulling audio from url and transcribing
        info_dict = ydl.extract_info(line, download=True)
        text_result = model.transcribe(info_dict['requested_downloads'][0]['filepath'])
        os.remove(info_dict['requested_downloads'][0]['filepath'])

        char_count = 0
        seg_list = []
        text = ""

        # create a dict for each segment return from whisper for start and end times
        # also create the full text doc to then chunk correctly
        for segment in text_result['segments']:
            seg_length = len(segment['text'])
            seg_dict = {
                'seg_char_start': char_count, 
                'seg_char_end': char_count + seg_length, 
                'seg_start_t': segment['start'], 
                'seg_end_t': segment['end']
            }
            seg_list.append(seg_dict)
            char_count += seg_length

            text += segment['text']

        # chunking the text with langchain splitter
        chunks = text_splitter.create_documents([text])
        
        # each chunk is embedded and stored with different metadata
        for chunk in chunks:
            chunk_text = chunk.page_content
            chunk_char_start = chunk.metadata['start_index']
            chunk_char_end = chunk_char_start + len(chunk_text)
            
            # chunk to json, embed with aws bedrock, save embedding
            body = json.dumps({ "inputText": chunk_text })
            response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
            response_body = json.loads(response.get('body').read())
            chunk_embedding = response_body['embedding']

            chunk_start_t, chunk_end_t = get_chunk_times(seg_list, chunk_char_start, chunk_char_end)

            database_entry = {
                "text": chunk_text,
                "embedding": chunk_embedding,
                "start": chunk_start_t,
                "end": chunk_end_t,
                "title": info_dict['title'],
                "url": info_dict['webpage_url'],
            }

            f.write(json.dumps(database_entry) + "\n")