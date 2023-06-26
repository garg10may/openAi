import openai
import os
import tiktoken
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv("key")

MAX_TOKENS = 2048 #max tokens per api call, change this for different models some have 4096 also

def split_text_intelligently(text):
    # Split the text into chunks intelligently based on punctuation marks
    chunks = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    return chunks

def process_text_with_api(text):
    # Split the text into intelligently-sized chunks
    chunks = split_text_intelligently(text)

    processed_chunks = []
    processed_tokens = 0

    for chunk in chunks:
        # Check if adding the current chunk exceeds the token limit
        if processed_tokens + len(chunk.split()) <= MAX_TOKENS:
            processed_chunks.append(chunk)
            processed_tokens += len(chunk.split())
        else:
            # Process the collected chunks as a chunk
            processed_chunk = ' '.join(processed_chunks)

            # Process the chunk with the OpenAI API
            response = openai.Completion.create(
                engine='davinci-codex',  # Choose the appropriate engine
                prompt=processed_chunk,
                max_tokens=MAX_TOKENS,
                stop=None  # Adjust the stop condition based on your requirements
            )

            # Get the processed chunk from the API response
            processed_result = response.choices[0].text.strip()

            # Clear the processed chunks and tokens
            processed_chunks = []
            processed_tokens = 0

            # Append the processed chunk
            processed_chunks.append(processed_result)

    # Process the remaining collected chunks as a final chunk
    final_chunk = ' '.join(processed_chunks)

    # Process the final chunk with the OpenAI API
    response = openai.Completion.create(
        engine='davinci-codex',  # Choose the appropriate engine
        prompt=final_chunk,
        max_tokens=MAX_TOKENS,
        stop=None  # Adjust the stop condition based on your requirements
    )

    # Get the processed final chunk from the API response
    processed_final_chunk = response.choices[0].text.strip()

    return processed_final_chunk


def generate_section_summary(section_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=section_text,
        max_token=100,
        temperature=0.2,
        n=1,
        stop=None,
    )
    summary = response.choices[0].text.strip()


def generate_recursive_summaries(sections):
    if len(sections) == 0:
        return []
    section = sections[0]
    remaining_sections = sections[1:]

    section_summary = generate_section_summary(section)
    remaining_summaries = generate_recursive_summaries(remaining_sections)

    return [section_summary] + remaining_summaries


def generate_question_answer(context, question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=context + "\nQuestion: " + question,
        max_tokens=100,
        temperature=0.6,
        n=1,
        stop=None,
    )
    answer = response.choice[0].text.strip()
    return answer
