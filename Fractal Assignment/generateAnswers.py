import openai
import os
import tiktoken
import re
from dotenv import find_dotenv, load_dotenv
from pdfParser import extract_text_from_pdf

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv("key")

MAX_TOKENS = 3500 #max tokens per api call, change this for different models some have lower or higher limit

def split_text_intelligently(text):
    # Split the text into chunks intelligently based on punctuation marks
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    # print (chunks)
    # print( 'Length of sentences chunks: ', len(chunks))
    return sentences

def process_text_with_api(text):
    # Split the text into intelligently-sized chunks
    sentences = split_text_intelligently(text)

    paragraph = []
    processed_tokens = 0
    summaries = []

    for sentence in sentences:
        # Check if adding the current chunk exceeds the token limit
        # print(chunk)
        # print('Length of chunk splitted:', len(chunk.split()))
        # if processed_tokens + len(chunk) <= MAX_TOKENS *3: #token are generally comprised of 3-4 chars
        if processed_tokens + len(sentence) <= MAX_TOKENS: #token are generally comprised of 3-4 chars
            paragraph.append(sentence)
            processed_tokens += len(sentence)
        else:
            # paragraph = ' '.join(processed_chunks)

            response = openai.Completion.create(
                engine='text-davinci-003', 
                prompt=''.join(paragraph),
                max_tokens=300,
                stop=None  # Adjust the stop condition based on your requirements
            )

            # print(response)
            summary = response.choices[0].text.strip()
            print(summary)
            print('-' * 90)

            processed_tokens = 0
            paragraph = []

            summaries.append(summary)

    # final_chunk = ' '.join(processed_chunks)

    print(summaries)

    response = openai.Completion.create(
        engine='text-davinci-003',  # Choose the appropriate engine
        prompt=''.join(summaries),
        max_tokens=300,
        stop=None  # Adjust the stop condition based on your requirements
    )

    total_summary = response.choices[0].text.strip()

    return total_summary


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

    return summary


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
