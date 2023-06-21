import openai
import os
import tiktoken
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv("key")


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
        engine="gpt-4-32k-0613",
        prompt=context + "\nQuestion: " + question,
        max_tokens=100,
        temperature=0.6,
        n=1,
        stop=None,
    )
    answer = response.choice[0].text.strip()
    return answer
