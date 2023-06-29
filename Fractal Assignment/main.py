from pdfParser import extract_text_from_pdf
from generateAnswers import generate_question_answer, process_text_with_api, generate_section_summary
from cache import conn, get_all_summaries
from openAi import load_open_ai_key
# from pymemcache.client import base
# client = base.Client('localhost', 11211)
# import memcache

# client = base.Client(('localhost', 11211))

# section_summaries = client.get('section_summaries')

text = get_all_summaries()
# if not section_summaries:

if not text:
  print('Not existing summaries found, generating new')
  pdf_text = extract_text_from_pdf("./MDL-785_0.pdf")
  section_summaries  = process_text_with_api(pdf_text)
# print(section_summaries)
# client.set('section_summaries', section_summaries)

# print('*-' * 100)
# final_summary = generate_section_summary(section_summaries)
# print(final_summary)


# text = """
  # I have taken a pdf and generated summaries for different different chunks\
  # since you can't understand the whole pdf in one go. Now I want the final summary.\
  # The section summaries are delimited by triple brackets
  # ```{section_summaries}```
  # """
# 
# response = openai.Completion.create(
        # engine="text-davinci-003",
        # prompt=text,
        # max_tokens=300,
        # temperature=0.2,
        # n=1,
        # stop=None,
    # )
# 
# final_summary = response.choices[0].text.strip()
# print(final_summary)


context = "The extracted text from PDF: " + text
# print("Context :", context)
question = "What is the main topic discussed in the PDF?"
answer = generate_question_answer(context, question)

print("Question:", question)
print(answer)