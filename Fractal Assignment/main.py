from pdfParser import extract_text_from_pdf
from generateAnswers import generate_question_answer, process_text_with_api, generate_section_summary

pdf_text = extract_text_from_pdf("./MDL-785_0.pdf")
section_summaries  = process_text_with_api(pdf_text)
print(section_summaries)

print('*-' * 100)
final_summary = generate_section_summary(section_summaries)
print(final_summary)


# context = "The extracted text from PDF: " + pdf_text
# print("Context :", context)
# question = "What is the main topic discussed in the PDF?"
# answer = generate_question_answer(context, question)

# print("Question:", question)
# print("Answer:", answer)