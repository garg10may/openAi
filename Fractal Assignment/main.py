from pdfParser import extract_text_from_pdf
from generateAnswers import generate_question_answer
from generateAnswers import process_text_with_api

pdf_text = extract_text_from_pdf("./MDL-785_0.pdf")
summary  = process_text_with_api(pdf_text)
print('Length of summary:', len(summary))
print('\n')
print(summary)



context = "The extracted text from PDF: " + pdf_text
# print("Context :", context)
question = "What is the main topic discussed in the PDF?"
# answer = generate_question_answer(context, question)

print("Question:", question)
print("Answer:", answer)