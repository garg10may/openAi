import pdfplumber

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

if __name__ == '__main__':
    pdf_text = extract_text_from_pdf("./MDL-785_0.pdf")
    with open('pdfplumber_extracted_text.txt', 'w') as f:
        f.write(pdf_text)

