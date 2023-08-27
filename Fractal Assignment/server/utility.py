from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas 

def convert_text_to_pdf(text):

    output_pdf_file = 'output.pdf'
    c = canvas.Canvas(output_pdf_file)
    # c.setFont('Helvetica', 12)
    y = 700
    line_height = 14

    for line in text.split('\n'):
        c.drawString(50,y,line)
        y -= line_height

    c.save()