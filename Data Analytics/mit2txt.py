import os
import PyPDF2
from tqdm import tqdm

# Path to the directory containing the PDFs
root_dir = "../Desktop/CS@MIT"
output_file = "combined_text.txt"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in tqdm(range(num_pages), desc=f"Extracting from {os.path.basename(pdf_path)}"):
                text += reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
    return text

def extract_text_recursive(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8", errors="replace") as output:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(foldername, filename)
                    print(f"Extracting text from: {pdf_path}")
                    text = extract_text_from_pdf(pdf_path)
                    output.write(text + "\n")

if __name__ == "__main__":
    extract_text_recursive(root_dir, output_file)
    print(f"Text extraction completed. Output saved to {output_file}")
