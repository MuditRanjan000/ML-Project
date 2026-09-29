import urllib.request
import os
import subprocess
import markdown

def create_html(md_files, output_html):
    css = """
    <style>
        body { font-family: 'Times New Roman', Times, serif; font-size: 12pt; line-height: 1.5; margin: 40px auto; max-width: 800px; padding: 0 20px; }
        h1, h2, h3 { font-family: Arial, sans-serif; }
        h1 { font-size: 18pt; text-align: center; }
        h2 { font-size: 14pt; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        h3 { font-size: 12pt; }
        p { margin-bottom: 16px; }
        ul { margin-bottom: 16px; }
        .page-break { page-break-before: always; }
    </style>
    """
    html_content = f"<html><head>{css}</head><body>\n"
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        html_content += markdown.markdown(md_text)
        html_content += "\n<div class='page-break'></div>\n"
        
    html_content += "</body></html>"
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

def print_to_pdf(input_html, output_pdf):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    # convert absolute path to file:// URL
    abs_html = os.path.abspath(input_html)
    abs_out = os.path.abspath(output_pdf)
    file_url = f"file:///{abs_html.replace(chr(92), '/')}"
    
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={abs_out}",
        "--no-margins",
        file_url
    ]
    print(f"Running Chrome to generate PDF...")
    subprocess.run(cmd, check=True)

def download_paper(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
    else:
        print("Base paper already exists.")

def merge_pdfs(pdf_list, output_pdf):
    from pypdf import PdfWriter
    merger = PdfWriter()
    for pdf in pdf_list:
        merger.append(pdf)
    merger.write(output_pdf)
    merger.close()
    print(f"Merged PDF saved to {output_pdf}")

def main():
    md_files = ["problem_statement.md", "base_paper_notes.md"]
    html_file = "temp_proposal.html"
    part_a_pdf = "part_a.pdf"
    base_paper_pdf = "base_paper.pdf"
    final_pdf = "google_form_submission.pdf"
    
    print("Creating HTML...")
    create_html(md_files, html_file)
    print("Converting HTML to PDF...")
    print_to_pdf(html_file, part_a_pdf)
    
    paper_url = "https://proceedings.neurips.cc/paper_files/paper/2021/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf"
    download_paper(paper_url, base_paper_pdf)
    
    print("Merging PDFs...")
    merge_pdfs([part_a_pdf, base_paper_pdf], final_pdf)

if __name__ == "__main__":
    main()
