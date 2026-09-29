from fpdf import FPDF
import urllib.request
import os
import pypdf
import fitz

class ProposalPDF(FPDF):
    pass

def generate_part_a():
    print("Generating part_a.pdf using fpdf2...")
    pdf = ProposalPDF()
    pdf.add_page()
    pdf.set_margins(25.4, 25.4, 25.4)
    pdf.set_auto_page_break(auto=True, margin=25.4)
    
    # Title
    pdf.set_font("helvetica", "B", 18)
    pdf.multi_cell(0, 8, "Architecture or Training Recipe?\nA Controlled Study of Corruption Robustness\nand Calibration in CNNs and Vision Transformers", align="C")
    pdf.ln(15)
    
    # Helpers
    def add_h2(text):
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(5)

    def add_h3(text):
        pdf.ln(3)
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, text.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def add_p(text):
        pdf.set_font("times", "", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # PAGE 1
    add_h2("Project Overview")
    add_p("Convolutional Neural Networks (CNNs) have historically dominated image recognition tasks. Recently, Vision Transformers (ViTs) have emerged as a powerful alternative, often demonstrating superior performance on large-scale datasets and exhibiting greater robustness to common corruptions. However, the exact source of this robustness advantage remains heavily debated. ViTs are typically trained with vastly different, highly regularized modern training recipes compared to the conventional supervised recipes historically used for CNNs.")
    add_p("Apparent robustness differences may be deeply influenced by differences in optimization strategies, augmentations, and model configurations rather than purely architectural mechanisms. When models trained under different paradigms are compared directly, attributing the robustness gains exclusively to the self-attention mechanism becomes problematic.")
    add_p("This project conducts a strictly controlled empirical study to separate the effects of architectural inductive biases from training recipes. By evaluating representative CNN and ViT models across identical, rigorously defined optimization environments, we aim to quantify the relative contributions of these factors to both corruption robustness and calibration under distribution shift.")
    
    # Info Panel
    pdf.ln(5)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(50, 50, 50)
    pdf.set_font("helvetica", "", 10)
    
    panel_info = [
        ("Team Members:", "Mudit Ranjan (2410110207) and Ashank Kumar Singh (2410110082)"),
        ("Project Area:", "Computer Vision"),
        ("Core Models:", "ResNet-18 & ViT-Tiny"),
        ("Dataset:", "CIFAR-100"),
        ("Robustness Benchmark:", "CIFAR-100-C"),
        ("Core Design:", "2 Architectures x 2 Training Recipes x 3 Seeds"),
        ("Core Objective:", "Separate architecture and training-recipe effects on robustness.")
    ]
    for label, val in panel_info:
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(50, 8, label, border=0, fill=True)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 8, val, border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
        
    pdf.add_page()
    # PAGE 2
    add_h2("Problem Statement")
    add_p("CNNs have historically dominated image recognition, establishing the baseline for architectural efficiency and performance. Vision Transformers provide a compelling alternative architecture utilizing global self-attention. Robustness comparisons between them can be heavily confounded by their divergent training recipes. Specifically, a direct comparison between differently trained models cannot cleanly attribute robustness differences to the architecture itself, as modern regularization suites independently provide massive robustness gains.")
    add_p("This motivates a controlled empirical study: without locking the training framework and computational budget across both architectural families, observed performance deltas remain causally entangled.")

    add_h2("Research Motivation")
    add_p("Distinguishing between the effects of Architecture versus Training Recipe matters fundamentally for the future of computer vision design. If robustness is primarily a product of the training recipe, then the field can significantly improve CNN reliability by universally updating training protocols, circumventing the need for a complete architectural paradigm shift to ViTs in computationally constrained environments. Conversely, if the architectural inductive bias is proven to be the primary driver of robustness, it strongly justifies the continued structural shift toward attention-based models in safety-critical applications.")

    add_h2("Research Gap")
    add_p("Existing work has demonstrated the profound impact of modern training frameworks on model robustness (e.g., Bai et al., 2021), while our study focuses on explicitly quantifying these effects under a tightly controlled CIFAR-100/CIFAR-100-C factorial design. The literature surrounding entry-scale models often relies on uncontrolled, disparate pretraining pipelines. Our project bridges this gap by crossing standard CNN and ViT architectures with distinctly different training recipes in a from-scratch training environment, allowing us to compute precise empirical attributions across multiple independent seeds.")

    pdf.add_page()
    # PAGE 3
    add_h2("Research Questions & Hypotheses")
    add_h3("Primary Research Question")
    add_p('"Under controlled training conditions, how do CNNs and Vision Transformers differ in corruption robustness, and to what extent do these differences change when the same training recipe is applied across architectures?"')
    
    add_h3("Secondary Research Question")
    add_p('"Do architecture- and recipe-related differences in corruption robustness correspond to differences in calibration under the same distribution shifts?"')
    
    add_h3("Hypotheses")
    pdf.set_font("times", "B", 11)
    pdf.write(6, "H1 (Robustness): ")
    pdf.set_font("times", "", 11)
    pdf.write(6, "We hypothesize that applying a shared modern training recipe will significantly reduce the observed corruption robustness difference between the two architectures, demonstrating that training protocol explains a substantial portion of the robustness gap.\n\n")
    
    pdf.set_font("times", "B", 11)
    pdf.write(6, "H2 (Calibration): ")
    pdf.set_font("times", "", 11)
    pdf.write(6, "We hypothesize that robustness and calibration may not change in exactly the same way under corruption. The architectural effect on Expected Calibration Error (ECE) at high corruption severity will remain distinct even after recipe matching.\n\n")
    
    add_h2("Experimental Design")
    pdf.set_font("helvetica", "", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, "Architecture (ResNet-18 / ViT-Tiny)   x   Training Recipe (Recipe A / Recipe B)   x   Seed (42, 43, 44)", align="C", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "V", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, "CIFAR-100 Training", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "V", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, "CIFAR-100-C Evaluation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "V", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 10, "Clean Accuracy  |  Robustness  |  Calibration  |  Failure Analysis", align="C", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(10)
    
    def add_li(strong_text, text):
        pdf.set_font("helvetica", "B", 11)
        pdf.write(6, f"* {strong_text} ")
        pdf.set_font("helvetica", "", 11)
        pdf.write(6, f"{text}\n")
    
    add_li("Architectures:", "ResNet-18 (Standard CNN) and CIFAR-adapted ViT-Tiny (Standard ViT).")
    add_li("Recipes:", "Recipe A (Conventional supervised training) and Recipe B (Modern regularized).")
    add_li("Seeds:", "42, 43, and 44. Core runs: 12.")

    pdf.add_page()
    # PAGE 4
    add_h2("Evaluation & Scope")
    add_h3("Evaluation Metrics")
    add_li("Clean Accuracy:", "Standard performance on uncorrupted test data.")
    add_li("mCE:", "Mean Corruption Error across all severities and corruption types.")
    add_li("Absolute Error Increase:", "Difference between Corrupted Error and Clean Error, isolating degradation.")
    add_li("ECE:", "Expected Calibration Error computed using 15 uniform bins.")
    add_li("Granular Accuracy:", "Accuracy decomposed by specific corruption type and by severity level.")
    add_li("Statistical Analysis:", "Factorial analysis reporting means, standard deviations, effect sizes, and uncertainty.")
    
    add_h3("Scope / Feasibility")
    add_li("Timeline:", "8 weeks dedicated execution timeline.")
    add_li("Team:", "Mudit Ranjan (2410110207) and Ashank Kumar Singh (2410110082).")
    add_li("Compute:", "Single NVIDIA A100 GPU when available, with a viable fallback to more limited compute (T4).")
    add_li("Dependencies:", "No multi-GPU dependency required for the entry-scale architectures selected.")
    
    pdf.ln(15)
    pdf.add_page()
    # BASE PAPER SEPARATOR
    pdf.set_y(pdf.get_y() + 80)  # Push it down vertically for better presentation
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "PART II", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "BASE PAPER", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("helvetica", "I", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Are Transformers More Robust Than CNNs?", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, "Bai et al. - NeurIPS 2021", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(15)
    add_h3("Why This Is Our Base Paper")
    def add_num_li(num, text):
        pdf.set_font("times", "", 11)
        pdf.multi_cell(0, 6, f"{num}. {text}", new_x="LMARGIN", new_y="NEXT")
        
    add_num_li(1, "The base paper challenges unfair CNN-vs-Transformer robustness comparisons in the literature.")
    add_num_li(2, "It studies the impact of training frameworks and model architecture on out-of-distribution performance.")
    add_num_li(3, "Our project extends that research question into a tightly controlled CIFAR-100/CIFAR-100-C experimental setting.")
    add_num_li(4, "We explicitly study the architecture x recipe interaction using multiple fixed seeds.")
    add_num_li(5, "We additionally examine model calibration (ECE) under corruption, expanding reliability analysis.")
    
    pdf.output("part_a.pdf")
    print("part_a.pdf generated successfully.")

def download_paper(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading base paper from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
    else:
        print("Base paper already exists locally.")

def merge_pdfs(part_a, part_b, final_output):
    print("Merging PDFs...")
    merger = pypdf.PdfWriter()
    merger.append(part_a)
    merger.append(part_b)
    merger.write(final_output)
    merger.close()
    print(f"Final merged PDF saved to {final_output}")

def run_qc(pdf_path):
    print("\n--- RUNNING PDF QUALITY CONTROL ---")
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Total Pages: {num_pages}")
    
    blank_pages = 0
    local_paths = 0
    
    for i, page in enumerate(doc):
        text = page.get_text()
        
        if len(text.strip()) < 10:
            print(f"WARNING: Page {i+1} appears to be blank.")
            blank_pages += 1
            
        if "file:///" in text or "C:\\Users\\" in text:
            print(f"WARNING: Local path footprint found on Page {i+1}.")
            local_paths += 1
            
    print(f"QC RESULTS:")
    print(f"Blank Pages: {'NONE' if blank_pages == 0 else f'FOUND ({blank_pages})'}")
    print(f"Local Paths: {'NONE' if local_paths == 0 else f'FOUND ({local_paths})'}")
    print(f"Under 10 MB: {'YES' if file_size_mb < 10 else 'NO'}")
    
    qc_pass = (blank_pages == 0 and local_paths == 0 and file_size_mb < 10 and num_pages > 10)
    print(f"Overall QC: {'PASS' if qc_pass else 'FAIL'}")

if __name__ == "__main__":
    paper_url = "https://proceedings.neurips.cc/paper_files/paper/2021/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf"
    generate_part_a()
    download_paper(paper_url, "base_paper.pdf")
    merge_pdfs("part_a.pdf", "base_paper.pdf", "google_form_submission.pdf")
    run_qc("google_form_submission.pdf")
