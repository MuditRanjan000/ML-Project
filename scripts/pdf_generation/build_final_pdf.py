import os
import urllib.request
from fpdf import FPDF
import pypdf
import fitz

class FinalProposalPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Use standard fonts
        self.set_auto_page_break(auto=True, margin=20)
        
    def add_title_section(self):
        self.set_margins(25.4, 25.4, 25.4)
        self.add_page()
        # Title
        self.set_font("helvetica", "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, "Architecture or Training Recipe?\nA Controlled Study of Corruption Robustness\nand Calibration in CNNs and Vision Transformers", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        
        # Metadata strip
        self.set_font("helvetica", "B", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "INTRODUCTION TO MACHINE LEARNING - CSD361  |  COMPUTER VISION  |  2-STUDENT TEAM  |  8-WEEK PROJECT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(12)

    def section_title(self, title):
        self.set_font("helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, title.upper(), new_x="LMARGIN", new_y="NEXT")
        # Underline
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        
    def section_subtitle(self, title):
        self.ln(2)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, title.upper(), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("times", "", 11.5)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        
    def bullet_point(self, strong_text, text=""):
        self.set_font("helvetica", "B", 10.5)
        self.set_text_color(0, 0, 0)
        self.write(6, f"  *  {strong_text} ")
        if text:
            self.set_font("times", "", 11.5)
            self.write(6, f"{text}\n")
        else:
            self.write(6, "\n")
        
    def draw_diagram(self):
        # Draw the vector diagram natively using FPDF
        self.add_page()
        start_y = 30
        center_x = self.w / 2
        
        # Helper to draw a box
        def draw_box(x, y, w, h, text1, text2=""):
            self.set_fill_color(248, 248, 250)
            self.set_draw_color(100, 100, 100)
            self.set_line_width(0.3)
            self.rect(x, y, w, h, style="DF")
            
            self.set_font("helvetica", "B", 10)
            self.set_xy(x, y + (h/2 - 5 if text2 else h/2 - 2))
            self.cell(w, 5, text1, align="C", border=0)
            
            if text2:
                self.set_font("helvetica", "", 10)
                self.set_xy(x, y + (h/2))
                self.cell(w, 5, text2, align="C", border=0)
                
        def draw_arrow(x1, y1, x2, y2):
            self.set_draw_color(150, 150, 150)
            self.set_line_width(0.5)
            self.line(x1, y1, x2, y2)
            # Arrow head
            self.line(x2, y2, x2-2, y2-3)
            self.line(x2, y2, x2+2, y2-3)

        # 1. Models (Side by side)
        box_w = 40
        box_h = 16
        gap = 20
        
        # ResNet Box
        rx = center_x - box_w - gap/2
        ry = start_y
        draw_box(rx, ry, box_w, box_h, "ResNet-18", "CNN")
        
        # ViT Box
        vx = center_x + gap/2
        vy = start_y
        draw_box(vx, vy, box_w, box_h, "ViT-Tiny", "Transformer")
        
        # Arrows down to Recipe
        draw_arrow(rx + box_w/2, ry + box_h, center_x - 10, ry + box_h + 15)
        draw_arrow(vx + box_w/2, vy + box_h, center_x + 10, vy + box_h + 15)
        
        # 2. Recipe Box
        rec_w = 60
        rec_y = ry + box_h + 15
        draw_box(center_x - rec_w/2, rec_y, rec_w, box_h, "Training Recipe", "A / B")
        
        # Arrow down
        draw_arrow(center_x, rec_y + box_h, center_x, rec_y + box_h + 15)
        
        # 3. Training Box
        train_w = 60
        train_y = rec_y + box_h + 15
        draw_box(center_x - train_w/2, train_y, train_w, box_h, "CIFAR-100", "Training")
        
        # Arrow down
        draw_arrow(center_x, train_y + box_h, center_x, train_y + box_h + 15)
        
        # 4. Evaluation Box
        eval_w = 60
        eval_y = train_y + box_h + 15
        draw_box(center_x - eval_w/2, eval_y, eval_w, box_h, "CIFAR-100-C", "Corruption Shift")
        
        # Arrow down
        draw_arrow(center_x, eval_y + box_h, center_x, eval_y + box_h + 15)
        
        # 5. Metrics Box (wide)
        met_w = 100
        met_h = 24
        met_y = eval_y + box_h + 15
        self.set_fill_color(248, 248, 250)
        self.set_draw_color(100, 100, 100)
        self.set_line_width(0.3)
        self.rect(center_x - met_w/2, met_y, met_w, met_h, style="DF")
        
        self.set_font("helvetica", "", 10.5)
        self.set_xy(center_x - met_w/2, met_y + 2)
        self.cell(met_w, 5, "Clean Accuracy", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(center_x - met_w/2, met_y + 7)
        self.cell(met_w, 5, "Robustness", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(center_x - met_w/2, met_y + 12)
        self.cell(met_w, 5, "Calibration", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(center_x - met_w/2, met_y + 17)
        self.cell(met_w, 5, "Failure Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
        
        self.set_y(met_y + met_h + 15)


def generate_front_matter(output_pdf):
    pdf = FinalProposalPDF()
    
    # --- PAGE 1 ---
    pdf.add_title_section()
    
    pdf.section_title("Project Overview")
    pdf.body_text("Convolutional Neural Networks (CNNs) have historically dominated image recognition tasks, establishing the baseline for architectural efficiency and performance. Recently, Vision Transformers (ViTs) have emerged as a powerful alternative utilizing global self-attention, often demonstrating superior performance on large-scale datasets and exhibiting greater robustness to common image corruptions.")
    pdf.body_text("However, the exact source of this robustness advantage remains heavily debated. Apparent robustness differences may be deeply influenced by differences in optimization strategies, data augmentations, and model configurations rather than purely architectural mechanisms. When models trained under different paradigms are compared directly, attributing the robustness gains exclusively to the self-attention mechanism becomes problematic.")
    pdf.body_text("This project conducts a strictly controlled empirical study to separate the effects of architectural inductive biases from training recipes. By evaluating representative CNN and ViT models across identical, rigorously defined optimization environments, we aim to quantify the relative contributions of these factors to both corruption robustness and calibration under distribution shift.")
    
    pdf.ln(5)
    pdf.set_fill_color(250, 250, 252)
    pdf.set_draw_color(220, 220, 220)
    pdf.set_line_width(0.3)
    
    info = [
        ("TEAM", "Mudit Ranjan (2410110207)"),
        ("", "Ashank Kumar Singh (2410110082)"),
        ("PROJECT AREA", "Computer Vision"),
        ("CORE MODELS", "ResNet-18, ViT-Tiny"),
        ("DATASET", "CIFAR-100"),
        ("ROBUSTNESS BENCHMARK", "CIFAR-100-C"),
        ("CORE EXPERIMENT", "2 Architectures x 2 Training Recipes x 3 Seeds"),
        ("CORE OBJECTIVE", "Quantify architecture and training-recipe effects on corruption"),
        ("", "robustness and calibration.")
    ]
    
    # Draw info panel table
    pdf.set_x(pdf.l_margin + 5)
    start_x = pdf.get_x()
    
    for k, v in info:
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(50, 6.5, k, border=0, fill=True)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6.5, v, border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_x(start_x)
        
    # --- PAGE 2 ---
    pdf.add_page()
    pdf.section_title("Problem Statement")
    pdf.body_text("CNNs and Vision Transformers represent different architectural approaches to visual recognition. However, comparisons of their robustness can be confounded when the models are trained using different optimizers, learning-rate schedules, data augmentation, regularization, training budgets, and model configurations.")
    pdf.body_text("Therefore, differences observed under mismatched training conditions cannot be cleanly interpreted as purely architectural. Our study addresses this through a controlled empirical design in which the same training recipe is applied across the two architectural families.")
    
    pdf.section_title("Research Motivation")
    pdf.body_text("Distinguishing Architecture from Training Recipe is important for understanding robustness. If robustness is primarily a product of the training recipe, then the field can significantly improve CNN reliability by universally updating training protocols, circumventing the need for a complete architectural paradigm shift to ViTs in computationally constrained environments. Conversely, if the architectural inductive bias is proven to be a primary driver of robustness, it justifies the continued structural shift toward attention-based models.")

    pdf.section_title("Research Gap")
    pdf.body_text("Prior work has demonstrated that training frameworks materially affect CNN-Transformer robustness comparisons. Building on this evidence, our study examines the architecture x training-recipe interaction in a controlled from-scratch CIFAR-100/CIFAR-100-C setting, with repeated seeds and an accompanying calibration analysis.")

    # --- PAGE 3 ---
    pdf.add_page()
    pdf.section_title("Research Questions")
    
    pdf.section_subtitle("Primary Research Question")
    pdf.body_text('"Under controlled training conditions, how do CNNs and Vision Transformers differ in corruption robustness, and to what extent do these differences change when the same training recipe is applied across architectures?"')
    
    pdf.section_subtitle("Secondary Research Question")
    pdf.body_text('"Do architecture- and recipe-related differences in corruption robustness correspond to differences in calibration under the same distribution shifts?"')
    
    pdf.ln(5)
    pdf.section_title("Hypotheses")
    pdf.set_font("helvetica", "B", 11)
    pdf.write(6, "H1 (Robustness): ")
    pdf.set_font("times", "", 11.5)
    pdf.write(6, "We hypothesize that applying a shared modern training recipe will alter the comparative corruption robustness of the two architectures.\n\n")
    
    pdf.set_font("helvetica", "B", 11)
    pdf.write(6, "H2 (Calibration): ")
    pdf.set_font("times", "", 11.5)
    pdf.write(6, "We hypothesize that robustness and calibration may not change identically under corruption.\n\n")
    
    pdf.ln(5)
    pdf.section_title("Experimental Design")
    pdf.draw_diagram()
    
    # --- PAGE 4 ---
    pdf.add_page()
    pdf.section_title("Experimental Methodology")
    pdf.bullet_point("Architectures:", "ResNet-18 and CIFAR-adapted ViT-Tiny")
    pdf.bullet_point("Training Recipes:", "Recipe A (Conventional supervised) and Recipe B (Modern regularized)")
    pdf.bullet_point("Seeds:", "42, 43, 44 (12 Core runs)")
    pdf.bullet_point("Dataset:", "CIFAR-100")
    pdf.bullet_point("Corruption Benchmark:", "CIFAR-100-C")
    pdf.ln(2)

    pdf.section_title("Fairness Protocol")
    pdf.bullet_point("Fixed architectures and fixed training budget")
    pdf.bullet_point("Same recipe applied across architectures")
    pdf.bullet_point("Natural clean performance", "(No capacity adjustment, no artificial early stopping)")
    pdf.bullet_point("No test-set tuning")
    pdf.bullet_point("Temperature scaling", "only on the held-out clean validation set")
    pdf.ln(2)
    
    pdf.section_title("Evaluation")
    pdf.bullet_point("Clean Accuracy and Standard mCE")
    pdf.bullet_point("Absolute Error Increase")
    pdf.bullet_point("15-bin Expected Calibration Error (ECE)")
    pdf.bullet_point("Accuracy by corruption type and severity")
    pdf.bullet_point("Factorial analysis", "reporting effect sizes and uncertainty")
    pdf.ln(2)

    pdf.section_title("Expected Contribution")
    pdf.body_text("We provide a controlled empirical comparison that quantifies the effects of architecture, training recipe, and their interaction on corruption robustness and calibration.")

    pdf.ln(4)
    pdf.section_title("Base Paper")
    pdf.set_font("helvetica", "B", 11.5)
    pdf.cell(0, 6, "Are Transformers More Robust Than CNNs?", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("times", "", 11.5)
    pdf.cell(0, 6, "Yutong Bai, Jieru Mei, Alan L. Yuille, Cihang Xie", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "NeurIPS 2021", new_x="LMARGIN", new_y="NEXT")
    
    pdf.section_subtitle("Why This Is Our Base Paper")
    pdf.body_text("The selected paper directly motivates this project by questioning whether reported CNN-Transformer robustness differences are confounded by mismatched model scales and training frameworks. Our study builds on this fairness question in a controlled from-scratch CIFAR-100/CIFAR-100-C setting, explicitly crossing architecture and training recipe across repeated seeds and extending the evaluation to calibration under corruption.")
    
    pdf.output(output_pdf)


def merge_and_qc(part_a_path, base_paper_path, final_pdf_path):
    print("Merging PDFs...")
    merger = pypdf.PdfWriter()
    merger.append(part_a_path)
    merger.append(base_paper_path)
    merger.write(final_pdf_path)
    merger.close()
    
    print("\n--- RUNNING PDF QUALITY CONTROL ---")
    doc = fitz.open(final_pdf_path)
    num_pages = len(doc)
    file_size_mb = os.path.getsize(final_pdf_path) / (1024 * 1024)
    
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Total Pages: {num_pages}")
    
    blank_pages = 0
    local_paths = 0
    
    # Extract images of proposal pages for manual visual QC simulation
    for i in range(min(num_pages, 5)):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        pix.save(f"qc_page_{i+1}.png")
        
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
    
    qc_pass = (blank_pages == 0 and local_paths == 0 and file_size_mb < 10 and num_pages in [18, 19])
    print(f"Overall QC: {'PASS' if qc_pass else 'FAIL'}")


if __name__ == "__main__":
    paper_url = "https://proceedings.neurips.cc/paper_files/paper/2021/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf"
    
    if not os.path.exists("base_paper.pdf"):
        print("Downloading base paper...")
        req = urllib.request.Request(paper_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open("base_paper.pdf", 'wb') as out_file:
            out_file.write(response.read())
            
    generate_front_matter("part_a_final.pdf")
    merge_and_qc("part_a_final.pdf", "base_paper.pdf", "google_form_submission.pdf")
    print("Done.")
