import gradio as gr
from rag_pipeline import RAGPipeline
from pdf_processor import process_pdf
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

rag = RAGPipeline()

def generate_pdf(content, title):
    """Generate a PDF file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        doc = SimpleDocTemplate(tmp.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(title, styles['Heading1']),
            Spacer(1, 12),
            Paragraph(content, styles['Normal'])
        ]
        doc.build(story)
        return tmp.name

def process_and_summarize(pdf_file, state):
    if pdf_file is None:
        return "Please upload a PDF.", None, state
    
    try:
        os.makedirs("data", exist_ok=True)
        
        if isinstance(pdf_file, tuple):
            pdf_bytes = pdf_file[0]
            pdf_filename = pdf_file[1] if len(pdf_file) > 1 and isinstance(pdf_file[1], str) else "uploaded.pdf"
        else:
            pdf_bytes = pdf_file
            pdf_filename = "uploaded.pdf"
        
        pdf_path = os.path.join("data", pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        
        process_pdf(pdf_path, rag)
        summary = rag.query("Provide a concise summary of the entire document.")
        
        state["summary"] = summary
        state["history"] = []
        
        pdf_path_out = generate_pdf(summary, "Document Summary")
        return summary, pdf_path_out, state
    except Exception as e:
        return f"Error processing PDF: {str(e)}", None, state

def ask_question(query, state):
    if not state.get("summary"):
        return "Please upload a PDF first.", None, state
    
    try:
        response = rag.query(query)
        state["history"].append({"question": query, "answer": response})
        
        history_text = ""
        for idx, entry in enumerate(state["history"]):
            history_text += f"**Q{idx+1}:** {entry['question']}\n**A{idx+1}:** {entry['answer']}\n---\n"
        
        pdf_path_out = generate_pdf(response, f"Question: {query}")
        return history_text, pdf_path_out, state
    except Exception as e:
        return f"Error: {str(e)}", None, state

with gr.Blocks(title="PDFSense – AI-Powered PDF Analysis") as demo:
    gr.Markdown("# PDFSense – Makes sense of your PDFs with AI")
    gr.Markdown("Upload a PDF to get a summary and ask questions about its content!")
    
    state = gr.State({"summary": None, "history": []})
    
    with gr.Row():
        pdf_input = gr.File(label="Upload a PDF", type="binary")
        summary_output = gr.Textbox(label="Document Summary", lines=5)
    
    with gr.Row():
        summary_download = gr.File(label="Download Summary")
    
    with gr.Row():
        query_input = gr.Textbox(label="Ask a question about the PDF:")
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        history_output = gr.Textbox(label="Conversation History", lines=10)
        qa_download = gr.File(label="Download Latest Q&A")
    
    pdf_input.change(
        fn=process_and_summarize,
        inputs=[pdf_input, state],
        outputs=[summary_output, summary_download, state]
    )
    submit_btn.click(
        fn=ask_question,
        inputs=[query_input, state],
        outputs=[history_output, qa_download, state]
    )

demo.launch()