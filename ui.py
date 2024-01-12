
import gradio as gr
from base import *
from pathlib import Path
def gradio_rag_retrieval(data_path, question, k=3):
    return rag_retrieval(Path(data_path), question, int(k))

ui = gr.Interface(
    fn = gradio_rag_retrieval,
    inputs = ["text", "text", "number"],
    outputs = ["text", "text"],
)
ui.launch()

