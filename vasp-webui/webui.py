import os
import glob
import gradio as gr
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket
from working_directory import working_directory_blocks
from calculation import calculation_tab_content
from result import result_tab_content

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# clean up the directory
for filepath in glob.iglob('./static/*.html'):
    os.remove(filepath)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# function to find an available port
def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port

available_port = find_available_port()

with gr.Blocks() as blocks:
    with gr.Row():
        working_directory_path_state, working_directory_file_list_state = working_directory_blocks()
        with gr.Column(scale=2):
            with gr.Row(min_height=40):
                status_markdown = gr.Markdown()
            with gr.Row():
                with gr.Tabs() as tabs:
                    calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)
                    result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, css_paths=Path('./styles.css'), path="/")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port, access_log=False)
