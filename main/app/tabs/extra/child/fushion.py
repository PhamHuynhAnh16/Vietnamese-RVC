import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.model_utils import fushion_model

from main.app.core.ui import (
    visible, 
    shutil_move
)

from main.app.variables import (
    configs, 
    translations 
)

def fushion_tab():
    with gr.Row():
        gr.Markdown(translations["fushion_markdown_2"])
    with gr.Row():
        model_name = gr.Textbox(
            label=translations["modelname"], 
            placeholder="Model.pth", 
            value="", 
            max_lines=2, 
            interactive=True
        )
    with gr.Row():
        fushion_button = gr.Button(
            translations["fushion"], 
            variant="primary", 
            scale=4
        )
    with gr.Column():
        with gr.Row():
            upload_model_1 = gr.File(
                label=f"{translations['model_name']} 1", 
                file_types=[".pth", ".onnx"]
            ) 
            upload_model_2 = gr.File(
                label=f"{translations['model_name']} 2", 
                file_types=[".pth", ".onnx"]
            )
        with gr.Row():
            model_path_1 = gr.Textbox(
                label=f"{translations['model_path']} 1", 
                value="", 
                placeholder="assets/weights/Model_1.pth"
            )
            model_path_2 = gr.Textbox(
                label=f"{translations['model_path']} 2", 
                value="", 
                placeholder="assets/weights/Model_2.pth"
            )
    with gr.Row():
        ratio = gr.Slider(
            minimum=0, 
            maximum=1, 
            label=translations["model_ratio"], 
            info=translations["model_ratio_info"], 
            value=0.5, 
            interactive=True
        )
    with gr.Row():
        output_model = gr.File(
            label=translations["output_model_path"], 
            file_types=[".pth", ".onnx"], 
            interactive=False, 
            visible=False
        )
    with gr.Row():
        upload_model_1.upload(
            fn=lambda model: shutil_move(model.name, configs["weights_path"]), 
            inputs=[
                upload_model_1
            ], 
            outputs=[
                model_path_1
            ]
        )
        upload_model_2.upload(
            fn=lambda model: shutil_move(model.name, configs["weights_path"]), 
            inputs=[
                upload_model_2
            ], 
            outputs=[
                model_path_2
            ]
        )
    with gr.Row():
        fushion_button.click(
            fn=fushion_model,
            inputs=[
                model_name, 
                model_path_1, 
                model_path_2, 
                ratio
            ],
            outputs=[
                model_name, 
                output_model
            ],
            api_name="fushion_model"
        )
        fushion_button.click(
            fn=lambda: visible(True), 
            inputs=[], 
            outputs=[
                output_model
            ]
        )  