import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.model_utils import onnx_export

from main.app.core.ui import (
    visible, 
    shutil_move,
    change_models_choices
)

from main.app.variables import (
    configs, 
    model_name, 
    translations 
)

def convert_model_tab():
    with gr.Row():
        gr.Markdown(translations["pytorch2onnx_markdown"])
    with gr.Row():
        model_pth_upload = gr.File(
            label=translations["drop_model"], 
            file_types=[".pth"]
        )
    with gr.Accordion(
        label=translations["model_name"], 
        open=True
    ):
        with gr.Row():
            model_pth_path = gr.Dropdown(
                label=translations["model_name"], 
                choices=model_name, 
                value=model_name[0] if len(model_name) >= 1 else "", 
                interactive=True, 
                allow_custom_value=True
            )
        with gr.Row():
            refresh_model = gr.Button(
                translations["refresh"]
            )
    with gr.Row():
        convert_onnx_button = gr.Button(
            translations["convert_model"], 
            variant="primary", 
            scale=2
        )
    with gr.Row():
        output_model_file = gr.File(
            label=translations["output_model_path"], 
            file_types=[".pth", ".onnx"], 
            interactive=False, 
            visible=False
        )
    with gr.Row():
        model_pth_upload.upload(
            fn=lambda model_pth_upload: shutil_move(model_pth_upload.name, configs["weights_path"]), 
            inputs=[
                model_pth_upload
            ], 
            outputs=[
                model_pth_path
            ]
        )
        refresh_model.click(
            fn=lambda: change_models_choices()[0], 
            inputs=[], 
            outputs=[
                model_pth_path
            ]
        )
    with gr.Row():
        convert_onnx_button.click(
            fn=onnx_export,
            inputs=[
                model_pth_path
            ],
            outputs=[
                output_model_file
            ],
            api_name="model_onnx_export"
        )
        convert_onnx_button.click(
            fn=lambda: visible(True), 
            inputs=[], 
            outputs=[
                output_model_file
            ]
        )  