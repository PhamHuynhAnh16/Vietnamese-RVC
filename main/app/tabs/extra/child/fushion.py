import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.model_utils import fushion_model

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

def fushion_tab():
    with gr.Row():
        gr.Markdown(translations["fushion_markdown_2"])
    with gr.Row():
        modelname = gr.Textbox(
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
            model_path_1 = gr.Dropdown(
                label=translations["model_name"] + " 1", 
                choices=model_name, 
                value=model_name[0] if len(model_name) >= 1 else "", 
                interactive=True, 
                allow_custom_value=True
            )
            model_path_2 = gr.Dropdown(
                label=translations["model_name"] + " 2", 
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
        refresh_model.click(
            fn=lambda: [change_models_choices()[0] for _ in range(2)], 
            inputs=[], 
            outputs=[
                model_path_1,
                model_path_2
            ]
        )
    with gr.Row():
        fushion_button.click(
            fn=fushion_model,
            inputs=[
                modelname, 
                model_path_1, 
                model_path_2, 
                ratio
            ],
            outputs=[
                modelname, 
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