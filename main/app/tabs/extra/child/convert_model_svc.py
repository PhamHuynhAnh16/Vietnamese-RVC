import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.model_utils import svc_export

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

def convert_svc_model_tab():
    with gr.Row():
        gr.Markdown(translations["convert_model_svc_markdown_2"])
    with gr.Row():
        delete_when_success = gr.Checkbox(
            label="Xóa các tệp gốc khi hoàn thành",
            value=True,
            interactive=True
        )
    with gr.Row():
        model_pth_upload = gr.File(
            label=translations["drop_model"], 
            file_types=[".pth"]
        )
        config_upload = gr.File(
            label=translations["drop_json"], 
            file_types=[".json"]
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
            config_path = gr.Textbox(
                label=translations["config_path"],
                value="",
                placeholder="config.json",
                interactive=True
            )
        with gr.Row():
            modelname = gr.Textbox(
                label=translations["modelname"],
                value="", 
                placeholder=translations["modelname"], 
                interactive=True
            )
        with gr.Row():
            refresh_model = gr.Button(
                translations["refresh"]
            )
    with gr.Row():
        convert_svc_button = gr.Button(
            translations["convert_model_svc"], 
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
        config_upload.upload(
            fn=lambda config: shutil_move(config.name, configs["weights_path"]),
            inputs=[
                config_upload
            ], 
            outputs=[
                config_path
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
        convert_svc_button.click(
            fn=svc_export,
            inputs=[
                model_pth_path,
                config_path,
                modelname,
                delete_when_success
            ],
            outputs=[
                output_model_file
            ],
            api_name="model_svc_export"
        )
        convert_svc_button.click(
            fn=lambda: visible(True), 
            inputs=[], 
            outputs=[
                output_model_file
            ]
        )  