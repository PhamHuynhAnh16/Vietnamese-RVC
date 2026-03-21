import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import (
    configs, 
    translations 
)

def training_tab():
    with gr.TabItem(translations["training_model"]):
        if configs.get("training_tab", True):
            from main.app.tabs.training.child.training import training_model_tab

            with gr.TabItem(translations["training_model"]):
                gr.Markdown(f"## {translations['training_model']}")
                training_model_tab()

        if configs.get("create_dataset_tab", True):
            from main.app.tabs.training.child.create_dataset import create_dataset_tab

            with gr.TabItem(translations["createdataset"]):
                gr.Markdown(translations["create_dataset_markdown"])
                create_dataset_tab()

        if configs.get("create_reference_tab", True):
            from main.app.tabs.training.child.create_reference import create_reference_tab

            with gr.TabItem(translations["create_reference"]):
                gr.Markdown(translations["create_reference_markdown"])
                create_reference_tab()