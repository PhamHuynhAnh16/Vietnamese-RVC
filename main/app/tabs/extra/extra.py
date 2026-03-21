import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import (
    configs, 
    translations 
)

from main.app.tabs.extra.child.fushion import fushion_tab
from main.app.tabs.extra.child.settings import settings_tab
from main.app.tabs.extra.child.read_model import read_model_tab
from main.app.tabs.extra.child.f0_extract import f0_extract_tab
from main.app.tabs.extra.child.create_srt import create_srt_tab
from main.app.tabs.extra.child.convert_model_svc import convert_svc_model_tab
from main.app.tabs.extra.child.convert_model_onnx import convert_onnx_model_tab

def extra_tab(app):
    with gr.TabItem(translations["extra"]):
        if configs.get("fushion_tab", True):
            with gr.TabItem(translations["fushion"]):
                gr.Markdown(translations["fushion_markdown"])
                fushion_tab()

        if configs.get("read_tab", True):
            with gr.TabItem(translations["read_model"]):
                gr.Markdown(translations["read_model_markdown"])
                read_model_tab()

        if configs.get("onnx_tab", True):
            with gr.TabItem(translations["convert_model"]):
                gr.Markdown(translations["pytorch2onnx"])
                convert_onnx_model_tab()
        
        if configs.get("svc_tab", True):
            with gr.TabItem(translations["convert_model_svc"]):
                gr.Markdown(translations["convert_model_svc_markdown"])
                convert_svc_model_tab()

        if configs.get("f0_extractor_tab", True):
            with gr.TabItem(translations["f0_extractor_tab"]):
                gr.Markdown(translations["f0_extractor_markdown"])
                f0_extract_tab()

        if configs.get("create_srt_tab", True):
            with gr.TabItem(translations["create_srt_tab"]):
                gr.Markdown(translations["create_srt_markdown"])
                create_srt_tab()

        if configs.get("settings_tab", True):
            with gr.TabItem(translations["settings"]):
                gr.Markdown(translations["settings_markdown"])
                settings_tab(app)