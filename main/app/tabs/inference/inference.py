import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import (
    configs, 
    translations 
)

def inference_tab():
    with gr.TabItem(translations["inference"]):
        if configs.get("separator_tab", True):
            from main.app.tabs.inference.child.separate import separate_tab

            with gr.TabItem(translations["separator_tab"]):
                gr.Markdown(f"## {translations['separator_tab']}")
                separate_tab()

        if configs.get("convert_tab", True):
            from main.app.tabs.inference.child.convert import convert_tab

            with gr.TabItem(translations["convert_audio"]):
                gr.Markdown(f"## {translations['convert_audio']}")
                convert_tab()

        if configs.get("convert_with_vad_tab", True):
            from main.app.tabs.inference.child.convert_with_vad import convert_with_vad_tab

            with gr.TabItem(translations["convert_with_vad"]):
                gr.Markdown(f"## {translations['convert_with_vad']}")
                convert_with_vad_tab()

        if configs.get("tts_tab", True):
            from main.app.tabs.inference.child.convert_tts import convert_tts_tab

            with gr.TabItem(translations["convert_text"]):
                gr.Markdown(translations["convert_text_markdown"])
                convert_tts_tab()
