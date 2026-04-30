import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import (
    configs, 
    translations 
)

def editing_tab():
    with gr.TabItem(translations["editing"]):
        if configs.get("effects_tab", True):
            from main.app.tabs.editing.child.audio_effects import audio_effects_tab

            with gr.TabItem(translations["audio_effects"]):
                gr.Markdown(translations["apply_audio_effects"])
                audio_effects_tab()
            
        if configs.get("quirk_tab", True):
            from main.app.tabs.editing.child.quirk import quirk_tab

            with gr.TabItem(translations["quirk"]):
                gr.Markdown(translations["quirk_info"])
                quirk_tab()