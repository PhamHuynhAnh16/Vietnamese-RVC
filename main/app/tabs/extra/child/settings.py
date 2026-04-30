import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.utils import stop_pid
from main.app.core.ui import change_fp, run_commands

from main.app.core.restart import restart

from main.app.variables import (
    font, 
    theme, 
    config, 
    configs, 
    language, 
    translations 
)

reload_js = """
() => {
    function Checking() {
        fetch(window.location.origin)
            .then(response => {
                if (response.ok) {
                    location.reload();
                } else {
                    setTimeout(Checking, 3000);
                }
            })
            .catch(err => {
                setTimeout(Checking, 3000);
            });
    }
    setTimeout(Checking, 10000);
}
"""

def settings_tab(app):
    with gr.Row():
        gr.Markdown(translations["settings_markdown_2"])
    with gr.Row():
        toggle_button = gr.Button(
            translations["change_light_dark"], 
            variant="secondary", 
            scale=2
        )
    with gr.Row():
        with gr.Column():
            language_dropdown = gr.Dropdown(
                label=translations["lang"], 
                interactive=True, 
                info=translations["lang_restart"], 
                choices=configs.get("support_language", "vi-VN"), 
                value=language
            )
        with gr.Column():
            theme_dropdown = gr.Dropdown(
                label=translations["theme"], 
                interactive=True, 
                info=translations["theme_restart"], 
                choices=configs.get("themes", theme), 
                value=theme, 
                allow_custom_value=True
            )
        with gr.Column():
            font_choice = gr.Textbox(
                label=translations["font"], 
                info=translations["font_info"], 
                value=font, 
                interactive=True
            )
    with gr.Row():
        restart_button = gr.Button(
            translations["restart_button"]
        )
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    bf16_checkbox = gr.Checkbox(
                        label=translations["bf16"], 
                        value=False, 
                        interactive=config.device.startswith(("cuda", "xpu")) and config.bf16_support
                    )
                    tf32_checkbox = gr.Checkbox(
                        label=translations["tf32"],
                        value=False, 
                        interactive=config.device.startswith("cuda") and config.tf32_support
                    )
                fp_choice = gr.Radio(
                    choices=["fp16","fp32"], 
                    value="fp16" if configs.get("fp16", False) else "fp32", 
                    label=translations["precision"], 
                    info=translations["precision_info"], 
                    interactive=config.allow_is_half
                )
            fp_button = gr.Button(
                translations["update_precision"], 
                variant="primary", 
                scale=2
            )
    with gr.Row():
        commands = gr.Textbox(
            label=translations["commands"],
            info=translations["commands_info"],
            value="",
            interactive=True
        )
    with gr.Row():
        run_commands_button = gr.Button(
            translations["run_commands"]
        )
    with gr.Row():
        with gr.Column():
            with gr.Accordion(translations["stop"], open=False, visible=True):
                separate_stop = gr.Button(
                    translations["stop_separate"]
                )
                convert_stop = gr.Button(
                    translations["stop_convert"]
                )
                create_dataset_stop = gr.Button(
                    translations["stop_create_dataset"]
                )
                with gr.Accordion(translations["stop_training"], open=False):
                    model_name_stop = gr.Textbox(
                        label=translations["modelname"], 
                        info=translations["training_model_name"], 
                        value="", 
                        placeholder=translations["modelname"], 
                        interactive=True
                    )
                    preprocess_stop = gr.Button(
                        translations["stop_preprocess"]
                    )
                    extract_stop = gr.Button(
                        translations["stop_extract"]
                    )
                    train_stop = gr.Button(
                        translations["stop_training"]
                    )
    with gr.Row():
        toggle_button.click(
            fn=None, 
            js="() => {document.body.classList.toggle('dark')}"
        )
        fp_button.click(
            fn=change_fp, 
            inputs=[
                fp_choice, 
                bf16_checkbox,
                tf32_checkbox
            ], 
            outputs=[
                fp_choice
            ]
        )
        run_commands_button.click(
            fn=run_commands,
            inputs=[
                commands
            ],
            outputs=[]
        )
    with gr.Row():
        restart_button.click(
            fn=lambda lang, theme, font: restart(lang, theme, font, app), 
            inputs=[language_dropdown, theme_dropdown, font_choice], 
            outputs=[]
        )
        restart_button.click(
            fn=None, 
            js=reload_js, 
            inputs=[],
            outputs=[]
        )
    with gr.Row():
        separate_stop.click(
            fn=lambda: stop_pid("separate_pid", None, False), 
            inputs=[], 
            outputs=[]
        )
        convert_stop.click(
            fn=lambda: stop_pid("convert_pid", None, False), 
            inputs=[], 
            outputs=[]
        )
        create_dataset_stop.click(
            fn=lambda: stop_pid("create_dataset_pid", None, False), 
            inputs=[], 
            outputs=[]
        )
    with gr.Row():
        preprocess_stop.click(
            fn=lambda model_name_stop: stop_pid("preprocess_pid", model_name_stop, False), 
            inputs=[
                model_name_stop
            ], 
            outputs=[]
        )
        extract_stop.click(
            fn=lambda model_name_stop: stop_pid("extract_pid", model_name_stop, False), 
            inputs=[
                model_name_stop
            ], 
            outputs=[]
        )
        train_stop.click(
            fn=lambda model_name_stop: stop_pid("train_pid", model_name_stop, True), 
            inputs=[
                model_name_stop
            ], 
            outputs=[]
        )