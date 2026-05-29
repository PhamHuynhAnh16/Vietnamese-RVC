import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.tools.analyzer import analyze_audio
from main.app.core.ui import change_audios_choices, shutil_move
from main.app.variables import translations, paths_for_files, audio_params, file_types, configs

def analyzer_tab():
    with gr.Row():
        gr.Markdown(translations["analyzer_markdown_2"])
    with gr.Row():
        with gr.Accordion(
            translations["input_output"], 
            open=False
        ):
            input_audio = gr.Dropdown(
                label=translations["audio_path"], 
                value="", 
                choices=paths_for_files, 
                info=translations["provide_audio"], 
                allow_custom_value=True, 
                interactive=True
            )
            upload_audio_file = gr.Files(
                label=translations["drop_audio"], 
                file_types=file_types
            )
            with gr.Row():
                play_audio = gr.Audio(
                    interactive=False, 
                    label=translations["input_audio"],
                    **audio_params
                )
            with gr.Row():
                refresh_audio = gr.Button(
                    translations["refresh"]
                )
    with gr.Row():
        get_info_button = gr.Button(
            value=translations["read_audio"], 
            variant="primary"
        )
    with gr.Row():
        output_info = gr.Textbox(
            label=translations["output_information"],
            value="",
            max_lines=8,
            interactive=False,
        )
    with gr.Row():
        image_output = gr.Image(
            interactive=False
        )
    with gr.Row():
        get_info_button.click(
            fn=analyze_audio,
            inputs=[input_audio],
            outputs=[output_info, image_output],
        )
        input_audio.change(
            fn=lambda audio: audio if os.path.isfile(audio) else None, 
            inputs=[
                input_audio
            ], 
            outputs=[
                play_audio
            ]
        )
        refresh_audio.click(
            fn=change_audios_choices, 
            inputs=[
                input_audio
            ], 
            outputs=[
                input_audio
            ]
        )
    with gr.Row():
        upload_audio_file.upload(
            fn=lambda audio_in: [
                shutil_move(audio.name, configs["audios_path"]) 
                for audio in audio_in
            ][0], 
            inputs=[
                upload_audio_file
            ], 
            outputs=[
                input_audio
            ]
        )