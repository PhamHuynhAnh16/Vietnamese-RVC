import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.f0_extract import f0_extract

from main.app.core.ui import (
    visible,
    unlock_f0, 
    shutil_move,
    change_audios_choices 
)

from main.app.variables import (
    configs, 
    method_f0, 
    file_types, 
    audio_params, 
    translations, 
    paths_for_files 
)

def f0_extract_tab():
    with gr.Row():
        gr.Markdown(translations["f0_extractor_markdown_2"])
    with gr.Row():
        extractor_button = gr.Button(
            translations["extract_button"].replace("2. ", ""), 
            variant="primary"
        )
    with gr.Row():
        with gr.Column():
            upload_audio_file = gr.Files(
                label=translations["drop_audio"], 
                file_types=file_types
            )
            audioplay = gr.Audio(
                interactive=False, 
                label=translations["input_audio"],
                **audio_params
            )
        with gr.Column():
            with gr.Accordion(
                translations["f0_method"], 
                open=False
            ):
                with gr.Group():
                    with gr.Row():
                        predictor_onnx = gr.Checkbox(
                            label=translations["predictor_onnx"], 
                            value=False, 
                            interactive=True
                        )
                        unlock_full_method = gr.Checkbox(
                            label=translations["f0_unlock"], 
                            value=False, 
                            interactive=True
                        )
                        autotune = gr.Checkbox(
                            label=translations["autotune"], 
                            value=False, 
                            interactive=True
                        )
                        proposal_pitch = gr.Checkbox(
                            label=translations["proposal_pitch"], 
                            value=False, 
                            interactive=True
                        )
                    f0_method_extract = gr.Radio(
                        label=translations["f0_method"], 
                        info=translations["f0_method_info"], 
                        choices=[m for m in method_f0 if m != "hybrid"], 
                        value="rmvpe", 
                        interactive=True
                    )
                    pitch = gr.Slider(
                        minimum=-24, 
                        maximum=24, 
                        step=1, 
                        info=translations["pitch_info"], 
                        label=translations["pitch"], 
                        value=0, 
                        interactive=True
                    )
                    f0_autotune_strength = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        label=translations["autotune_rate"], 
                        info=translations["autotune_rate_info"], 
                        value=1, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                    proposal_pitch_threshold = gr.Slider(
                        minimum=50.0, 
                        maximum=1200.0, 
                        label=translations["proposal_pitch_threshold"], 
                        info=translations["proposal_pitch_threshold_info"], 
                        value=255.0, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                    filter_radius = gr.Slider(
                        minimum=0, 
                        maximum=7, 
                        label=translations["filter_radius"], 
                        info=translations["filter_radius_info"], 
                        value=3, 
                        step=1, 
                        interactive=True
                    )
            with gr.Accordion(
                translations["audio_path"], 
                open=True
            ):
                input_audio_path = gr.Dropdown(
                    label=translations["audio_path"], 
                    value="", 
                    choices=paths_for_files, 
                    allow_custom_value=True, 
                    interactive=True
                )
                refresh_audio_button = gr.Button(
                    translations["refresh"]
                )
    with gr.Row():
        gr.Markdown("___")
    with gr.Row():
        file_output = gr.File(
            label="", 
            file_types=[".txt"], 
            interactive=False
        )
        image_output = gr.Image(
            label="", 
            interactive=False
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
                input_audio_path
            ]
        )
        input_audio_path.change(
            fn=lambda audio: audio if os.path.isfile(audio) else None, 
            inputs=[
                input_audio_path
            ], 
            outputs=[
                audioplay
            ]
        )
        refresh_audio_button.click(
            fn=change_audios_choices, 
            inputs=[
                input_audio_path
            ], 
            outputs=[
                input_audio_path
            ]
        )
    with gr.Row():
        autotune.change(
            fn=visible, 
            inputs=[
                autotune
            ], 
            outputs=[
                f0_autotune_strength
            ]
        )
        proposal_pitch.change(
            fn=visible, 
            inputs=[
                proposal_pitch
            ], 
            outputs=[
                proposal_pitch_threshold
            ]
        )
    with gr.Row():
        unlock_full_method.change(
            fn=lambda method: {
                "choices": [m for m in unlock_f0(method)["choices"] if m != "hybrid"], 
                "value": "rmvpe", 
                "__type__": "update"
            }, 
            inputs=[
                unlock_full_method
            ], 
            outputs=[
                f0_method_extract
            ]
        )
        extractor_button.click(
            fn=f0_extract,
            inputs=[
                input_audio_path,
                f0_method_extract,
                predictor_onnx,
                pitch,
                filter_radius,
                autotune,
                f0_autotune_strength,
                proposal_pitch,
                proposal_pitch_threshold
            ],
            outputs=[
                file_output, 
                image_output
            ],
            api_name="f0_extract"
        )