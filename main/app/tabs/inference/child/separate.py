import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.downloads import download_url
from main.app.core.separate import separator_music
from main.app.core.ui import visible, valueFalse_interactive, change_audios_choices, shutil_move
from main.app.variables import translations, uvr_model, paths_for_files, mdx_model, sample_rate_choice, configs, file_types, export_format_choices

def separate_tab():
    with gr.Row(): 
        gr.Markdown(translations["4_part"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():       
                    cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True, min_width=140)       
                    backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True, min_width=140)
                    reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True, min_width=140)
                    backing_reverb = gr.Checkbox(label=translations["dereveb_backing"], value=False, interactive=False, min_width=140)               
                    denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False, min_width=140)     
                with gr.Row():
                    separator_model = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
                    separator_backing_model = gr.Dropdown(label=translations["separator_backing_model"], value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=backing.value)
    with gr.Row():
        with gr.Column():
            separator_button = gr.Button(translations["separator_tab"], variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                    segment_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                with gr.Row():
                    mdx_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                with gr.Row():
                    mdx_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
    with gr.Row():
        with gr.Column():
            input = gr.File(label=translations["drop_audio"], file_types=file_types)    
            with gr.Accordion(translations["use_url"], open=False):
                url = gr.Textbox(label=translations["url_audio"], value="", placeholder="https://www.youtube.com/...", scale=6)
                download_button = gr.Button(translations["downloads"])
        with gr.Column():
            with gr.Row():
                clean_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner.value)
                sample_rate1 = gr.Radio(choices=sample_rate_choice, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
            with gr.Accordion(translations["input_output"], open=False):
                format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=export_format_choices, value="wav", interactive=True)
                input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                refresh_separator = gr.Button(translations["refresh"])
                output_separator = gr.Textbox(label=translations["output_folder"], value="audios", placeholder="audios", info=translations["output_folder_info"], interactive=True)
            audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
    with gr.Row():
        gr.Markdown(translations["output_separator"])
    with gr.Row():
        instruments_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["instruments"])
        original_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["original_vocal"])
        main_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["main_vocal"], visible=backing.value)
        backing_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["backing_vocal"], visible=backing.value)
    with gr.Row():
        separator_model.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(c not in mdx_model)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, shifts])
        backing.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(a), visible(a), visible(a), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, separator_backing_model, main_vocals, backing_vocals, backing_reverb])
        reverb.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, backing_reverb])
    with gr.Row():
        input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[audio_input])
        cleaner.change(fn=visible, inputs=[cleaner], outputs=[clean_strength])
    with gr.Row():
        input.upload(fn=lambda audio_in: shutil_move(audio_in.name, configs["audios_path"]), inputs=[input], outputs=[input_audio])
        refresh_separator.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
    with gr.Row():
        download_button.click(
            fn=download_url, 
            inputs=[url], 
            outputs=[input_audio, audio_input, url],
            api_name='download_url'
        )
        separator_button.click(
            fn=separator_music, 
            inputs=[
                input_audio, 
                output_separator,
                format, 
                shifts, 
                segment_size, 
                overlap, 
                cleaner, 
                clean_strength, 
                denoise, 
                separator_model, 
                separator_backing_model, 
                backing,
                reverb, 
                backing_reverb,
                mdx_hop_length,
                mdx_batch_size,
                sample_rate1
            ],
            outputs=[original_vocals, instruments_audio, main_vocals, backing_vocals],
            api_name='separator_music'
        )