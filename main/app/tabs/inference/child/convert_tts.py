import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.tts import TTS
from main.app.core.process import process_input
from main.app.core.inference import convert_tts
from main.app.core.utils import google_translate
from main.app.variables import translations, sample_rate_choice, model_name, index_path, method_f0, f0_file, embedders_mode, embedders_model, edgetts, google_tts_voice, configs
from main.app.core.ui import visible, change_f0_choices, unlock_f0, hoplength_show, change_models_choices, get_index, index_strength_show, visible_embedders, change_tts_voice_choices, shutil_move

def convert_tts_tab():
    with gr.Row():
        gr.Markdown(translations["convert_text_markdown_2"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    use_txt = gr.Checkbox(label=translations["input_txt"], value=False, interactive=True)
                    google_tts_check_box = gr.Checkbox(label=translations["googletts"], value=False, interactive=True)
                prompt = gr.Textbox(label=translations["text_to_speech"], value="", placeholder="Hello Words", lines=3)
        with gr.Column():
            speed = gr.Slider(label=translations["voice_speed"], info=translations["voice_speed_info"], minimum=-100, maximum=100, value=0, step=1)
            pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
    with gr.Row():
        tts_button = gr.Button(translations["tts_1"], variant="primary", scale=2)
        convert_button0 = gr.Button(translations["tts_2"], variant="secondary", scale=2)
    with gr.Row():
        with gr.Column():
            txt_input = gr.File(label=translations["drop_text"], file_types=[".txt", ".srt", ".docx"], visible=use_txt.value)  
            tts_voice = gr.Dropdown(label=translations["voice"], choices=edgetts, interactive=True, value="vi-VN-NamMinhNeural")
            tts_pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info_2"], label=translations["pitch"], value=0, interactive=True)
            with gr.Accordion(translations["translate"], open=False):
                with gr.Row():
                    source_lang = gr.Dropdown(label=translations["source_lang"], choices=["auto"]+google_tts_voice, interactive=True, value="auto")
                    target_lang = gr.Dropdown(label=translations["target_lang"], choices=google_tts_voice, interactive=True, value="en")
                translate_button = gr.Button(translations["translate"])
        with gr.Column():
            with gr.Accordion(translations["model_accordion"], open=True):
                with gr.Row():
                    model_pth0 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                    model_index0 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                with gr.Row():
                    refesh1 = gr.Button(translations["refesh"])
                with gr.Row():
                    index_strength0 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index0.value != "")
            with gr.Accordion(translations["output_path"], open=False):
                export_format0 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                output_audio0 = gr.Textbox(label=translations["output_tts"], value="audios/tts.wav", placeholder="audios/tts.wav", info=translations["tts_output"], interactive=True)
                output_audio1 = gr.Textbox(label=translations["output_tts_convert"], value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info=translations["tts_output"], interactive=True)
            with gr.Accordion(translations["setting"], open=False):
                with gr.Accordion(translations["f0_method"], open=False):
                    with gr.Group():
                        with gr.Row():
                            onnx_f0_mode1 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            unlock_full_method3 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                        method0 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                        hybrid_method0 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method0.value == "hybrid")
                    hop_length0 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                with gr.Accordion(translations["f0_file"], open=False):
                    upload_f0_file0 = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                    f0_file_dropdown0 = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                    refesh_f0_file0 = gr.Button(translations["refesh"])
                with gr.Accordion(translations["hubert_model"], open=False):
                    embed_mode1 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                    embedders0 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                    custom_embedders0 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders0.value == "custom")
                with gr.Group():
                    with gr.Row():
                        formant_shifting1 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)  
                        split_audio0 = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)   
                        cleaner1 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)  
                    with gr.Row():
                        autotune3 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True) 
                        checkpointing0 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)     
                        proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
                with gr.Column():
                    f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune3.value)
                    clean_strength1 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner1.value)
                    resample_sr0 = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                    filter_radius0 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                    volume_envelope0 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                    protect0 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                with gr.Row():
                    formant_qfrency1 = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                    formant_timbre1 = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
    with gr.Row():
        gr.Markdown(translations["output_tts_markdown"])
    with gr.Row():
        tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["output_text_to_speech"])
        tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
    with gr.Row():
        translate_button.click(fn=google_translate, inputs=[prompt, source_lang, target_lang], outputs=[prompt], api_name="google_translate")
    with gr.Row():
        unlock_full_method3.change(fn=unlock_f0, inputs=[unlock_full_method3], outputs=[method0])
        upload_f0_file0.upload(fn=lambda inp: shutil_move(inp.name, configs["f0_path"]), inputs=[upload_f0_file0], outputs=[f0_file_dropdown0])
        refesh_f0_file0.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown0])
    with gr.Row():
        embed_mode1.change(fn=visible_embedders, inputs=[embed_mode1], outputs=[embedders0])
        autotune3.change(fn=visible, inputs=[autotune3], outputs=[f0_autotune_strength0])
        model_pth0.change(fn=get_index, inputs=[model_pth0], outputs=[model_index0])
    with gr.Row():
        cleaner1.change(fn=visible, inputs=[cleaner1], outputs=[clean_strength1])
        method0.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method0, hybrid_method0], outputs=[hybrid_method0, hop_length0])
        hybrid_method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
    with gr.Row():
        refesh1.click(fn=change_models_choices, inputs=[], outputs=[model_pth0, model_index0])
        embedders0.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders0], outputs=[custom_embedders0])
        formant_shifting1.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting1], outputs=[formant_qfrency1, formant_timbre1])
    with gr.Row():
        model_index0.change(fn=index_strength_show, inputs=[model_index0], outputs=[index_strength0])
        txt_input.upload(fn=process_input, inputs=[txt_input], outputs=[prompt])
        use_txt.change(fn=visible, inputs=[use_txt], outputs=[txt_input])
    with gr.Row():
        google_tts_check_box.change(fn=change_tts_voice_choices, inputs=[google_tts_check_box], outputs=[tts_voice])
        tts_button.click(
            fn=TTS, 
            inputs=[
                prompt, 
                tts_voice, 
                speed, 
                output_audio0,
                tts_pitch,
                google_tts_check_box,
                txt_input
            ], 
            outputs=[tts_voice_audio],
            api_name="text-to-speech"
        )
        convert_button0.click(
            fn=convert_tts,
            inputs=[
                cleaner1, 
                autotune3, 
                pitch0, 
                clean_strength1, 
                model_pth0, 
                model_index0, 
                index_strength0, 
                output_audio0, 
                output_audio1,
                export_format0,
                method0, 
                hybrid_method0, 
                hop_length0, 
                embedders0, 
                custom_embedders0, 
                resample_sr0, 
                filter_radius0, 
                volume_envelope0, 
                protect0,
                split_audio0,
                f0_autotune_strength0,
                checkpointing0,
                onnx_f0_mode1,
                formant_shifting1, 
                formant_qfrency1, 
                formant_timbre1,
                f0_file_dropdown0,
                embed_mode1,
                proposal_pitch
            ],
            outputs=[tts_voice_convert],
            api_name="convert_tts"
        )