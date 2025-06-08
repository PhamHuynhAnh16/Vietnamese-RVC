import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.inference import convert_with_whisper
from main.app.variables import translations, paths_for_files, sample_rate_choice, model_name, index_path, method_f0, embedders_mode, embedders_model, configs
from main.app.core.ui import visible, change_audios_choices, unlock_f0, hoplength_show, change_models_choices, get_index, index_strength_show, visible_embedders, shutil_move

def convert_with_whisper_tab():
    with gr.Row():
        gr.Markdown(translations["convert_with_whisper_info"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    cleaner2 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                    autotune2 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                    checkpointing2 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                    formant_shifting2 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                    proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
                with gr.Row():
                    num_spk = gr.Slider(minimum=2, maximum=8, step=1, info=translations["num_spk_info"], label=translations["num_spk"], value=2, interactive=True)
    with gr.Row():
        with gr.Column():
            convert_button3 = gr.Button(translations["convert_audio"], variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(translations["model_accordion"] + " 1", open=True):
                with gr.Row():
                    model_pth2 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                    model_index2 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                with gr.Row():
                    refesh2 = gr.Button(translations["refesh"])
                with gr.Row():
                    pitch3 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                    index_strength2 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index2.value != "")
            with gr.Accordion(translations["input_output"], open=False):
                with gr.Column():
                    export_format2 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                    input_audio1 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                    output_audio2 = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                with gr.Column():
                    refesh4 = gr.Button(translations["refesh"])
                with gr.Row():
                    input2 = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
        with gr.Column():
            with gr.Accordion(translations["model_accordion"] + " 2", open=True):
                with gr.Row():
                    model_pth3 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                    model_index3 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                with gr.Row():
                    refesh3 = gr.Button(translations["refesh"])
                with gr.Row():
                    pitch4 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                    index_strength3 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index3.value != "")
            with gr.Accordion(translations["setting"], open=False):
                with gr.Row():
                    model_size = gr.Radio(label=translations["model_size"], info=translations["model_size_info"], choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], value="medium", interactive=True)
                with gr.Accordion(translations["f0_method"], open=False):
                    with gr.Group():
                        with gr.Row():
                            onnx_f0_mode4 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            unlock_full_method2 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                        method3 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                        hybrid_method3 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method3.value == "hybrid")
                    hop_length3 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                with gr.Accordion(translations["hubert_model"], open=False):
                    embed_mode3 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                    embedders3 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                    custom_embedders3 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders3.value == "custom")
                with gr.Column():      
                    clean_strength3 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner2.value)
                    f0_autotune_strength3 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune2.value)
                    resample_sr3 = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                    filter_radius3 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                    volume_envelope3 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                    protect3 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                with gr.Row():
                    formant_qfrency3 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 1", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                    formant_timbre3 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 1", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                with gr.Row():
                    formant_qfrency4 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 2", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                    formant_timbre4 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 2", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
    with gr.Row():
        gr.Markdown(translations["input_output"])
    with gr.Row():
        play_audio2 = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
        play_audio3 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
    with gr.Row():
        autotune2.change(fn=visible, inputs=[autotune2], outputs=[f0_autotune_strength3])
        cleaner2.change(fn=visible, inputs=[cleaner2], outputs=[clean_strength3])
        method3.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method3, hybrid_method3], outputs=[hybrid_method3, hop_length3])
    with gr.Row():
        hybrid_method3.change(fn=hoplength_show, inputs=[method3, hybrid_method3], outputs=[hop_length3])
        refesh2.click(fn=change_models_choices, inputs=[], outputs=[model_pth2, model_index2])
        model_pth2.change(fn=get_index, inputs=[model_pth2], outputs=[model_index2])
    with gr.Row():
        refesh3.click(fn=change_models_choices, inputs=[], outputs=[model_pth3, model_index3])
        model_pth3.change(fn=get_index, inputs=[model_pth3], outputs=[model_index3])
        input2.upload(fn=lambda audio_in: shutil_move(audio_in.name, configs["audios_path"]), inputs=[input2], outputs=[input_audio1])
    with gr.Row():
        input_audio1.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio1], outputs=[play_audio2])
        formant_shifting2.change(fn=lambda a: [visible(a)]*4, inputs=[formant_shifting2], outputs=[formant_qfrency3, formant_timbre3, formant_qfrency4, formant_timbre4])
        embedders3.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders3], outputs=[custom_embedders3])
    with gr.Row():
        refesh4.click(fn=change_audios_choices, inputs=[input_audio1], outputs=[input_audio1])
        model_index2.change(fn=index_strength_show, inputs=[model_index2], outputs=[index_strength2])
        model_index3.change(fn=index_strength_show, inputs=[model_index3], outputs=[index_strength3])
    with gr.Row():
        unlock_full_method2.change(fn=unlock_f0, inputs=[unlock_full_method2], outputs=[method3])
        embed_mode3.change(fn=visible_embedders, inputs=[embed_mode3], outputs=[embedders3])
        convert_button3.click(
            fn=convert_with_whisper,
            inputs=[
                num_spk,
                model_size,
                cleaner2,
                clean_strength3,
                autotune2,
                f0_autotune_strength3,
                checkpointing2,
                model_pth2,
                model_pth3,
                model_index2,
                model_index3,
                pitch3,
                pitch4,
                index_strength2,
                index_strength3,
                export_format2,
                input_audio1,
                output_audio2,
                onnx_f0_mode4,
                method3,
                hybrid_method3,
                hop_length3,
                embed_mode3,
                embedders3,
                custom_embedders3,
                resample_sr3,
                filter_radius3,
                volume_envelope3,
                protect3,
                formant_shifting2,
                formant_qfrency3,
                formant_timbre3,
                formant_qfrency4,
                formant_timbre4,
                proposal_pitch
            ],
            outputs=[play_audio3],
            api_name="convert_with_whisper"
        )