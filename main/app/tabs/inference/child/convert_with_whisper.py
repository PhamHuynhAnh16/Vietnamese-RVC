import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.inference import convert_with_whisper

from main.app.core.ui import (
    visible, 
    get_index, 
    unlock_f0, 
    shutil_move, 
    hoplength_show, 
    get_speakers_id, 
    index_strength_show, 
    change_audios_choices, 
    change_models_choices, 
    change_embedders_mode 
)

from main.app.variables import (
    configs, 
    method_f0, 
    model_name, 
    index_path, 
    file_types, 
    translations, 
    whisper_model, 
    embedders_mode, 
    embedders_model, 
    paths_for_files, 
    hybrid_f0_method, 
    sample_rate_choice, 
    export_format_choices, 
)

def convert_with_whisper_tab():
    with gr.Row():
        gr.Markdown(translations["convert_with_whisper_info"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    cleaner = gr.Checkbox(
                        label=translations["clear_audio"], 
                        value=False, 
                        interactive=True
                    )
                    autotune = gr.Checkbox(
                        label=translations["autotune"], 
                        value=False, 
                        interactive=True
                    )
                    checkpointing = gr.Checkbox(
                        label=translations["memory_efficient_training"], 
                        value=False, 
                        interactive=True
                    )
                    formant_shifting = gr.Checkbox(
                        label=translations["formantshift"], 
                        value=False, 
                        interactive=True
                    )
                    proposal_pitch = gr.Checkbox(
                        label=translations["proposal_pitch"], 
                        value=False, 
                        interactive=True
                    )
                    audio_processing = gr.Checkbox(
                        label=translations["audio_processing"], 
                        value=False, 
                        interactive=True
                    )
                with gr.Row():
                    num_spk = gr.Slider(
                        minimum=2, 
                        maximum=8, 
                        step=1, 
                        info=translations["num_spk_info"], 
                        label=translations["num_spk"], 
                        value=2, 
                        interactive=True
                    )
    with gr.Row():
        with gr.Column():
            convert_button = gr.Button(
                translations["convert_audio"], 
                variant="primary"
            )
    with gr.Row():
        with gr.Column():
            with gr.Accordion(
                translations["model_accordion"] + " 1", 
                open=True
            ):
                with gr.Row():
                    model_pth_1 = gr.Dropdown(
                        label=translations["model_name"], 
                        choices=model_name, 
                        value=model_name[0] if len(model_name) >= 1 else "", 
                        interactive=True, 
                        allow_custom_value=True
                    )
                    model_index_1 = gr.Dropdown(
                        label=translations["index_path"], 
                        choices=index_path, 
                        value=index_path[0] if len(index_path) >= 1 else "", 
                        interactive=True, 
                        allow_custom_value=True
                    )
                with gr.Row():
                    refresh_model_1 = gr.Button(
                        translations["refresh"]
                    )
                with gr.Row():
                    pitch_1 = gr.Slider(
                        minimum=-20, 
                        maximum=20, 
                        step=1, 
                        info=translations["pitch_info"], 
                        label=translations["pitch"], 
                        value=0, 
                        interactive=True
                    )
                    index_strength_1 = gr.Slider(
                        label=translations["index_strength"], 
                        info=translations["index_strength_info"], 
                        minimum=0, 
                        maximum=1, 
                        value=0.5, 
                        step=0.01, 
                        interactive=True, 
                        visible=True
                    )
                with gr.Row():
                    sid_dict_1 = get_speakers_id(model_pth_1.value)
                    sids_1 = gr.Dropdown(
                        label=translations["sids_label"], 
                        info=translations["sids_info"], 
                        choices=sid_dict_1["choices"], 
                        value=sid_dict_1["value"], 
                        visible=sid_dict_1["visible"], 
                        interactive=True
                    )
            with gr.Accordion(
                translations["input_output"], 
                open=False
            ):
                with gr.Column():
                    export_format = gr.Radio(
                        label=translations["export_format"], 
                        info=translations["export_info"], 
                        choices=export_format_choices, 
                        value="wav", 
                        interactive=True
                    )
                    input_audio = gr.Dropdown(
                        label=translations["audio_path"], 
                        value="", 
                        choices=paths_for_files, 
                        info=translations["provide_audio"], 
                        allow_custom_value=True, 
                        interactive=True
                    )
                    output_audio = gr.Textbox(
                        label=translations["output_path"], 
                        value="audios/output.wav", 
                        placeholder="audios/output.wav", 
                        info=translations["output_path_info"], 
                        interactive=True
                    )
                with gr.Column():
                    refresh_audio = gr.Button(
                        translations["refresh"]
                    )
                with gr.Row():
                    drop_audio = gr.Files(
                        label=translations["drop_audio"], 
                        file_types=file_types
                    )
        with gr.Column():
            with gr.Accordion(
                translations["model_accordion"] + " 2", 
                open=True
            ):
                with gr.Row():
                    model_pth_2 = gr.Dropdown(
                        label=translations["model_name"], 
                        choices=model_name, 
                        value=model_name[0] if len(model_name) >= 1 else "", 
                        interactive=True, 
                        allow_custom_value=True
                    )
                    model_index_2 = gr.Dropdown(
                        label=translations["index_path"], 
                        choices=index_path, 
                        value=index_path[0] if len(index_path) >= 1 else "", 
                        interactive=True, 
                        allow_custom_value=True
                    )
                with gr.Row():
                    refresh_model_2 = gr.Button(
                        translations["refresh"]
                    )
                with gr.Row():
                    pitch_2 = gr.Slider(
                        minimum=-20, 
                        maximum=20, 
                        step=1, 
                        info=translations["pitch_info"], 
                        label=translations["pitch"], 
                        value=0, 
                        interactive=True
                    )
                    index_strength_2 = gr.Slider(
                        label=translations["index_strength"], 
                        info=translations["index_strength_info"], 
                        minimum=0, 
                        maximum=1, 
                        value=0.5, 
                        step=0.01, 
                        interactive=True, 
                        visible=True
                    )
                with gr.Row():
                    sid_dict_2 = get_speakers_id(model_pth_2.value)
                    sids_2 = gr.Dropdown(
                        label=translations["sids_label"], 
                        info=translations["sids_info"], 
                        choices=sid_dict_2["choices"], 
                        value=sid_dict_2["value"], 
                        visible=sid_dict_2["visible"], 
                        interactive=True
                    )
            with gr.Accordion(
                translations["setting"], 
                open=False
            ):
                with gr.Row():
                    model_size = gr.Radio(
                        label=translations["model_size"], 
                        info=translations["model_size_info"], 
                        choices=whisper_model, 
                        value="medium", 
                        interactive=True
                    )
                with gr.Accordion(
                    translations["f0_method"], 
                    open=False
                ):
                    with gr.Group():
                        with gr.Row():
                            predictor_onnx = gr.Checkbox(
                                label=translations["predictor_onnx"], 
                                info=translations["predictor_onnx_info"], 
                                value=False, 
                                interactive=True
                            )
                            unlock_full_method = gr.Checkbox(
                                label=translations["f0_unlock"], 
                                info=translations["f0_unlock_info"], 
                                value=False, 
                                interactive=True
                            )
                        f0_method = gr.Radio(
                            label=translations["f0_method"], 
                            info=translations["f0_method_info"], 
                            choices=method_f0, 
                            value="rmvpe", 
                            interactive=True
                        )
                        hybrid_f0method = gr.Dropdown(
                            label=translations["f0_method_hybrid"], 
                            info=translations["f0_method_hybrid_info"], 
                            choices=hybrid_f0_method, 
                            value=hybrid_f0_method[0], 
                            interactive=True, 
                            allow_custom_value=True, 
                            visible=False
                        )
                    hop_length = gr.Slider(
                        label=translations['hop_length'], 
                        info=translations["hop_length_info"], 
                        minimum=64, 
                        maximum=512, 
                        value=160, 
                        step=1, 
                        interactive=True, 
                        visible=False
                    )
                    alpha = gr.Slider(
                        label=translations["alpha_label"], 
                        info=translations["alpha_info"], 
                        minimum=0.1, 
                        maximum=1, 
                        value=0.5, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                with gr.Accordion(
                    translations["hubert_model"], 
                    open=False
                ):
                    embedder_mode = gr.Radio(
                        label=translations["embed_mode"], 
                        info=translations["embed_mode_info"], 
                        value="fairseq", 
                        choices=embedders_mode, 
                        interactive=True, 
                        visible=True
                    )
                    embedders = gr.Radio(
                        label=translations["hubert_model"], 
                        info=translations["hubert_info"], 
                        choices=embedders_model, 
                        value="hubert_base", 
                        interactive=True
                    )
                    custom_embedders = gr.Textbox(
                        label=translations["modelname"], 
                        info=translations["modelname_info"], 
                        value="", 
                        placeholder="hubert_base", 
                        interactive=True, 
                        visible=False
                    )
                with gr.Column():     
                    resample_sr3 = gr.Radio(
                        choices=[0]+sample_rate_choice, 
                        label=translations["resample"], 
                        info=translations["resample_info"], 
                        value=0, 
                        interactive=True
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
                    clean_strength = gr.Slider(
                        label=translations["clean_strength"], 
                        info=translations["clean_strength_info"], 
                        minimum=0, 
                        maximum=1, 
                        value=0.5, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
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
                    filter_radius = gr.Slider(
                        minimum=0, 
                        maximum=7, 
                        label=translations["filter_radius"], 
                        info=translations["filter_radius_info"], 
                        value=3, 
                        step=1, 
                        interactive=True
                    )
                    rms_mix_rate = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        label=translations["rms_mix_rate"], 
                        info=translations["rms_mix_rate_info"], 
                        value=1, 
                        step=0.1, 
                        interactive=True
                    )
                    protect = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        label=translations["protect"], 
                        info=translations["protect_info"], 
                        value=0.5, 
                        step=0.01, 
                        interactive=True
                    )
                with gr.Row():
                    formant_qfrency = gr.Slider(
                        value=1.0, 
                        label=translations["formant_qfrency"] + " 1", 
                        info=translations["formant_qfrency"], 
                        minimum=0.0, 
                        maximum=16.0, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                    formant_timbre = gr.Slider(
                        value=1.0, 
                        label=translations["formant_timbre"] + " 1", 
                        info=translations["formant_timbre"], 
                        minimum=0.0, 
                        maximum=16.0, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                with gr.Row():
                    formant_qfrency = gr.Slider(
                        value=1.0, 
                        label=translations["formant_qfrency"] + " 2", 
                        info=translations["formant_qfrency"], 
                        minimum=0.0, 
                        maximum=16.0, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
                    formant_timbre = gr.Slider(
                        value=1.0, 
                        label=translations["formant_timbre"] + " 2", 
                        info=translations["formant_timbre"], 
                        minimum=0.0, 
                        maximum=16.0, 
                        step=0.1, 
                        interactive=True, 
                        visible=False
                    )
    with gr.Row():
        gr.Markdown(translations["input_output"])
    with gr.Row():
        play_input_audio = gr.Audio(
            show_download_button=True, 
            interactive=False, 
            label=translations["input_audio"]
        )
        play_output_audio = gr.Audio(
            show_download_button=True, 
            interactive=False, 
            label=translations["output_file_tts_convert"]
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
        cleaner.change(
            fn=visible, 
            inputs=[
                cleaner
            ], 
            outputs=[
                clean_strength
            ]
        )
        f0_method.change(
            fn=lambda method, hybrid: [
                visible(method == "hybrid"), 
                visible(method == "hybrid"), 
                hoplength_show(method, hybrid)
            ], 
            inputs=[
                f0_method, 
                hybrid_f0method
            ], 
            outputs=[
                hybrid_f0method, 
                alpha, 
                hop_length
            ]
        )
    with gr.Row():
        hybrid_f0method.change(
            fn=hoplength_show, 
            inputs=[
                f0_method, 
                hybrid_f0method
            ], 
            outputs=[
                hop_length
            ]
        )
        refresh_model_1.click(
            fn=change_models_choices, 
            inputs=[], 
            outputs=[
                model_pth_1, 
                model_index_1
            ]
        )
        model_pth_1.change(
            fn=get_index, 
            inputs=[
                model_pth_1
            ], 
            outputs=[
                model_index_1
            ]
        )
    with gr.Row():
        refresh_model_2.click(
            fn=change_models_choices, 
            inputs=[], 
            outputs=[
                model_pth_2, 
                model_index_2
            ]
        )
        model_pth_2.change(
            fn=get_index, 
            inputs=[
                model_pth_2
            ], 
            outputs=[
                model_index_2
            ]
        )
        drop_audio.upload(
            fn=lambda audio_in: [
                shutil_move(audio.name, configs["audios_path"]) 
                for audio in audio_in
            ][0], 
            inputs=[
                drop_audio
            ], 
            outputs=[
                input_audio
            ]
        )
    with gr.Row():
        input_audio.change(
            fn=lambda audio: audio if os.path.isfile(audio) else None, 
            inputs=[
                input_audio
            ], 
            outputs=[
                play_input_audio
            ]
        )
        formant_shifting.change(
            fn=lambda a: [
                visible(a) 
                for _ in range(4)
            ], 
            inputs=[
                formant_shifting
            ], 
            outputs=[
                formant_qfrency, 
                formant_timbre, 
                formant_qfrency, 
                formant_timbre
            ]
        )
        embedders.change(
            fn=lambda embedders: visible(embedders == "custom"), 
            inputs=[
                embedders
            ], 
            outputs=[
                custom_embedders
            ]
        )
    with gr.Row():
        refresh_audio.click(
            fn=change_audios_choices, 
            inputs=[
                input_audio
            ], 
            outputs=[
                input_audio
            ]
        )
        model_index_1.change(
            fn=index_strength_show, 
            inputs=[
                model_index_1
            ], 
            outputs=[
                index_strength_1
            ]
        )
        model_index_2.change(
            fn=index_strength_show, 
            inputs=[
                model_index_2
            ], 
            outputs=[
                index_strength_2
            ]
        )
    with gr.Row():
        unlock_full_method.change(
            fn=unlock_f0, 
            inputs=[
                unlock_full_method
            ], 
            outputs=[
                f0_method
            ]
        )
        embedder_mode.change(
            fn=change_embedders_mode, 
            inputs=[
                embedder_mode
            ], 
            outputs=[
                embedders
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
        model_pth_1.change(
            fn=get_speakers_id, 
            inputs=[
                model_pth_1
            ], 
            outputs=[
                sids_1
            ]
        )
        model_pth_2.change(
            fn=get_speakers_id, 
            inputs=[
                model_pth_2
            ], 
            outputs=[
                sids_2
            ]
        )
    with gr.Row():
        convert_button.click(
            fn=convert_with_whisper,
            inputs=[
                num_spk,
                model_size,
                cleaner,
                clean_strength,
                autotune,
                f0_autotune_strength,
                checkpointing,
                model_pth_1,
                model_pth_2,
                model_index_1,
                model_index_2,
                pitch_1,
                pitch_2,
                index_strength_1,
                index_strength_2,
                export_format,
                input_audio,
                output_audio,
                predictor_onnx,
                f0_method,
                hybrid_f0method,
                hop_length,
                embedder_mode,
                embedders,
                custom_embedders,
                resample_sr3,
                filter_radius,
                rms_mix_rate,
                protect,
                formant_shifting,
                formant_qfrency,
                formant_timbre,
                formant_qfrency,
                formant_timbre,
                proposal_pitch,
                proposal_pitch_threshold,
                audio_processing,
                alpha,
                sids_1,
                sids_2
            ],
            outputs=[
                play_output_audio
            ],
            api_name="convert_with_whisper"
        )