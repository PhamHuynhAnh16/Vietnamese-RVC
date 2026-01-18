import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.realtime import (
    realtime_stop,
    change_config,
    realtime_start, 
    soundfile_record_audio,
)

from main.app.variables import (
    configs, 
    method_f0, 
    model_name, 
    index_path, 
    translations, 
    embedders_mode, 
    embedders_model,
    export_format_choices
)

from main.app.core.ui import (
    visible, 
    get_index, 
    unlock_f0, 
    audio_device, 
    visibleFalse, 
    hoplength_show, 
    get_speakers_id, 
    update_audio_device, 
    index_strength_show, 
    change_models_choices, 
    change_embedders_mode, 
    update_value_from_json, 
    update_button_from_json, 
    update_dropdowns_from_json, 
    change_audio_device_choices
)

input_channels_map, output_channels_map = audio_device()
client_mode = "--client" in sys.argv

def realtime_tab():
    with gr.TabItem(translations["realtime"], visible=configs.get("realtime_tab", True)):
        gr.Markdown(translations["realtime_markdown"])
        with gr.Row():
            gr.Markdown(translations["realtime_markdown_2"])
        with gr.Row():
            status = gr.Label(
                label=translations["realtime_latency"], 
                value=translations["realtime_not_startup"], 
                elem_id="realtime-status-info"
            )
        with gr.Row():
            monitor = gr.Checkbox(
                label=translations["monitor"], 
                value=False, 
                interactive=True
            )
            exclusive_mode = gr.Checkbox(
                label=translations["exclusive_mode"], 
                value=False, 
                interactive=True
            )
            vad_enabled = gr.Checkbox(
                label=translations["vad_enabled"], 
                value=False, 
                interactive=True
            )
            clean_audio = gr.Checkbox(
                label=translations["clear_audio"], 
                value=False, 
                interactive=True
            )
        with gr.Row():
            with gr.Accordion(
                translations["audio_device"], 
                open=True
            ):
                with gr.Row():
                    input_audio_device = gr.Dropdown(
                        label=translations["input_audio_device_label"], 
                        info=translations["input_audio_device_info"], 
                        choices=list(input_channels_map.keys()) if not client_mode else [], 
                        value=(list(input_channels_map.keys())[0] if len(list(input_channels_map.keys())) >= 1 else "") if not client_mode else None, 
                        interactive=True
                    )
                    output_audio_device = gr.Dropdown(
                        label=translations["output_audio_device_label"], 
                        info=translations["output_audio_device_info"], 
                        choices=list(output_channels_map.keys()) if not client_mode else [], 
                        value=(list(output_channels_map.keys())[0] if len(list(output_channels_map.keys())) >= 1 else "") if not client_mode else None, 
                        interactive=True
                    )
                    monitor_output_device = gr.Dropdown(
                        label=translations["monitor_output_device_label"], 
                        info=translations["monitor_output_device_info"], 
                        choices=list(output_channels_map.keys()) if not client_mode else [], 
                        value=(list(output_channels_map.keys())[0] if len(list(output_channels_map.keys())) >= 1 else "") if not client_mode else None, 
                        interactive=True, 
                        visible=False
                    )
                with gr.Row():
                    input_audio_gain = gr.Slider(
                        minimum=0, 
                        maximum=2500, 
                        label=translations["input_audio_gain_label"], 
                        info=translations["input_audio_gain_info"], 
                        value=100, 
                        step=1, 
                        interactive=True
                    )
                    output_audio_gain = gr.Slider(
                        minimum=0, 
                        maximum=4000, 
                        label=translations["output_audio_gain_label"], 
                        info=translations["output_audio_gain_info"], 
                        value=100, 
                        step=1, 
                        interactive=True
                    )
                    monitor_audio_gain = gr.Slider(
                        minimum=0, 
                        maximum=4000, 
                        label=translations["monitor_audio_gain_label"], 
                        info=translations["monitor_audio_gain_info"], 
                        value=100, 
                        step=1, 
                        interactive=True, 
                        visible=False
                    )
                with gr.Row(visible=False) as asio_row:
                    input_asio_channels = gr.Slider(
                        minimum=-1, 
                        maximum=128, 
                        label=translations["input_asio_channels_label"], 
                        info=translations["input_asio_channels_info"], 
                        value=-1, 
                        step=1, 
                        interactive=True, 
                        visible=False
                    )
                    output_asio_channels = gr.Slider(
                        minimum=-1, 
                        maximum=128, 
                        label=translations["output_asio_channels_label"], 
                        info=translations["output_asio_channels_info"], 
                        value=-1, 
                        step=1, 
                        interactive=True, 
                        visible=False
                    )
                    monitor_asio_channels = gr.Slider(
                        minimum=-1, 
                        maximum=128, 
                        label=translations["monitor_asio_channels_label"], 
                        info=translations["monitor_asio_channels_info"], 
                        value=-1, 
                        step=1, 
                        interactive=True, 
                        visible=False
                    )
                with gr.Row():
                    refresh_audio_device = gr.Button(
                        value=translations["refresh_audio_device"], 
                        variant="secondary"
                    )
        with gr.Row():
            start_realtime = gr.Button(
                value=translations["start_realtime_button"], 
                variant="primary", 
                interactive=True
            )
            stop_realtime = gr.Button(
                value=translations["stop_realtime_button"], 
                variant="stop", 
                interactive=False
            )
        with gr.Row():
            with gr.Accordion(translations["record"], open=False):
                with gr.Column():
                    record_audio_path = gr.Textbox(
                        label=translations["output_path"],
                        placeholder="audios/record_audio.wav",
                        info=translations["output_path_info"],
                        value="audios/record_audio.wav",
                        interactive=True,
                    )
                export_format = gr.Radio(
                    label=translations["export_format"],
                    info=translations["export_info"],
                    choices=export_format_choices,
                    value="wav",
                    interactive=True,
                )
                record_audio = gr.Button(translations["start_record"])
                record_output = gr.Audio(label=translations["output_audio"])
        with gr.Row():
            chunk_size = gr.Slider(
                minimum=2.7, 
                maximum=2730.7, 
                step=0.1, 
                label=translations["chunk_size"], 
                info=translations["chunk_size_info"], 
                value=1024, 
                interactive=True
            )
            pitch = gr.Slider(
                minimum=-20, 
                maximum=20, 
                step=1, 
                info=translations["pitch_info"], 
                label=translations["pitch"], 
                value=0, 
                interactive=True
            )
        with gr.Row():
            with gr.Column():
                with gr.Accordion(
                    translations["model_accordion"], 
                    open=True
                ):
                    with gr.Row():
                        model_pth = gr.Dropdown(
                            label=translations["model_name"], 
                            choices=model_name, 
                            value=model_name[0] if len(model_name) >= 1 else "", 
                            interactive=True, 
                            allow_custom_value=True
                        )
                        model_index = gr.Dropdown(
                            label=translations["index_path"], 
                            choices=index_path, 
                            value=index_path[0] if len(index_path) >= 1 else "", 
                            interactive=True, 
                            allow_custom_value=True
                        )
                    with gr.Row():
                        model_refresh = gr.Button(
                            translations["refresh"]
                        )
                    with gr.Row():
                        index_strength = gr.Slider(
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
                        sid_dict = get_speakers_id(model_pth.value)
                        sids = gr.Dropdown(
                            label=translations["sids_label"], 
                            info=translations["sids_info"], 
                            choices=sid_dict["choices"], 
                            value=sid_dict["value"], 
                            visible=sid_dict["visible"], 
                            interactive=True
                        )
            with gr.Column():
                with gr.Accordion(
                    translations["f0_method"], 
                    open=True
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
                            choices=[m for m in method_f0 if m != "hybrid"], 
                            value="rmvpe", 
                            interactive=True
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
            with gr.Column():
                with gr.Accordion(
                    translations["hubert_model"], 
                    open=True
                ):
                    embedder_mode = gr.Radio(
                        label=translations["embed_mode"], 
                        info=translations["embed_mode_info"], 
                        value="fairseq", 
                        choices=embedders_mode, 
                        interactive=True, 
                        visible=True)
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
        with gr.Row():
                with gr.Accordion(
                    translations["setting"], 
                    open=True
                ):
                    with gr.Row():
                        f0_autotune = gr.Checkbox(
                            label=translations["autotune"], 
                            value=False, 
                            interactive=True
                        )
                        proposal_pitch = gr.Checkbox(
                            label=translations["proposal_pitch"], 
                            value=False, 
                            interactive=True
                        )
                    with gr.Group():
                        with gr.Row():
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
                        with gr.Row():
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
                            filter_radius = gr.Slider(
                                minimum=0, 
                                maximum=7, 
                                label=translations["filter_radius"], 
                                info=translations["filter_radius_info"], 
                                value=3, 
                                step=1, 
                                interactive=True
                            )
                    with gr.Column():
                        silent_threshold = gr.Slider(
                            minimum=-90, 
                            maximum=-60, 
                            label=translations["silent_threshold_label"], 
                            info=translations["silent_threshold_info"], 
                            value=-90, 
                            step=1, 
                            interactive=True)
                        extra_convert_size = gr.Slider(
                            minimum=0.1, 
                            maximum=5, 
                            label=translations["extra_convert_size_label"], 
                            info=translations["extra_convert_size_info"], 
                            value=0.5, 
                            step=0.1, 
                            interactive=True)
                        cross_fade_overlap_size = gr.Slider(
                            minimum=0.05, 
                            maximum=0.2, 
                            label=translations["cross_fade_overlap_size_label"], 
                            info=translations["cross_fade_overlap_size_info"], 
                            value=0.1, 
                            step=0.01, 
                            interactive=True)
                    with gr.Row():
                        vad_sensitivity = gr.Slider(
                            minimum=0, 
                            maximum=3, 
                            label=translations["vad_sensitivity_label"], 
                            info=translations["vad_sensitivity_info"], 
                            value=3, 
                            step=1, 
                            interactive=True, 
                            visible=False
                        )
                        vad_frame_ms = gr.Slider(
                            minimum=10, 
                            maximum=30, 
                            label=translations["vad_frame_ms_label"], 
                            info=translations["vad_frame_ms_info"], 
                            value=30, 
                            step=10, 
                            interactive=True, 
                            visible=False
                        )
                    with gr.Row():
                        post_process = gr.Checkbox(
                            label=translations["audio_effects"], 
                            value=False, 
                            interactive=True
                        )
                        reverb = gr.Checkbox(
                            label=translations["reverb"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        chorus = gr.Checkbox(
                            label=translations["chorus"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        delay = gr.Checkbox(
                            label=translations["delay"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        phaser = gr.Checkbox(
                            label=translations["phaser"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        compressor = gr.Checkbox(
                            label=translations["compressor"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                    with gr.Row():
                        limiter = gr.Checkbox(
                            label=translations["limiter"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        distortion = gr.Checkbox(
                            label=translations["distortion"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        pitch_shift = gr.Checkbox(
                            label=translations["pitch"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        gain = gr.Checkbox(
                            label=translations["gain"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        bitcrush = gr.Checkbox(
                            label=translations["bitcrush"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                        clipping = gr.Checkbox(
                            label=translations["clipping"], 
                            value=False, 
                            interactive=True, 
                            visible=False
                        )
                    with gr.Accordion(
                        translations["reverb"], 
                        open=True, 
                        visible=False
                    ) as reverb_accordion:
                        reverb_freeze_mode = gr.Checkbox(
                            label=translations["reverb_freeze"], 
                            info=translations["reverb_freeze_info"], 
                            value=False, 
                            interactive=True
                        )
                        reverb_room_size = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.15, 
                            label=translations["room_size"], 
                            info=translations["room_size_info"], 
                            interactive=True
                        )
                        reverb_damping = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.7, 
                            label=translations["damping"], 
                            info=translations["damping_info"], 
                            interactive=True
                        )
                        reverb_wet_gain = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.2, 
                            label=translations["wet_level"], 
                            info=translations["wet_level_info"], 
                            interactive=True
                        )
                        reverb_dry_gain = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.8, 
                            label=translations["dry_level"], 
                            info=translations["dry_level_info"], 
                            interactive=True
                        )
                        reverb_width = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=1, 
                            label=translations["width"], 
                            info=translations["width_info"], 
                            interactive=True
                        )
                    with gr.Accordion(
                        translations["chorus"], 
                        open=True, 
                        visible=False
                    ) as chorus_accordion:
                        chorus_depth = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["chorus_depth"], 
                            info=translations["chorus_depth_info"], 
                            interactive=True
                        )
                        chorus_rate = gr.Slider(
                            minimum=0.1, 
                            maximum=10, 
                            step=0.1, 
                            value=1.5, 
                            label=translations["chorus_rate_hz"], 
                            info=translations["chorus_rate_hz_info"], 
                            interactive=True
                        )
                        chorus_mix = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["chorus_mix"], 
                            info=translations["chorus_mix_info"], 
                            interactive=True
                        )
                        chorus_center_delay = gr.Slider(
                            minimum=0, 
                            maximum=50, 
                            step=1, 
                            value=10, 
                            label=translations["chorus_center_delay_ms"], 
                            info=translations["chorus_center_delay_ms_info"], 
                            interactive=True
                        )
                        chorus_feedback = gr.Slider(
                            minimum=-1, 
                            maximum=1, 
                            step=0.01, 
                            value=0, 
                            label=translations["chorus_feedback"], 
                            info=translations["chorus_feedback_info"], 
                            interactive=True
                        )
                    with gr.Accordion(
                        translations["phaser"], 
                        open=True, 
                        visible=False
                    ) as phaser_accordion:
                        phaser_depth = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["phaser_depth"], 
                            info=translations["phaser_depth_info"], 
                            interactive=True
                        )
                        phaser_rate_hz = gr.Slider(
                            minimum=0.1, 
                            maximum=10, 
                            step=0.1, 
                            value=1, 
                            label=translations["phaser_rate_hz"], 
                            info=translations["phaser_rate_hz_info"], 
                            interactive=True
                        )
                        phaser_mix = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["phaser_mix"], 
                            info=translations["phaser_mix_info"], 
                            interactive=True
                        )
                        phaser_centre_frequency_hz = gr.Slider(
                            minimum=50, 
                            maximum=5000, 
                            step=10, 
                            value=1000, 
                            label=translations["phaser_centre_frequency_hz"], 
                            info=translations["phaser_centre_frequency_hz_info"], 
                            interactive=True
                        )
                        phaser_feedback = gr.Slider(
                            minimum=-1, 
                            maximum=1, 
                            step=0.01, 
                            value=0, 
                            label=translations["phaser_feedback"], 
                            info=translations["phaser_feedback_info"], 
                            interactive=True
                        )
                    with gr.Accordion(
                        translations["delay"], 
                        open=True, 
                        visible=False
                    ) as delay_accordion:
                        delay_seconds = gr.Slider(
                            minimum=0, 
                            maximum=5, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["delay_seconds"], 
                            info=translations["delay_seconds_info"], 
                            interactive=True
                        )
                        delay_feedback = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["delay_feedback"], 
                            info=translations["delay_feedback_info"], 
                            interactive=True
                        )
                        delay_mix = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.01, 
                            value=0.5, 
                            label=translations["delay_mix"], 
                            info=translations["delay_mix_info"], 
                            interactive=True
                        )
                    with gr.Accordion(
                        translations["compressor"], 
                        open=True, 
                        visible=False
                    ) as compressor_accordion:
                        compressor_threshold = gr.Slider(
                            minimum=-60, 
                            maximum=0, 
                            step=1, 
                            value=-20, 
                            label=translations["compressor_threshold_db"], 
                            info=translations["compressor_threshold_db_info"], 
                            interactive=True
                        )
                        compressor_ratio = gr.Slider(
                            minimum=1, 
                            maximum=20, 
                            step=0.1, 
                            value=1, 
                            label=translations["compressor_ratio"], 
                            info=translations["compressor_ratio_info"], 
                            interactive=True
                        )
                        compressor_attack = gr.Slider(
                            minimum=0.1, 
                            maximum=100, 
                            step=0.1, 
                            value=10, 
                            label=translations["compressor_attack_ms"], 
                            info=translations["compressor_attack_ms_info"], 
                            interactive=True
                        )
                        compressor_release = gr.Slider(
                            minimum=10, 
                            maximum=1000, 
                            step=1, 
                            value=100, 
                            label=translations["compressor_release_ms"], 
                            info=translations["compressor_release_ms_info"], 
                            interactive=True
                        )
                    with gr.Accordion(
                        translations["limiter"], 
                        open=True, 
                        visible=False
                    ) as limiter_accordion:
                        limiter_threshold = gr.Slider(
                            minimum=-60, 
                            maximum=0, 
                            step=1, 
                            value=-1, 
                            label=translations["limiter_threshold_db"], 
                            info=translations["limiter_threshold_db_info"], 
                            interactive=True
                        )
                        limiter_release_time = gr.Slider(
                            minimum=10, 
                            maximum=1000, 
                            step=1, 
                            value=100, 
                            label=translations["limiter_release_ms"], 
                            info=translations["limiter_release_ms_info"], 
                            interactive=True
                        )
                    with gr.Row():
                        distortion_gain = gr.Slider(
                            minimum=0, 
                            maximum=50, 
                            step=1, 
                            value=20, 
                            label=translations["distortion"], 
                            info=translations["distortion_info"], 
                            interactive=True, 
                            visible=False
                        )
                        pitch_shift_semitones = gr.Slider(
                            minimum=-20, 
                            maximum=20, 
                            step=1, 
                            value=0, 
                            label=translations["pitch"], 
                            info=translations["pitch_info"], 
                            interactive=True, 
                            visible=False
                        )
                        gain_db = gr.Slider(
                            minimum=-60, 
                            maximum=60, 
                            step=1, 
                            value=0, 
                            label=translations["gain"], 
                            info=translations["gain_info"], 
                            interactive=True, 
                            visible=False
                        )
                        bitcrush_bit_depth = gr.Slider(
                            minimum=1, 
                            maximum=24, 
                            step=1, 
                            value=16, 
                            label=translations["bitcrush_bit_depth"], 
                            info=translations["bitcrush_bit_depth_info"], 
                            interactive=True, 
                            visible=False
                        )
                        clipping_threshold = gr.Slider(
                            minimum=-60, 
                            maximum=0, 
                            step=1, 
                            value=-1, 
                            label=translations["clipping_threshold_db"], 
                            info=translations["clipping_threshold_db_info"], 
                            interactive=True, 
                            visible=False
                        )
        with gr.Row():
            json_audio_hidden = gr.JSON(
                visible=False
            )
            json_button_hidden = gr.JSON(
                visible=False
            )
            json_value_hidden = gr.JSON(
                visible=False
            )
        with gr.Row():
            post_process.change(
                fn=lambda a: [
                    visibleFalse(a) 
                    for _ in range(11)
                ],
                inputs=[
                    post_process
                ],
                outputs=[
                    reverb, 
                    chorus, 
                    delay, 
                    phaser, 
                    compressor, 
                    pitch_shift, 
                    limiter, 
                    distortion, 
                    gain, 
                    bitcrush, 
                    clipping
                ]
            )
            reverb.change(
                fn=visible,
                inputs=[
                    reverb
                ],
                outputs=[
                    reverb_accordion
                ]
            )
            chorus.change(
                fn=visible,
                inputs=[
                    chorus
                ],
                outputs=[
                    chorus_accordion
                ]
            )
        with gr.Row():
            delay.change(
                fn=visible,
                inputs=[
                    delay
                ],
                outputs=[
                    delay_accordion
                ]
            )
            phaser.change(
                fn=visible,
                inputs=[
                    phaser
                ],
                outputs=[
                    phaser_accordion
                ]
            )
            compressor.change(
                fn=visible,
                inputs=[
                    compressor
                ],
                outputs=[
                    compressor_accordion
                ]
            )
        with gr.Row():
            limiter.change(
                fn=visible,
                inputs=[
                    limiter
                ],
                outputs=[
                    limiter_accordion
                ]
            )
            distortion.change(
                fn=visible,
                inputs=[
                    distortion
                ],
                outputs=[
                    distortion_gain
                ]
            )
            pitch_shift.change(
                fn=visible,
                inputs=[
                    pitch_shift
                ],
                outputs=[
                    pitch_shift_semitones
                ]
            )
        with gr.Row():
            gain.change(
                fn=visible,
                inputs=[
                    gain
                ],
                outputs=[
                    gain_db
                ]
            )
            bitcrush.change(
                fn=visible,
                inputs=[
                    bitcrush
                ],
                outputs=[
                    bitcrush_bit_depth
                ]
            )
            clipping.change(
                fn=visible,
                inputs=[
                    clipping
                ],
                outputs=[
                    clipping_threshold
                ]
            )
        with gr.Row():
            model_pth.change(
                fn=get_index, 
                inputs=[
                    model_pth
                ], 
                outputs=[
                    model_index
                ]
            )
            model_index.change(
                fn=index_strength_show, 
                inputs=[
                    model_index
                ], 
                outputs=[
                    index_strength
                ]
            )
            model_refresh.click(
                fn=change_models_choices, 
                inputs=[], 
                outputs=[
                    model_pth, 
                    model_index
                ]
            )
        with gr.Row():
            unlock_full_method.change(
                fn=lambda f0_method: {
                    "choices": [
                        m for m in unlock_f0(f0_method)["choices"] 
                        if m != "hybrid"
                    ], 
                    "value": "rmvpe", 
                    "__type__": "update"
                }, 
                inputs=[
                    unlock_full_method
                ], 
                outputs=[
                    f0_method
                ]
            )
            f0_method.change(
                fn=lambda f0_method: hoplength_show(f0_method, None), 
                inputs=[
                    f0_method
                ], 
                outputs=[
                    hop_length
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
        with gr.Row():
            embedders.change(
                fn=lambda embedders: visible(embedders == "custom"), 
                inputs=[
                    embedders
                ], 
                outputs=[
                    custom_embedders
                ]
            )
            f0_autotune.change(
                fn=visible, 
                inputs=[
                    f0_autotune
                ], 
                outputs=[
                    f0_autotune_strength
                ]
            )
            clean_audio.change(
                fn=visible, 
                inputs=[
                    clean_audio
                ], 
                outputs=[
                    clean_strength
                ]
            )
        with gr.Row():
            if not client_mode:
                input_audio_device.change(
                    fn=update_audio_device,
                    inputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device, 
                        monitor
                    ],
                    outputs=[
                        monitor_output_device, 
                        monitor_audio_gain, 
                        monitor_asio_channels, 
                        asio_row, 
                        input_asio_channels, 
                        output_asio_channels, 
                        monitor_asio_channels
                    ]
                )
                output_audio_device.change(
                    fn=update_audio_device,
                    inputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device, 
                        monitor
                    ],
                    outputs=[
                        monitor_output_device, 
                        monitor_audio_gain, 
                        monitor_asio_channels, 
                        asio_row, 
                        input_asio_channels, 
                        output_asio_channels, 
                        monitor_asio_channels
                    ]
                )
                monitor_output_device.change(
                    fn=update_audio_device,
                    inputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device, 
                        monitor
                    ],
                    outputs=[
                        monitor_output_device, 
                        monitor_audio_gain, 
                        monitor_asio_channels, 
                        asio_row, 
                        input_asio_channels, 
                        output_asio_channels, 
                        monitor_asio_channels
                    ]
                )
        with gr.Row():
            if not client_mode:
                monitor.change(
                    fn=update_audio_device,
                    inputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device, 
                        monitor
                    ],
                    outputs=[
                        monitor_output_device, 
                        monitor_audio_gain, 
                        monitor_asio_channels, 
                        asio_row, 
                        input_asio_channels, 
                        output_asio_channels, 
                        monitor_asio_channels
                    ]
                )
        with gr.Row():
            proposal_pitch.change(
                fn=visible, 
                inputs=[
                    proposal_pitch
                ], 
                outputs=[
                    proposal_pitch_threshold
                ]
            )
            vad_enabled.change(
                fn=lambda a: [
                    visible(a) 
                    for _ in range(2)
                ],
                inputs=[
                    vad_enabled
                ],
                outputs=[
                    vad_sensitivity, 
                    vad_frame_ms
                ]
            )
            if not client_mode:
                refresh_audio_device.click(
                    fn=change_audio_device_choices,
                    inputs=[],
                    outputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device
                    ]
                )
            else:
                refresh_audio_device.click(
                    fn=None,
                    js="getAudioDevices",
                    inputs=[],
                    outputs=[
                        json_audio_hidden
                    ]
                )
        with gr.Row():
            model_pth.change(
                fn=get_speakers_id, 
                inputs=[
                    model_pth
                ], 
                outputs=[
                    sids
                ]
            )
            if client_mode:
                json_audio_hidden.change(
                    fn=update_dropdowns_from_json,
                    inputs=[
                        json_audio_hidden
                    ],
                    outputs=[
                        input_audio_device, 
                        output_audio_device, 
                        monitor_output_device
                    ]
                )
                json_button_hidden.change(
                    fn=update_button_from_json,
                    inputs=[
                        json_button_hidden
                    ],
                    outputs=[
                        start_realtime, 
                        stop_realtime
                    ]
                )
                json_value_hidden.change(
                    fn=update_value_from_json,
                    inputs=[
                        json_value_hidden
                    ],
                    outputs=[
                        record_audio, 
                        record_output
                    ]
                )
        with gr.Row():
            if not client_mode:
                start_realtime.click(
                    fn=realtime_start,
                    inputs=[
                        monitor,
                        exclusive_mode,
                        vad_enabled,
                        input_audio_device,
                        output_audio_device,
                        monitor_output_device,
                        input_audio_gain,
                        output_audio_gain,
                        monitor_audio_gain,
                        input_asio_channels,
                        output_asio_channels,
                        monitor_asio_channels,
                        chunk_size,
                        pitch,
                        model_pth,
                        model_index,
                        index_strength,
                        predictor_onnx,
                        f0_method,
                        hop_length,
                        embedder_mode,
                        embedders,
                        custom_embedders,
                        f0_autotune,
                        proposal_pitch,
                        f0_autotune_strength,
                        proposal_pitch_threshold,
                        rms_mix_rate,
                        protect,
                        filter_radius,
                        silent_threshold,
                        extra_convert_size,
                        cross_fade_overlap_size,
                        vad_sensitivity,
                        vad_frame_ms,
                        clean_audio,
                        clean_strength,
                        post_process,
                        sids,
                        chorus,
                        distortion,
                        reverb,
                        pitch_shift,
                        delay,
                        compressor,
                        limiter,
                        gain,
                        bitcrush,
                        clipping,
                        phaser,
                        chorus_depth,
                        chorus_rate,
                        chorus_mix,
                        chorus_center_delay,
                        chorus_feedback,
                        distortion_gain,
                        reverb_room_size,
                        reverb_damping,
                        reverb_wet_gain,
                        reverb_dry_gain,
                        reverb_width,
                        reverb_freeze_mode,
                        pitch_shift_semitones,
                        delay_seconds,
                        delay_feedback,
                        delay_mix,
                        compressor_threshold,
                        compressor_ratio,
                        compressor_attack,
                        compressor_release,
                        limiter_threshold,
                        limiter_release_time,
                        gain_db,
                        bitcrush_bit_depth,
                        clipping_threshold,
                        phaser_rate_hz,
                        phaser_depth,
                        phaser_centre_frequency_hz,
                        phaser_feedback,
                        phaser_mix
                    ],
                    outputs=[
                        status, 
                        start_realtime, 
                        stop_realtime
                    ]
                )
                stop_realtime.click(
                    fn=realtime_stop,
                    inputs=[],
                    outputs=[
                        status, 
                        start_realtime, 
                        stop_realtime
                    ]
                )
                record_audio.click(
                    fn=soundfile_record_audio,
                    inputs=[
                        record_audio,
                        record_audio_path,
                        export_format
                    ],
                    outputs=[
                        record_audio,
                        record_output
                    ]
                )
            else:
                start_realtime.click(
                    fn=None,
                    js="StreamAudioRealtime",
                    inputs=[
                        monitor,
                        vad_enabled,
                        input_audio_device,
                        output_audio_device,
                        monitor_output_device,
                        input_audio_gain,
                        output_audio_gain,
                        monitor_audio_gain,
                        chunk_size,
                        pitch,
                        model_pth,
                        model_index,
                        index_strength,
                        predictor_onnx,
                        f0_method,
                        hop_length,
                        embedder_mode,
                        embedders,
                        custom_embedders,
                        f0_autotune,
                        proposal_pitch,
                        f0_autotune_strength,
                        proposal_pitch_threshold,
                        rms_mix_rate,
                        protect,
                        filter_radius,
                        silent_threshold,
                        extra_convert_size,
                        cross_fade_overlap_size,
                        vad_sensitivity,
                        vad_frame_ms,
                        clean_audio,
                        clean_strength,
                        exclusive_mode,
                        post_process,
                        sids,
                        chorus,
                        distortion,
                        reverb,
                        pitch_shift,
                        delay,
                        compressor,
                        limiter,
                        gain,
                        bitcrush,
                        clipping,
                        phaser,
                        chorus_depth,
                        chorus_rate,
                        chorus_mix,
                        chorus_center_delay,
                        chorus_feedback,
                        distortion_gain,
                        reverb_room_size,
                        reverb_damping,
                        reverb_wet_gain,
                        reverb_dry_gain,
                        reverb_width,
                        reverb_freeze_mode,
                        pitch_shift_semitones,
                        delay_seconds,
                        delay_feedback,
                        delay_mix,
                        compressor_threshold,
                        compressor_ratio,
                        compressor_attack,
                        compressor_release,
                        limiter_threshold,
                        limiter_release_time,
                        gain_db,
                        bitcrush_bit_depth,
                        clipping_threshold,
                        phaser_rate_hz,
                        phaser_depth,
                        phaser_centre_frequency_hz,
                        phaser_feedback,
                        phaser_mix
                    ],
                    outputs=[
                        json_button_hidden
                    ]
                )
                stop_realtime.click(
                    fn=None,
                    js="StopAudioStream",
                    inputs=[],
                    outputs=[
                        json_button_hidden
                    ]
                )
                record_audio.click(
                    fn=None,
                    js="SoundfileRecordAudio",
                    inputs=[
                        record_audio,
                        record_audio_path,
                        export_format
                    ],
                    outputs=[
                        json_value_hidden
                    ]
                )
        with gr.Row():
            cross_fade_overlap_size.change(
                js="(value) => window.ChangeConfig(value, 'cross_fade_overlap_size')" if client_mode else None, 
                fn=lambda value: change_config(value, "cross_fade_overlap_size") if not client_mode else None, 
                inputs=[
                    cross_fade_overlap_size
                ], 
                outputs=[]
            )
            extra_convert_size.change(
                js="(value) => window.ChangeConfig(value, 'extra_convert_size')" if client_mode else None, 
                fn=lambda value: change_config(value, "extra_convert_size") if not client_mode else None, 
                inputs=[
                    extra_convert_size
                ], 
                outputs=[]
            )
            silent_threshold.change(
                js="(value) => window.ChangeConfig(value, 'silent_threshold')" if client_mode else None, 
                fn=lambda value: change_config(value, "silent_threshold") if not client_mode else None, 
                inputs=[
                    silent_threshold
                ], 
                outputs=[]
            )
        with gr.Row():
            vad_enabled.change(
                js="(value) => window.ChangeConfig(value, 'vad_enabled')" if client_mode else None, 
                fn=lambda value: change_config(value, "vad_enabled") if not client_mode else None, 
                inputs=[
                    vad_enabled
                ], 
                outputs=[]
            )
            vad_sensitivity.change(
                js="(value) => window.ChangeConfig(value, 'vad_sensitivity')" if client_mode else None, 
                fn=lambda value: change_config(value, "vad_sensitivity") if not client_mode else None, 
                inputs=[
                    vad_sensitivity
                ], 
                outputs=[]
            )            
            vad_frame_ms.change(
                js="(value) => window.ChangeConfig(value, 'vad_frame_ms')" if client_mode else None, 
                fn=lambda value: change_config(value, "vad_frame_ms") if not client_mode else None, 
                inputs=[
                    vad_frame_ms
                ], 
                outputs=[]
            )
        with gr.Row():
            input_audio_gain.change(
                js="(value) => window.ChangeConfig(value, 'input_audio_gain')" if client_mode else None, 
                fn=lambda value: change_config(value / 100.0, "input_audio_gain") if not client_mode else None, 
                inputs=[
                    input_audio_gain
                ], 
                outputs=[]
            )
            output_audio_gain.change(
                js="(value) => window.ChangeConfig(value, 'output_audio_gain')" if client_mode else None, 
                fn=lambda value: change_config(value / 100.0, "output_audio_gain") if not client_mode else None, 
                inputs=[
                    output_audio_gain
                ], 
                outputs=[]
            )
            monitor_audio_gain.change(
                js="(value) => window.ChangeConfig(value, 'monitor_audio_gain')" if client_mode else None, 
                fn=lambda value: change_config(value / 100.0, "monitor_audio_gain") if not client_mode else None, 
                inputs=[
                    monitor_audio_gain
                ], 
                outputs=[]
            )
        with gr.Row():
            index_strength.change(
                js="(value) => window.ChangeConfig(value, 'index_rate')" if client_mode else None, 
                fn=lambda value: change_config(value, "index_rate") if not client_mode else None, 
                inputs=[
                    index_strength
                ], 
                outputs=[]
            )
            rms_mix_rate.change(
                js="(value) => window.ChangeConfig(value, 'rms_mix_rate')" if client_mode else None, 
                fn=lambda value: change_config(value, "rms_mix_rate") if not client_mode else None, 
                inputs=[
                    rms_mix_rate
                ], 
                outputs=[]
            )
            protect.change(
                js="(value) => window.ChangeConfig(value, 'protect')" if client_mode else None, 
                fn=lambda value: change_config(value, "protect") if not client_mode else None, 
                inputs=[
                    protect
                ], 
                outputs=[]
            )
        with gr.Row():
            pitch.change(
                js="(value) => window.ChangeConfig(value, 'f0_up_key')" if client_mode else None, 
                fn=lambda value: change_config(value, "f0_up_key") if not client_mode else None, 
                inputs=[
                    pitch
                ], 
                outputs=[]
            )
            f0_autotune.change(
                js="(value) => window.ChangeConfig(value, 'f0_autotune')" if client_mode else None, 
                fn=lambda value: change_config(value, "f0_autotune") if not client_mode else None, 
                inputs=[
                    f0_autotune
                ], 
                outputs=[]
            )
            f0_autotune_strength.change(
                js="(value) => window.ChangeConfig(value, 'f0_autotune_strength')" if client_mode else None, 
                fn=lambda value: change_config(value, "f0_autotune_strength") if not client_mode else None, 
                inputs=[
                    f0_autotune_strength
                ], 
                outputs=[]
            )
        with gr.Row():
            proposal_pitch.change(
                js="(value) => window.ChangeConfig(value, 'proposal_pitch')" if client_mode else None, 
                fn=lambda value: change_config(value, "proposal_pitch") if not client_mode else None, 
                inputs=[
                    proposal_pitch
                ], 
                outputs=[]
            )
            proposal_pitch_threshold.change(
                js="(value) => window.ChangeConfig(value, 'proposal_pitch_threshold')" if client_mode else None, 
                fn=lambda value: change_config(value, "proposal_pitch_threshold") if not client_mode else None, 
                inputs=[
                    proposal_pitch_threshold
                ], 
                outputs=[]
            )
            clean_audio.change(
                js="(value) => window.ChangeConfig(value, 'clean_audio')" if client_mode else None, 
                fn=lambda value: change_config(value, "clean_audio") if not client_mode else None, 
                inputs=[
                    clean_audio
                ], 
                outputs=[]
            )
        with gr.Row():
            clean_strength.change(
                js="(value) => window.ChangeConfig(value, 'clean_strength')" if client_mode else None, 
                fn=lambda value: change_config(value, "clean_strength") if not client_mode else None, 
                inputs=[
                    clean_strength
                ], 
                outputs=[]
            )
            post_process.change(
                js="(value) => window.ChangeConfig(value, 'post_process')" if client_mode else None, 
                fn=lambda value: change_config(value, "post_process") if not client_mode else None, 
                inputs=[post_process], 
                outputs=[]
            )
            reverb.change(
                js="(value) => window.ChangeConfig(value, 'reverb', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb
                ], 
                outputs=[]
            )
        with gr.Row():
            chorus.change(
                js="(value) => window.ChangeConfig(value, 'chorus', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus
                ], 
                outputs=[]
            )
            delay.change(
                js="(value) => window.ChangeConfig(value, 'delay', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "delay", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    delay
                ], 
                outputs=[]
            )
            phaser.change(
                js="(value) => window.ChangeConfig(value, 'phaser', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    phaser
                ], 
                outputs=[]
            )
        with gr.Row():
            compressor.change(
                js="(value) => window.ChangeConfig(value, 'compressor', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "compressor", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    compressor
                ], 
                outputs=[]
            )
            limiter.change(
                js="(value) => window.ChangeConfig(value, 'limiter', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "limiter", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    limiter
                ], 
                outputs=[]
            )
            distortion.change(
                js="(value) => window.ChangeConfig(value, 'distortion', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "distortion", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    distortion
                ], 
                outputs=[]
            )
        with gr.Row():
            pitch_shift.change(
                js="(value) => window.ChangeConfig(value, 'pitch_shift', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "pitch_shift", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    pitch_shift
                ], 
                outputs=[]
            )
            gain.change(
                js="(value) => window.ChangeConfig(value, 'gain', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "gain", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    gain
                ], 
                outputs=[]
            )
            bitcrush.change(
                js="(value) => window.ChangeConfig(value, 'bitcrush', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "bitcrush", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    bitcrush
                ], 
                outputs=[]
            )
        with gr.Row():
            clipping.change(
                js="(value) => window.ChangeConfig(value, 'clipping', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "clipping", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    clipping
                ], 
                outputs=[]
            )
            pitch_shift_semitones.change(
                js="(value) => window.ChangeConfig(value, 'pitch_shift_semitones', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "pitch_shift_semitones", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    pitch_shift_semitones
                ], 
                outputs=[]
            )
            gain_db.change(
                js="(value) => window.ChangeConfig(value, 'gain_db', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "gain_db", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    gain_db
                ], 
                outputs=[]
            )
        with gr.Row():
            phaser_depth.change(
                js="(value) => window.ChangeConfig(value, 'phaser_depth', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser_depth", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    phaser_depth
                ], 
                outputs=[]
            )
            phaser_rate_hz.change(
                js="(value) => window.ChangeConfig(value, 'phaser_rate_hz', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser_rate_hz", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    phaser_rate_hz
                ], 
                outputs=[]
            )
            phaser_mix.change(
                js="(value) => window.ChangeConfig(value, 'phaser_mix', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser_mix", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    phaser_mix
                ], 
                outputs=[]
            )
        with gr.Row():
            phaser_centre_frequency_hz.change(
                js="(value) => window.ChangeConfig(value, 'phaser_centre_frequency_hz', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser_centre_frequency_hz", if_kwargs=True) if not client_mode else None, 
                inputs=[phaser_centre_frequency_hz], 
                outputs=[]
            )
            phaser_feedback.change(
                js="(value) => window.ChangeConfig(value, 'phaser_feedback', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "phaser_feedback", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    phaser_feedback
                ], 
                outputs=[]
            )
            reverb_room_size.change(
                js="(value) => window.ChangeConfig(value, 'reverb_room_size', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_room_size", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_room_size
                ], 
                outputs=[]
            )
        with gr.Row():
            reverb_damping.change(
                js="(value) => window.ChangeConfig(value, 'reverb_damping', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_damping", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_damping
                ], 
                outputs=[]
            )
            reverb_wet_gain.change(
                js="(value) => window.ChangeConfig(value, 'reverb_wet_level', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_wet_level", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_wet_gain
                ], 
                outputs=[]
            )
            reverb_dry_gain.change(
                js="(value) => window.ChangeConfig(value, 'reverb_dry_level', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_dry_level", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_dry_gain
                ], 
                outputs=[]
            )
        with gr.Row():    
            reverb_width.change(
                js="(value) => window.ChangeConfig(value, 'reverb_width', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_width", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_width
                ], 
                outputs=[]
            )
            reverb_freeze_mode.change(
                js="(value) => window.ChangeConfig(value, 'reverb_freeze_mode', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "reverb_freeze_mode", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    reverb_freeze_mode
                ], 
                outputs=[]
            )
            limiter_threshold.change(
                js="(value) => window.ChangeConfig(value, 'limiter_threshold', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "limiter_threshold", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    limiter_threshold
                ], 
                outputs=[]
            )
        with gr.Row():    
            limiter_release_time.change(
                js="(value) => window.ChangeConfig(value, 'limiter_release', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "limiter_release", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    limiter_release_time
                ], 
                outputs=[]
            )
            chorus_rate.change(
                js="(value) => window.ChangeConfig(value, 'chorus_rate', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus_rate", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus_rate
                ], 
                outputs=[]
            )
            chorus_depth.change(
                js="(value) => window.ChangeConfig(value, 'chorus_depth', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus_depth", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus_depth
                ], 
                outputs=[]
            )
        with gr.Row():    
            chorus_center_delay.change(
                js="(value) => window.ChangeConfig(value, 'chorus_delay', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus_delay", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus_center_delay
                ], 
                outputs=[]
            )
            chorus_feedback.change(
                js="(value) => window.ChangeConfig(value, 'chorus_feedback', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus_feedback", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus_feedback
                ], 
                outputs=[]
            )
            chorus_mix.change(
                js="(value) => window.ChangeConfig(value, 'chorus_mix', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "chorus_mix", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    chorus_mix
                ], 
                outputs=[]
            )
        with gr.Row():
            bitcrush_bit_depth.change(
                js="(value) => window.ChangeConfig(value, 'bitcrush_bit_depth', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "bitcrush_bit_depth", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    bitcrush_bit_depth
                ], 
                outputs=[]
            )
            clipping_threshold.change(
                js="(value) => window.ChangeConfig(value, 'clipping_threshold', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "clipping_threshold", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    clipping_threshold
                ], 
                outputs=[]
            )
            distortion_gain.change(
                js="(value) => window.ChangeConfig(value, 'distortion_gain', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "distortion_gain", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    distortion_gain
                ], 
                outputs=[]
            )
        with gr.Row():
            compressor_threshold.change(
                js="(value) => window.ChangeConfig(value, 'compressor_threshold', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "compressor_threshold", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    compressor_threshold
                ], 
                outputs=[]
            )
            compressor_ratio.change(
                js="(value) => window.ChangeConfig(value, 'compressor_ratio', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "compressor_ratio", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    compressor_ratio
                ], 
                outputs=[]
            )
            compressor_attack.change(
                js="(value) => window.ChangeConfig(value, 'compressor_attack', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "compressor_attack", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    compressor_attack
                ], 
                outputs=[]
            )
        with gr.Row():
            compressor_release.change(
                js="(value) => window.ChangeConfig(value, 'compressor_release', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "compressor_release", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    compressor_release
                ], 
                outputs=[]
            )
            delay_seconds.change(
                js="(value) => window.ChangeConfig(value, 'delay_seconds', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "delay_seconds", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    delay_seconds
                ], 
                outputs=[]
            )
            delay_feedback.change(
                js="(value) => window.ChangeConfig(value, 'delay_feedback', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "delay_feedback", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    delay_feedback
                ], 
                outputs=[]
            )
        with gr.Row():
            delay_mix.change(
                js="(value) => window.ChangeConfig(value, 'delay_mix', if_kwargs=true)" if client_mode else None, 
                fn=lambda value: change_config(value, "delay_mix", if_kwargs=True) if not client_mode else None, 
                inputs=[
                    delay_mix
                ], 
                outputs=[]
            )
            model_pth.change(
                js="(value) => window.ChangeConfig(value, 'model_path')" if client_mode else None, 
                fn=lambda value: change_config(value, "model_path") if not client_mode else None, 
                inputs=[model_pth], 
                outputs=[]
            )
            model_index.change(
                js="(value) => window.ChangeConfig(value, 'index_path')" if client_mode else None, 
                fn=lambda value: change_config(value, "index_path") if not client_mode else None, 
                inputs=[model_index], 
                outputs=[]
            )
        with gr.Row():
            f0_method.change(
                js="(value) => window.ChangeConfig(value, 'f0_method')" if client_mode else None, 
                fn=lambda value: change_config(value, "f0_method") if not client_mode else None, 
                inputs=[f0_method], 
                outputs=[]
            )
            predictor_onnx.change(
                js="(value) => window.ChangeConfig(value, 'predictor_onnx')" if client_mode else None, 
                fn=lambda value: change_config(value, "predictor_onnx") if not client_mode else None, 
                inputs=[predictor_onnx], 
                outputs=[]
            )
            embedders.change(
                js="(value) => window.ChangeConfig(value, 'embedder_model')" if client_mode else None, 
                fn=lambda value: change_config(value, "embedder_model") if not client_mode else None, 
                inputs=[embedders], 
                outputs=[]
            )
        with gr.Row():
            custom_embedders.change(
                js="(value) => window.ChangeConfig(value, 'embedder_model_custom')" if client_mode else None, 
                fn=lambda value: change_config(value, "embedder_model_custom") if not client_mode else None, 
                inputs=[custom_embedders], 
                outputs=[]
            )
            embedder_mode.change(
                js="(value) => window.ChangeConfig(value, 'embedders_mode')" if client_mode else None, 
                fn=lambda value: change_config(value, "embedders_mode") if not client_mode else None, 
                inputs=[embedder_mode], 
                outputs=[]
            )
