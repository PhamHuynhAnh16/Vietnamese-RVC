import os
import io
import ssl
import sys
import time
import codecs
import logging
import warnings

import gradio as gr

sys.path.append(os.getcwd())
start_time = time.time()

from main.app.tabs.extra.extra import extra_tab
from main.app.tabs.editing.editing import editing_tab
from main.app.tabs.training.training import training_tab
from main.app.tabs.realtime.realtime import realtime_tab
from main.app.tabs.downloads.downloads import download_tab
from main.app.tabs.inference.inference import inference_tab
from main.configs.rpc import connect_discord_ipc, send_discord_rpc
from main.app.variables import logger, config, translations, theme, font, configs, language, allow_disk

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

js_code = """
() => {
    window._activeStream = null;
    window._audioCtx = null;
    window._workletNode = null;
    window._playbackNode = null;
    window._ws = null;
    window.OutputAudioRoute = null;
    window.MonitorAudioRoute = null;
    window.lastSend = 0;
    window.responseMs = 0;

    function setStatus(msg, use_alert = true) {
        const realtimeStatus = document.querySelector("#realtime-status-info h2.output-class");
        if (use_alert) alert(msg);

        if (realtimeStatus) {
            realtimeStatus.innerText = msg;
            realtimeStatus.style.whiteSpace = "nowrap";
            realtimeStatus.style.textAlign = "center";
        }
    }

    async function addModuleFromString(ctx, codeStr) {
        const blob = new Blob([codeStr], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);

        await ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
    };

    function createOutputRoute(audioCtx, playbackNode, sinkId, gainValue = 1.0) {
        const dest = audioCtx.createMediaStreamDestination();
        const gainNode = audioCtx.createGain();
        gainNode.gain.value = gainValue;

        playbackNode.connect(gainNode);
        gainNode.connect(dest);

        const el = document.createElement('audio');
        el.autoplay = true;
        el.srcObject = dest.stream;
        el.style.display = 'none';
        document.body.appendChild(el);

        if (el.setSinkId) el.setSinkId(sinkId).catch(err => console.error(err));
        return { dest, gainNode, el };
    }

    const inputWorkletSource = `
        class InputProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.buffer = new Float32Array(0);
                this.block_frame = 128;
                this.port.onmessage = (e) => {
                    if (e.data && e.data.block_frame) this.block_frame = e.data.block_frame;
                };
            }

            process(inputs) {
                const input = inputs[0];
                if (!input || !input[0]) return true;
                const frame = input[0];

                const newBuf = new Float32Array(this.buffer.length + frame.length);
                newBuf.set(this.buffer, 0);
                newBuf.set(frame, this.buffer.length);
                this.buffer = newBuf;

                while (this.buffer.length >= this.block_frame) {
                    const chunk = this.buffer.slice(0, this.block_frame);

                    this.port.postMessage({chunk}, [chunk.buffer]);
                    this.buffer = this.buffer.slice(this.block_frame);
                }

                return true;
            }
        }
        registerProcessor('input-processor', InputProcessor);
        `;

        const playbackWorkletSource = `
            class PlaybackProcessor extends AudioWorkletProcessor {
                constructor(options) {
                    super(options);
                    const bufferSize = options.processorOptions && options.processorOptions.bufferSize ? options.processorOptions.bufferSize: 98304;
                    this.buffer = new Float32Array(bufferSize); 
                    this.bufferCapacity = bufferSize; 
                    this.writePointer = 0;
                    this.readPointer = 0;
                    this.availableSamples = 0;
                    this.port.onmessage = (e) => {
                        if (e.data && e.data.chunk) {
                            const chunk = new Float32Array(e.data.chunk);
                            const chunkSize = chunk.length;

                            if (this.availableSamples + chunkSize > this.bufferCapacity) return;

                            for (let i = 0; i < chunkSize; i++) {
                                this.buffer[this.writePointer] = chunk[i];
                                this.writePointer = (this.writePointer + 1) % this.bufferCapacity;
                            }

                            this.availableSamples += chunkSize;
                        }
                    };
                }

                process(inputs, outputs) {
                    const output = outputs[0];
                    if (!output || !output[0]) return true;

                    const frame = output[0];
                    const frameSize = frame.length;

                    if (this.availableSamples >= frameSize) {
                        for (let i = 0; i < frameSize; i++) {
                            frame[i] = this.buffer[this.readPointer];
                            this.readPointer = (this.readPointer + 1) % this.bufferCapacity;
                        }
                        this.availableSamples -= frameSize;
                    } else {
                        frame.fill(0);
                    }

                    if (output.length > 1) output[1].set(output[0]);
                    return true;
                }
            }
            registerProcessor('playback-processor', PlaybackProcessor);
            `;

    window.getAudioDevices = async function() {
        if (!navigator.mediaDevices) {
            setStatus("__MEDIA_DEVICES__");
            return {"inputs": {}, "outputs": {}};
        }

        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error(err);
            setStatus("__MIC_INACCESSIBLE__")

            return {"inputs": {}, "outputs": {}};
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = {};
        const outputs = {};
        
        for (const device of devices) {
            if (device.kind === "audioinput") {
                inputs[device.label] = device.deviceId
            } else if (device.kind === "audiooutput") {
                outputs[device.label] = device.deviceId
            }
        }

        if (!Object.keys(inputs).length && !Object.keys(outputs).length) return {"inputs": {}, "outputs": {}};
        return {"inputs": inputs, "outputs": outputs};
    };
        
    window.StreamAudioRealtime = async function(
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
        embed_mode,
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
        sid,
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
        phaser_mix,
    ) {
        const SampleRate = 48000;
        const ReadChunkSize = Math.round(chunk_size * SampleRate / 1000 / 128);
        const block_frame = parseInt(ReadChunkSize) * 128;
        const ButtonState = { start_button: true, stop_button: false };
        const devices = await window.getAudioDevices();

        input_audio_device = devices["inputs"][input_audio_device];
        output_audio_device = devices["outputs"][output_audio_device];
        if (monitor && devices["outputs"][monitor_output_device]) monitor_output_device = devices["outputs"][monitor_output_device];

        try {
            if (!input_audio_device || !output_audio_device) {
                setStatus("__PROVIDE_AUDIO_DEVICE__");
                return ButtonState;
            }

            if (monitor && !monitor_output_device) {
                setStatus("__PROVIDE_MONITOR_DEVICE__");
                return ButtonState;
            }

            if (!model_pth) {
                setStatus("__PROVIDE_MODEL__")
                return ButtonState;
            }

            setStatus("__START_REALTIME__", false)

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: { exact: input_audio_device },
                    channelCount: 1,
                    sampleRate: SampleRate,
                    echoCancellation: !exclusive_mode,
                    noiseSuppression: !exclusive_mode,
                    autoGainControl: !exclusive_mode
                }
            });

            let latencyHint = "playback";
            if (exclusive_mode) latencyHint = "interactive";

            window._activeStream = stream;
            window._audioCtx = new AudioContext({ sampleRate: SampleRate, latencyHint: latencyHint });

            await addModuleFromString(window._audioCtx, inputWorkletSource);
            await addModuleFromString(window._audioCtx, playbackWorkletSource);

            const src = window._audioCtx.createMediaStreamSource(stream);
            const inputNode = new AudioWorkletNode(window._audioCtx, 'input-processor');
            const playbackNode = new AudioWorkletNode(window._audioCtx, 'playback-processor', {
                processorOptions: {
                    bufferSize: block_frame * 2
                }
            });

            inputNode.port.postMessage({ block_frame: block_frame });
            src.connect(inputNode);

            window.OutputAudioRoute = createOutputRoute(window._audioCtx, playbackNode, output_audio_device, output_audio_gain / 100);
            if (monitor && monitor_output_device) window.MonitorAudioRoute = createOutputRoute(window._audioCtx, playbackNode, monitor_output_device, monitor_audio_gain / 100);
            
            const protocol = (location.protocol === "https:") ? "wss:" : "ws:";
            const wsUrl = protocol + '//' + location.hostname + `:${location.port}` + '/api/ws-audio';
            const ws = new WebSocket(wsUrl);

            ButtonState.start_button = false;
            ButtonState.stop_button = true;

            ws.binaryType = "arraybuffer";
            window._ws = ws;

            ws.onopen = () => {
                console.log("__WS_CONNECTED__")

                ws.send(
                    JSON.stringify({
                        type: 'init',
                        chunk_size: ReadChunkSize,
                        embedders: embedders,
                        model_pth: model_pth,
                        custom_embedders: custom_embedders,
                        cross_fade_overlap_size: cross_fade_overlap_size,
                        extra_convert_size: extra_convert_size,
                        model_index: model_index,
                        f0_method: f0_method,
                        predictor_onnx: predictor_onnx,
                        embedders_mode: embed_mode,
                        hop_length: hop_length,
                        silent_threshold: silent_threshold,
                        vad_enabled: vad_enabled,
                        vad_sensitivity: vad_sensitivity,
                        vad_frame_ms: vad_frame_ms,
                        clean_audio: clean_audio,
                        clean_strength: clean_strength,
                        f0_up_key: pitch,
                        index_rate: index_strength,
                        protect: protect,
                        filter_radius: filter_radius,
                        rms_mix_rate: rms_mix_rate,
                        f0_autotune: f0_autotune,
                        f0_autotune_strength: f0_autotune_strength,
                        proposal_pitch: proposal_pitch,
                        proposal_pitch_threshold: proposal_pitch_threshold,
                        input_audio_gain: input_audio_gain,
                        post_process: post_process,
                        sid: sid,
                        kwargs: {
                            chorus: chorus,
                            distortion: distortion,
                            reverb: reverb,
                            pitch_shift: pitch_shift,
                            delay: delay,
                            compressor: compressor,
                            limiter: limiter,
                            gain: gain,
                            bitcrush: bitcrush,
                            clipping: clipping,
                            phaser: phaser,
                            chorus_depth: chorus_depth,
                            chorus_rate: chorus_rate,
                            chorus_mix: chorus_mix,
                            chorus_delay: chorus_center_delay,
                            chorus_feedback: chorus_feedback,
                            distortion_gain: distortion_gain,
                            reverb_room_size: reverb_room_size,
                            reverb_damping: reverb_damping,
                            reverb_wet_level: reverb_wet_gain,
                            reverb_dry_level: reverb_dry_gain,
                            reverb_width: reverb_width,
                            reverb_freeze_mode: reverb_freeze_mode,
                            pitch_shift_semitones: pitch_shift_semitones,
                            delay_seconds: delay_seconds,
                            delay_feedback: delay_feedback,
                            delay_mix: delay_mix,
                            compressor_threshold: compressor_threshold,
                            compressor_ratio: compressor_ratio,
                            compressor_attack: compressor_attack,
                            compressor_release: compressor_release,
                            limiter_threshold: limiter_threshold,
                            limiter_release: limiter_release_time,
                            gain_db: gain_db,
                            bitcrush_bit_depth: bitcrush_bit_depth,
                            clipping_threshold: clipping_threshold,
                            phaser_rate_hz: phaser_rate_hz,
                            phaser_depth: phaser_depth,
                            phaser_centre_frequency_hz: phaser_centre_frequency_hz,
                            phaser_feedback: phaser_feedback,
                            phaser_mix: phaser_mix
                        }
                    })
                );
            };

            inputNode.port.onmessage = (e) => {
                const chunk = e.data && e.data.chunk;

                if (!chunk) return;
                if (ws.readyState === WebSocket.OPEN) {
                    window.lastSend = performance.now();
                    ws.send(chunk);
                }
            };

            ws.onmessage = (ev) => {
                if (typeof ev.data === 'string') {
                    const msg = JSON.parse(ev.data);

                    if (msg.type === 'latency') setStatus(`__LATENCY__: ${msg.value.toFixed(2)} ms | __VOLUME__: ${msg.volume.toFixed(2)} dB | __RESPONSE__: ${window.responseMs.toFixed(2)} ms`, use_alert=false)
                    if (msg.type === 'warnings') {
                        setStatus(msg.value);
                        StopAudioStream();
                    }

                    return;
                }

                const ab = ev.data;
                playbackNode.port.postMessage({ chunk: ab }, [ab]);
                window.responseMs = performance.now() - window.lastSend;
            };

            ws.onclose = () => console.log("__WS_CLOSED__");
            window._workletNode = inputNode;
            window._playbackNode = playbackNode;

            if (window._audioCtx.state === 'suspended') await window._audioCtx.resume();

            console.log("__REALTIME_STARTED__");
            return ButtonState;
        } catch (err) {
            console.error("__ERROR__", err);
            alert("__ERROR__" + err.message);

            return StopAudioStream();
        }
    };

    window.ChangeConfig = async function(value, key, if_kwargs=false) {
        if (key === "output_audio_gain") {
            window.OutputAudioRoute.gainNode.gain.value = value / 100
        } else if (key == "monitor_audio_gain") {
            if (window.MonitorAudioRoute) window.MonitorAudioRoute.gainNode.gain.value = value / 100
        } else {
            const protocol = (location.protocol === "https:") ? "wss:" : "ws:";
            const wsUrl = protocol + '//' + location.hostname + `:${location.port}` + '/api/change-config';
            const ws = new WebSocket(wsUrl);

            ws.binaryType = "arraybuffer";
            ws.onopen = () => {
                ws.send(
                    JSON.stringify({
                        type: 'init',
                        key: key,
                        value: value,
                        if_kwargs: if_kwargs
                    })
                );
    
                ws.close();
            };
        }
    }

    window.StopAudioStream = async function() {
        try {
            if (window._ws) {
                window._ws.close();
                window._ws = null;
            }

            if (window._activeStream) {
                window._activeStream.getTracks().forEach(t => t.stop());
                window._activeStream = null;
            }

            if (window._workletNode) {
                window._workletNode.disconnect();
                window._workletNode = null;
            }

            if (window._playbackNode) {
                window._playbackNode.disconnect();
                window._playbackNode = null;
            }

            if (window._audioCtx) {
                await window._audioCtx.close();
                window._audioCtx = null;
            }

            if (window.OutputAudioRoute) window.OutputAudioRoute = null;
            if (window.MonitorAudioRoute) window.MonitorAudioRoute = null;

            document.querySelectorAll('audio').forEach(a => a.remove());
            setStatus("__REALTIME_HAS_STOP__", false);

            return {"start_button": true, "stop_button": false};
        } catch (e) {
            setStatus(`__ERROR__ ${e}`);

            return {"start_button": false, "stop_button": true}
        }
    };

    window.SoundfileRecordAudio = async function (RecordButton, RecordAudioPath, ExportFormat) {
        const protocol = (location.protocol === "https:") ? "https:" : "http:";
        const url = protocol + '//' + location.hostname + `:${location.port}` + '/api/record';

        const res = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                record_button: RecordButton,
                record_audio_path: RecordAudioPath,
                export_format: ExportFormat
            })
        });

        const msg = await res.json();

        if (msg.type === "info" || msg.type === "warnings") {
            alert(msg.value);

            return {
                "button": msg.button,
                "path": msg.path
            };
        }
    };
}
""".replace(
    "__MEDIA_DEVICES__", translations["media_devices"]
).replace(
    "__MIC_INACCESSIBLE__", translations["mic_inaccessible"]
).replace(
    "__PROVIDE_AUDIO_DEVICE__", translations["provide_audio_device"]
).replace(
    "__PROVIDE_MONITOR_DEVICE__", translations["provide_monitor_device"]
).replace(
    "__START_REALTIME__", translations["start_realtime"]
).replace(
    "__LATENCY__", translations['latency']
).replace(
    "__WS_CONNECTED__", translations["ws_connected"]
).replace(
    "__WS_CLOSED__", translations["ws_closed"]
).replace(
    "__REALTIME_STARTED__", translations["realtime_is_ready"]
).replace(
    "__ERROR__", translations["error_occurred"].format(e="")
).replace(
    "__REALTIME_HAS_STOP__", translations["realtime_has_stop"]
).replace(
    "__PROVIDE_MODEL__", translations["provide_file"].format(filename=translations["model"])
).replace(
    "__VOLUME__", translations["volume"]
).replace(
    "__RESPONSE__", translations["response"]
)

css = """
<style>
  @import url('{font_urls}');

  * {{
    font-family: '{fonts}', cursive !important;
  }}

  html,
  body {{
    font-family: '{fonts}', cursive !important;
  }}

  h1, h2, h3, h4, h5, h6,
  p,
  button,
  input,
  textarea,
  label,
  span,
  div,
  select {{
    font-family: '{fonts}', cursive !important;
  }}
</style>
""".format(
    font_urls=font or "https://fonts.googleapis.com/css2?family=Saira&display=swap",
    fonts=(
        font or "https://fonts.googleapis.com/css2?family=Saira&display=swap"
    ).replace("https://fonts.googleapis.com/css2?family=", "").replace("+", " ").split(":")[0].split("&")[0]

)

client_mode = "--client" in sys.argv

with gr.Blocks(
    title="📱 Vietnamese-RVC GUI BY ANH", 
    js=js_code if client_mode else None, 
    theme=theme, 
    css=css
) as app:
    gr.HTML("<h1 style='text-align: center;'>🎵VIETNAMESE RVC BY ANH🎵</h1>")
    gr.HTML(f"<h3 style='text-align: center;'>{translations['title']}</h3>")

    with gr.Tabs():      
        inference_tab()
        editing_tab()
        realtime_tab()
        training_tab()
        download_tab()
        extra_tab(app)

    with gr.Row(): 
        gr.Markdown(
            translations["rick_roll"].format(
                rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')
            )
        )

    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])

    with gr.Row():
        gr.Markdown(translations["exemption"])
    
    if __name__ == "__main__":
        logger.info(config.device.replace("privateuseone", "dml"))
        logger.info(translations["start_app"])
        logger.info(translations["set_lang"].format(lang=language))

        port = configs.get("app_port", 7860)
        server_name = configs.get("server_name", "0.0.0.0")
        share = "--share" in sys.argv

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        for i in range(configs.get("num_of_restart", 5)):
            try:
                gradio_app, _, share_url = app.queue().launch(
                    favicon_path=configs["ico_path"], 
                    server_name=server_name, 
                    server_port=port, 
                    show_error=configs.get("app_show_error", False), 
                    inbrowser="--open" in sys.argv, 
                    share=share, 
                    allowed_paths=allow_disk,
                    prevent_thread_lock=True,
                    quiet=True
                )
                break
            except OSError:
                logger.debug(translations["port"].format(port=port))
                port -= 1
            except Exception as e:
                logger.error(translations["error_occurred"].format(e=e))
                sys.exit(1)

        if client_mode:
            from main.inference.realtime.client import app as fastapi_app
            gradio_app.mount("/api", fastapi_app)
        
        sys.stdout = original_stdout

        if configs.get("discord_presence", True):
            pipe = connect_discord_ipc()
            if pipe:
                try:
                    logger.info(translations["start_rpc"])
                    send_discord_rpc(pipe)
                except KeyboardInterrupt:
                    logger.info(translations["stop_rpc"])
                    pipe.close()

        logger.info(f"{translations['running_local_url']}: {server_name}:{port}")
        if share: logger.info(f"{translations['running_share_url']}: {share_url}")
        logger.info(f"{translations['gradio_start']}: {(time.time() - start_time):.2f}s")

        while 1:
            time.sleep(5)