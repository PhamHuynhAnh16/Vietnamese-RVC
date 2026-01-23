import os
import sys
import time

sys.path.append(os.getcwd())

from main.app.variables import translations, configs, config
from main.app.core.ui import gr_info, gr_warning, audio_device

running, callbacks, audio_manager = False, None, None

PIPELINE_SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000

interactive_true = {"interactive": True, "__type__": "update"}
interactive_false = {"interactive": False, "__type__": "update"}
callbacks_kwargs = {}

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

def realtime_start(
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
):
    global running, callbacks, audio_manager, callbacks_kwargs
    running = True

    gr_info(translations["start_realtime"])

    yield (
        translations["start_realtime"], 
        interactive_false, 
        interactive_true
    )

    if not input_audio_device or not output_audio_device:
        gr_warning(translations["provide_audio_device"])

        yield (
            translations["provide_audio_device"], 
            interactive_true, 
            interactive_false
        )
        return

    if monitor and not monitor_output_device:
        gr_warning(translations["provide_monitor_device"])

        yield (
            translations["provide_monitor_device"], 
            interactive_true, 
            interactive_false
        )
        return

    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth
    embedder_model = (embedders if embedders != "custom" else custom_embedders)

    if (
        not model_pth or 
        not os.path.exists(model_pth) or 
        os.path.isdir(model_pth) or 
        not model_pth.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))

        yield (
            translations["provide_file"].format(filename=translations["model"]), 
            interactive_true, 
            interactive_false
        )
        return

    input_devices, output_devices = audio_device()
    input_device_id = input_devices[input_audio_device][0]
    output_device_id = output_devices[output_audio_device][0]
    output_monitor_id = output_devices[monitor_output_device][0] if monitor else None

    input_audio_gain /= 100.0
    output_audio_gain /= 100.0
    monitor_audio_gain /= 100.0

    chunk_size = int(chunk_size * DEVICE_SAMPLE_RATE / 1000 / 128)

    from main.inference.realtime.callbacks import AudioCallbacks

    callbacks_kwargs = {
        "pass_through": False, 
        "read_chunk_size": chunk_size, 
        "cross_fade_overlap_size": cross_fade_overlap_size, 
        "input_sample_rate": DEVICE_SAMPLE_RATE, 
        "output_sample_rate": DEVICE_SAMPLE_RATE, 
        "extra_convert_size": extra_convert_size, 
        "model_path": model_pth, 
        "index_path": model_index, 
        "f0_method": f0_method, 
        "predictor_onnx": predictor_onnx, 
        "embedder_model": embedder_model, 
        "embedders_mode": embed_mode, 
        "sample_rate": PIPELINE_SAMPLE_RATE, 
        "hop_length": hop_length, 
        "silent_threshold": silent_threshold, 
        "f0_up_key": pitch, 
        "index_rate": index_strength, 
        "protect": protect, 
        "filter_radius": filter_radius, 
        "rms_mix_rate": rms_mix_rate,
        "f0_autotune": f0_autotune, 
        "f0_autotune_strength": f0_autotune_strength, 
        "proposal_pitch": proposal_pitch, 
        "proposal_pitch_threshold": proposal_pitch_threshold,
        "input_audio_gain": input_audio_gain, 
        "output_audio_gain": output_audio_gain,
        "monitor_audio_gain": monitor_audio_gain,
        "monitor": monitor,
        "vad_enabled": vad_enabled,
        "vad_sensitivity": vad_sensitivity,
        "vad_frame_ms": vad_frame_ms,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "post_process": post_process,
        "sid": sid,
        "kwargs": {
            "chorus": chorus,
            "distortion": distortion,
            "reverb": reverb,
            "pitch_shift": pitch_shift,
            "delay": delay,
            "compressor": compressor,
            "limiter": limiter,
            "gain": gain,
            "bitcrush": bitcrush,
            "clipping": clipping,
            "phaser": phaser,
            "chorus_depth": chorus_depth,
            "chorus_rate": chorus_rate,
            "chorus_mix": chorus_mix,
            "chorus_delay": chorus_center_delay,
            "chorus_feedback": chorus_feedback,
            "distortion_gain": distortion_gain,
            "reverb_room_size": reverb_room_size,
            "reverb_damping": reverb_damping,
            "reverb_wet_level": reverb_wet_gain,
            "reverb_dry_level": reverb_dry_gain,
            "reverb_width": reverb_width,
            "reverb_freeze_mode": reverb_freeze_mode,
            "pitch_shift_semitones": pitch_shift_semitones,
            "delay_seconds": delay_seconds,
            "delay_feedback": delay_feedback,
            "delay_mix": delay_mix,
            "compressor_threshold": compressor_threshold,
            "compressor_ratio": compressor_ratio,
            "compressor_attack": compressor_attack,
            "compressor_release": compressor_release,
            "limiter_threshold": limiter_threshold,
            "limiter_release": limiter_release_time,
            "gain_db": gain_db,
            "bitcrush_bit_depth": bitcrush_bit_depth,
            "clipping_threshold": clipping_threshold,
            "phaser_rate_hz": phaser_rate_hz,
            "phaser_depth": phaser_depth,
            "phaser_centre_frequency_hz": phaser_centre_frequency_hz,
            "phaser_feedback": phaser_feedback,
            "phaser_mix": phaser_mix
        }
    }

    callbacks = AudioCallbacks(**callbacks_kwargs)

    audio_manager = callbacks.audio
    audio_manager.start(
        input_device_id=input_device_id, 
        output_device_id=output_device_id, 
        output_monitor_id=output_monitor_id, 
        exclusive_mode=exclusive_mode, 
        asio_input_channel=input_asio_channels, 
        asio_output_channel=output_asio_channels, 
        asio_output_monitor_channel=monitor_asio_channels,
        read_chunk_size=chunk_size, 
        input_audio_sample_rate=DEVICE_SAMPLE_RATE, 
        output_monitor_sample_rate=DEVICE_SAMPLE_RATE
    )

    gr_info(translations["realtime_is_ready"])

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)
        if hasattr(callbacks, "latency") and hasattr(callbacks, "volume"):
            yield (
                f"{translations['latency']}: {callbacks.latency:.2f} ms | {translations['volume']}: {callbacks.volume:.2f} dB", 
                interactive_false, 
                interactive_true
            )

    return (
        translations["realtime_has_stop"], 
        interactive_true, 
        interactive_false
    )

def change_callbacks_config():
    global callbacks

    if running and audio_manager is not None and callbacks is not None:
        crossfade_frame = int(callbacks_kwargs.get("cross_fade_overlap_size", 0.1) * DEVICE_SAMPLE_RATE)
        extra_frame = int(callbacks_kwargs.get("extra_convert_size", 0.5) * DEVICE_SAMPLE_RATE)

        if (
            callbacks.vc.crossfade_frame != crossfade_frame or
            callbacks.vc.extra_frame != extra_frame
        ):
            del (
                callbacks.vc.fade_in_window,
                callbacks.vc.fade_out_window,
                callbacks.vc.sola_buffer
            )

            callbacks.vc.vc_model.realloc(
                callbacks.vc.block_frame,
                callbacks.vc.extra_frame,
                callbacks.vc.crossfade_frame,
                callbacks.vc.sola_search_frame,
            )
            callbacks.vc.generate_strength()

        callbacks.vc.vc_model.input_sensitivity = 10 ** (callbacks_kwargs.get("silent_threshold", -90) / 20)

        vad_enabled = callbacks_kwargs.get("vad_enabled", True)
        sensitivity_mode = callbacks_kwargs.get("vad_sensitivity", 3)
        vad_frame_ms = callbacks_kwargs.get("vad_frame_ms", 30)

        if vad_enabled is False:
            callbacks.vc.vc_model.vad = None
        elif vad_enabled and callbacks.vc.vc_model.vad is None:
            from main.inference.realtime.vad_utils import VADProcessor

            callbacks.vc.vc_model.vad = VADProcessor(
                sensitivity_mode=sensitivity_mode,
                sample_rate=callbacks.vc.vc_model.sample_rate,
                frame_duration_ms=vad_frame_ms
            )

        if callbacks.vc.vc_model.vad is not None:
            callbacks.vc.vc_model.vad.vad.set_mode(sensitivity_mode)
            callbacks.vc.vc_model.vad.frame_length = int(callbacks.vc.vc_model.sample_rate * (vad_frame_ms / 1000.0))

        clean_audio = callbacks_kwargs.get("clean_audio", False)
        clean_strength = callbacks_kwargs.get("clean_strength", 0.5)

        if clean_audio is False:
            callbacks.vc.vc_model.tg = None
        elif clean_audio and callbacks.vc.vc_model.tg is None:
            from main.tools.noisereduce import TorchGate

            callbacks.vc.vc_model.tg = (
                TorchGate(
                    callbacks.vc.vc_model.pipeline.tgt_sr,
                    prop_decrease=clean_strength,
                ).to(config.device)
            )

        if callbacks.vc.vc_model.tg is not None:
            callbacks.vc.vc_model.tg.prop_decrease = clean_strength

        post_process = callbacks_kwargs.get("post_process", False)
        kwargs = callbacks_kwargs.get("kwargs", {})

        if post_process is False:
            callbacks.vc.vc_model.board = None
            callbacks.vc.vc_model.kwargs = None
        elif post_process and callbacks.vc.vc_model.kwargs != kwargs:
            new_board = callbacks.vc.vc_model.setup_pedalboard(**kwargs)
            callbacks.vc.vc_model.board = new_board
            callbacks.vc.vc_model.kwargs = kwargs.copy()

        callbacks.audio.f0_up_key = callbacks_kwargs.get("f0_up_key", 0)
        callbacks.audio.index_rate = callbacks_kwargs.get("index_rate", 0.75)
        callbacks.audio.protect = callbacks_kwargs.get("protect", 0.5)
        callbacks.audio.rms_mix_rate = callbacks_kwargs.get("rms_mix_rate", 1)

        callbacks.audio.f0_autotune = callbacks_kwargs.get("f0_autotune", False)
        callbacks.audio.f0_autotune_strength = callbacks_kwargs.get("f0_autotune_strength", 1.0)
        callbacks.audio.proposal_pitch = callbacks_kwargs.get("proposal_pitch", False)
        callbacks.audio.proposal_pitch_threshold = callbacks_kwargs.get("proposal_pitch_threshold", 155.0)

        callbacks.audio.input_audio_gain = callbacks_kwargs.get("input_audio_gain", 1.0)
        callbacks.audio.output_audio_gain = callbacks_kwargs.get("output_audio_gain", 1.0)
        callbacks.audio.monitor_audio_gain = callbacks_kwargs.get("monitor_audio_gain", 1.0)

        model_pth = callbacks_kwargs.get("model_path", callbacks.vc.vc_model.model_path)
        model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

        if model_pth and callbacks.vc.vc_model.model_path != model_pth:
            callbacks.vc.vc_model.model_path = model_pth
            callbacks.vc.vc_model.pipeline.inference.get_synthesizer(model_pth)

            callbacks.vc.vc_model.pipeline.version = callbacks.vc.vc_model.pipeline.inference.version
            callbacks.vc.vc_model.pipeline.energy = callbacks.vc.vc_model.pipeline.inference.energy

            if callbacks.vc.vc_model.pipeline.inference.energy:
                from main.inference.extracting.rms import RMSEnergyExtractor

                rms = RMSEnergyExtractor(
                    frame_length=2048, 
                    hop_length=160, 
                    center=True, 
                    pad_mode="reflect"
                ).to(config.device).eval()

                callbacks.vc.vc_model.pipeline.rms = rms
            else:
                callbacks.vc.vc_model.pipeline.rms = None

        sid = callbacks_kwargs.get("sid", callbacks.vc.vc_model.pipeline.sid)
        if callbacks.vc.vc_model.pipeline.sid != sid:
            import torch
            callbacks.vc.vc_model.pipeline.torch_sid = torch.tensor(
                [sid], device=callbacks.vc.vc_model.pipeline.device, dtype=torch.int64
            )

        index_path = callbacks_kwargs.get("index_path", None)
        if index_path:
            if callbacks.vc.vc_model.index_path != index_path:
                from main.library.utils import load_faiss_index

                index, index_reconstruct = load_faiss_index(
                    index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
                )

                callbacks.vc.vc_model.pipeline.index = index
                callbacks.vc.vc_model.pipeline.big_npy = index_reconstruct
                callbacks.vc.vc_model.index_path = index_path
        else:
            callbacks.vc.vc_model.pipeline.index = None
            callbacks.vc.vc_model.pipeline.big_npy = None
            callbacks.vc.vc_model.index_path = None

        f0_method = callbacks_kwargs.get("f0_method", callbacks.vc.vc_model.pipeline.f0_method)
        predictor_onnx = callbacks_kwargs.get("predictor_onnx", callbacks.vc.vc_model.pipeline.predictor.predictor_onnx)
        embedders = callbacks_kwargs.get("embedder_model", callbacks.vc.vc_model.embedder_model)
        embedders_mode = callbacks_kwargs.get("embedders_mode", callbacks.vc.vc_model.embedders_mode)
        custom_embedders = callbacks_kwargs.get("embedder_model_custom", None)
        embedder_model = (embedders if embedders != "custom" else custom_embedders)

        from main.library.utils import check_assets
        check_assets(f0_method, embedders, predictor_onnx, embedders_mode)

        if (
            callbacks.vc.vc_model.pipeline.f0_method != f0_method or
            callbacks.vc.vc_model.pipeline.predictor.predictor_onnx != predictor_onnx
        ):
            old_predictor = callbacks.vc.vc_model.pipeline.predictor
            del old_predictor

            from main.library.predictors.Generator import Generator

            predictor = Generator(
                sample_rate=callbacks_kwargs.get("sample_rate", callbacks.vc.vc_model.sample_rate), 
                hop_length=callbacks_kwargs.get("f0_method", callbacks.vc.vc_model.pipeline.predictor.hop_length), 
                f0_min=50.0, 
                f0_max=1100.0, 
                alpha=0.5, 
                is_half=config.is_half, 
                device=config.device, 
                predictor_onnx=predictor_onnx, 
                delete_predictor_onnx=False
            )

            callbacks.vc.vc_model.pipeline.predictor = predictor
            callbacks.vc.vc_model.pipeline.f0_method = f0_method
            callbacks.vc.vc_model.pipeline.predictor.predictor_onnx = predictor_onnx

        if embedder_model:
            if (
                callbacks.vc.vc_model.embedder_model != embedder_model or
                callbacks.vc.vc_model.embedders_mode != embedders_mode
            ):
                old_embedder = callbacks.vc.vc_model.pipeline.embedder
                del old_embedder

                import torch
                from main.library.utils import load_embedders_model

                embedder = load_embedders_model(
                    embedder_model, 
                    embedders_mode=embedders_mode
                )

                if isinstance(embedder, torch.nn.Module): 
                    dtype = torch.float16 if config.is_half else torch.float32
                    embedder = embedder.to(config.device).to(dtype).eval()

                callbacks.vc.vc_model.pipeline.embedder = embedder
                callbacks.vc.vc_model.embedder_model = embedder_model
                callbacks.vc.vc_model.embedders_mode = embedders_mode

def change_config(value, key, if_kwargs=False):
    global callbacks_kwargs

    if running and audio_manager is not None and callbacks is not None:
        if if_kwargs:
            callbacks_kwargs["kwargs"][key] = value
        else:
            callbacks_kwargs[key] = value

        change_callbacks_config()

def realtime_stop():
    global running, callbacks, audio_manager

    if running and audio_manager is not None and callbacks is not None:
        gr_info(translations["stop_realtime"])

        audio_manager.stop()
        running = False

        if hasattr(callbacks, "latency"): del callbacks.latency
        del audio_manager, callbacks

        audio_manager = callbacks = None
        gr_info(translations["realtime_has_stop"])

        from main.library.utils import clear_gpu_cache
        clear_gpu_cache()

        return (
            translations["realtime_has_stop"], 
            interactive_true, 
            interactive_false
        )
    else:
        gr_warning(translations["realtime_not_found"])

        return (
            translations["realtime_not_found"], 
            interactive_true, 
            interactive_false
        )

def soundfile_record_audio(
    record_button,
    record_audio_path = None,
    export_format = "wav"
):
    global callbacks

    if running and audio_manager is not None and callbacks is not None:
        if record_button == translations["start_record"]:
            gr_info(translations["starting_record"])

            if not record_audio_path:
                record_audio_path = os.path.join("audios", "record_audio.wav")

            callbacks.vc.record_audio = True
            callbacks.vc.record_audio_path = record_audio_path
            callbacks.vc.export_format = export_format
            callbacks.vc.setup_soundfile_record()

            return translations["stop_record"], None
        else:
            gr_info(translations["stopping_record"])

            callbacks.vc.record_audio = False
            callbacks.vc.record_audio_path = None
            callbacks.vc.soundfile = None

            return translations["start_record"], record_audio_path

    gr_warning(translations["realtime_not_found"])
    return translations["start_record"], None