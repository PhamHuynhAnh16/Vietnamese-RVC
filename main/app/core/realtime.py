import os
import sys
import time
import queue

import multiprocessing as mp

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import translations, configs, config, logger

running, realtime_process, ui_queue, config_queue = False, None, None, None

PIPELINE_SAMPLE_RATE = 16000

interactive_true = {"interactive": True, "__type__": "update"}
interactive_false = {"interactive": False, "__type__": "update"}
callbacks_kwargs = {}

js_code = """
window._activeStream = null;
window._audioCtx = null;
window._workletNode = null;
window._playbackNode = null;
window._ws = null;
window.OutputAudioRoute = null;
window.MonitorAudioRoute = null;
window.lastSend = 0;
window.responseMs = 0;

// Function to display status
function setStatus(msg, use_alert = true) {
    const realtimeStatus = document.querySelector("#realtime-status-info h2.output-class"); // find status text box
    if (use_alert) alert(msg); // Use alert instead of gr.Info

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
    const dest = audioCtx.createMediaStreamDestination(); // Create a MediaStreamDestination Node
    const gainNode = audioCtx.createGain(); // Create a GainNode (Volume Control Node)
    gainNode.gain.value = gainValue; // Sets the initial gain (volume).

    // Connect the Audio Nodes
    playbackNode.connect(gainNode);
    gainNode.connect(dest);

    // Create and Configure the audio Element
    const el = document.createElement('audio');
    el.autoplay = true;
    el.srcObject = dest.stream;
    el.style.display = 'none';
    document.body.appendChild(el);

    if (el.setSinkId) el.setSinkId(sinkId).catch(err => console.error(err));
    return { dest, gainNode, el }; // Returns the objects (destination node, gain node, and audio element)
}

const inputWorkletSource = `
class InputProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferCapacity = 48000; // Stores up to 1 second of audio at 48kHz
        this.buffer = new Float32Array(this.bufferCapacity);
        this.writePointer = 0;
        this.readPointer = 0;
        this.availableSamples = 0;
        this.block_frame = 960; // Standard layout processing slice size
        // Handle dynamic frame configuration updates from the main main thread
        this.port.onmessage = (e) => {
            if (e.data && e.data.block_frame) this.block_frame = e.data.block_frame;
        };
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true; // Keep worklet alive if no stream is active

        const frame = input[0];
        const frameSize = frame.length;

        // Push new incoming samples into the internal circular ring buffer
        if (this.availableSamples + frameSize <= this.bufferCapacity) {
            for (let i = 0; i < frameSize; i++) {
                this.buffer[this.writePointer] = frame[i];
                this.writePointer = (this.writePointer + 1) % this.bufferCapacity;
            }

            this.availableSamples += frameSize;
        }

        // Slice accumulated audio into uniform blocks and dispatch them to the main thread
        while (this.availableSamples >= this.block_frame) {
            const chunk = new Float32Array(this.block_frame);

            for (let i = 0; i < this.block_frame; i++) {
                chunk[i] = this.buffer[this.readPointer];
                this.readPointer = (this.readPointer + 1) % this.bufferCapacity;
            }

            this.availableSamples -= this.block_frame;
            this.port.postMessage({ chunk }); 
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
        const bufferSize = options.processorOptions && options.processorOptions.bufferSize ? options.processorOptions.bufferSize : 98304;
        this.buffer = new Float32Array(bufferSize);
        this.bufferCapacity = bufferSize;
        this.writePointer = 0;
        this.readPointer = 0;
        this.availableSamples = 0;

        // Listen for returned server-processed audio chunks and load them into the playback ring buffer
        this.port.onmessage = (e) => {
            if (e.data && e.data.chunk) {
                const chunk = new Float32Array(e.data.chunk);
                const chunkSize = chunk.length;

                // Guard against ring buffer overflows (drop chunk if filled)
                if (this.availableSamples + chunkSize > this.bufferCapacity) return;

                // Handle standard inline write vs circular wrapping wrap-around logic
                if (this.writePointer + chunkSize <= this.bufferCapacity) {
                    this.buffer.set(chunk, this.writePointer);
                } else {
                    const firstPart = this.bufferCapacity - this.writePointer;
                    this.buffer.set(chunk.subarray(0, firstPart), this.writePointer);
                    this.buffer.set(chunk.subarray(firstPart), 0);
                }

                this.writePointer = (this.writePointer + chunkSize) % this.bufferCapacity;
                this.availableSamples += chunkSize;
            }
        };
    }

    process(inputs, outputs) {
        const output = outputs[0];
        if (!output || !output[0]) return true;

        const frame = output[0];
        const frameSize = frame.length;

        // Populate the hardware output frame if there are enough accumulated samples
        if (this.availableSamples >= frameSize) {
            if (this.readPointer + frameSize <= this.bufferCapacity) {
                frame.set(this.buffer.subarray(this.readPointer, this.readPointer + frameSize));
            } else {
                const firstPart = this.bufferCapacity - this.readPointer;
                frame.set(this.buffer.subarray(this.readPointer, this.bufferCapacity), 0);
                frame.set(this.buffer.subarray(0, frameSize - firstPart), firstPart);
            }

            this.readPointer = (this.readPointer + frameSize) % this.bufferCapacity;
            this.availableSamples -= frameSize;
        } else {
            // Underflow protection: fill buffer with silence (zeros) to prevent harsh digital crackling
            frame.fill(0);
        }

        // Duplicate audio channel configurations for stereo topologies if supported
        if (output.length > 1) output[1].set(output[0]);
        return true;
    }
}
registerProcessor('playback-processor', PlaybackProcessor);
`;

window.getAudioDevices = async function() {
    if (!navigator.mediaDevices) {
        // If somehow the browser does not support.
        setStatus("__MEDIA_DEVICES__");
        return {"inputs": {}, "outputs": {}};
    }

    try {
        // Request audio permissions to the browser.
        await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        console.error(err);
        setStatus("__MIC_INACCESSIBLE__")

        return {"inputs": {}, "outputs": {}};
    }

    // Read the audio devices available on the browser and filter out the devices.
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = {};
    const outputs = {};
    
    for (const device of devices) {
        if (device.kind === "audioinput") {
        inputs[device.label + ` (${device.deviceId.slice(0, 10)})`] = device.deviceId;
        } else if (device.kind === "audiooutput") {
        outputs[device.label + ` (${device.deviceId.slice(0, 10)})`] = device.deviceId;
        }
    }

    // Returns the audio devices
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
    noise_scale,
    embedders_mix,
    embedders_mix_layers,
    embedders_mix_ratio,
    use_phase_vocoder,
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
    nprobe
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
                deviceId: { exact: input_audio_device }, // audio: input_audio_device ? { deviceId: { exact: input_audio_device } } : true
                channelCount: { exact: 1 },
                sampleRate: { exact: SampleRate },
                latency: { ideal: 0 },
                // disable all browser processing (You can make it optional)
                echoCancellation: !exclusive_mode,
                noiseSuppression: !exclusive_mode,
                autoGainControl: !exclusive_mode
            }
        });

        let latencyHint = "playback";
        if (exclusive_mode) latencyHint = "interactive";

        window._activeStream = stream;
        window._audioCtx = new AudioContext({ sampleRate: SampleRate, latencyHint: latencyHint });

        // Load processing modules.
        await addModuleFromString(window._audioCtx, inputWorkletSource);
        await addModuleFromString(window._audioCtx, playbackWorkletSource);
        // await window._audioCtx.audioWorklet.addModule('/input_processor.js');
        // await window._audioCtx.audioWorklet.addModule('/playback_processor.js');

        // Initialize audio web parts
        const src = window._audioCtx.createMediaStreamSource(stream);
        const inputNode = new AudioWorkletNode(window._audioCtx, 'input-processor');
        const playbackNode = new AudioWorkletNode(window._audioCtx, 'playback-processor', {
            processorOptions: {
                bufferSize: block_frame * 2 // Double or more is recommended to avoid loss of sound.
            }
        });

        inputNode.port.postMessage({ block_frame: block_frame });
        src.connect(inputNode);

        // Create audio and monitor output
        window.OutputAudioRoute = createOutputRoute(window._audioCtx, playbackNode, output_audio_device, output_audio_gain / 100);
        if (monitor && monitor_output_device) window.MonitorAudioRoute = createOutputRoute(window._audioCtx, playbackNode, monitor_output_device, monitor_audio_gain / 100);
        
        // Configure path and websocket
        const protocol = (location.protocol === "https:") ? "wss:" : "ws:";
        const wsUrl = protocol + '//' + location.hostname + `:${location.port}` + '/api/ws-audio';
        const ws = new WebSocket(wsUrl);

        // Set new values ​​of buttons to avoid users initiating multiple realtime threads
        ButtonState.start_button = false;
        ButtonState.stop_button = true;

        ws.binaryType = "arraybuffer";
        window._ws = ws;

        ws.onopen = () => {
            console.log("__WS_CONNECTED__")

            // send all parameters to websocket realtime
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
                    noise_scale: noise_scale,
                    embedders_mix: embedders_mix,
                    embedders_mix_layers: embedders_mix_layers,
                    embedders_mix_ratio: embedders_mix_ratio,
                    use_phase_vocoder: use_phase_vocoder,
                    nprobe: nprobe,
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
            // send audio from node to websocket for realtime
            const chunk = e.data && e.data.chunk;

            if (!chunk) return;
            if (ws.readyState === WebSocket.OPEN) {
                window.lastSend = performance.now();
                ws.send(chunk);
            }
        };

        ws.onmessage = (ev) => {
            // Read the string values ​​sent back from the websocket
            if (typeof ev.data === 'string') {
                const msg = JSON.parse(ev.data);

                // Show latency information in the status bar of the interface
                if (msg.type === 'latency') setStatus(`__LATENCY__: ${msg.value.toFixed(2)} ms | __VOLUME__: ${msg.volume.toFixed(2)} dB | __RESPONSE__: ${window.responseMs.toFixed(2)} ms`, use_alert=false)
                if (msg.type === 'warnings') {
                    setStatus(msg.value);
                    StopAudioStream();
                }

                return;
            }

            // Send audio to the playback node to the audio device
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
        console.error("__ERROR__" + err);
        alert("__ERROR__" + err.message);
        // stop realtime when error
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

        document.querySelectorAll("audio").forEach((a) => {
        a.pause();
        a.srcObject = null;
        a.remove();
        });
        setStatus("__REALTIME_HAS_STOP__", false);

        return {"start_button": true, "stop_button": false};
    } catch (e) {
        setStatus(`__ERROR__${e}`);

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
    "__PROVIDE_MODEL__", translations["provide_model"]
).replace(
    "__VOLUME__", translations["volume"]
).replace(
    "__RESPONSE__", translations["response"]
)

def realtime_worker(queue_out, config_queue, callbacks_kwargs, audio_manager_kwargs):
    """
    Worker process target that handles the background real-time audio processing loop.
    
    This function initializes audio callbacks, configures the audio manager, and handles
    runtime commands (START, STOP, UPDATE_CONFIG) dispatched from the main UI process.

    Args:
        queue_out (queue.Queue): Inter-process queue to send status/logs back to the UI.
        config_queue (queue.Queue): Inter-process queue to receive dynamic configuration commands.
        callbacks_kwargs (Dict[str, Any]): Initialization arguments for the AudioCallbacks engine.
        audio_manager_kwargs (Dict[str, Any]): Initialization arguments for hardware audio streams.
    """

    callbacks, audio_manager = None, None

    try:
        # Lazy import to avoid loading heavy deep learning/audio frameworks prematurely in the main thread
        from main.inference.realtime.callbacks import AudioCallbacks

        # Initialize the audio pipeline processing core
        callbacks = AudioCallbacks(**callbacks_kwargs)
        audio_manager = callbacks.audio

        def audio_manager_start(kwargs):
            """Helper function to map configuration dictionary to hardware audio manager start parameters."""

            audio_manager.start(
                input_device_id=kwargs["input_device_id"], 
                output_device_id=kwargs["output_device_id"], 
                output_monitor_id=kwargs["output_monitor_id"], 
                exclusive_mode=kwargs["exclusive_mode"], 
                asio_input_channel=kwargs["input_asio_channels"], 
                asio_output_channel=kwargs["output_asio_channels"], 
                asio_output_monitor_channel=kwargs["monitor_asio_channels"],
                read_chunk_size=kwargs["chunk_size"], 
                input_audio_sample_rate=kwargs["input_audio_sample_rate"], 
                output_audio_sample_rate=kwargs["output_audio_sample_rate"], 
                output_monitor_sample_rate=kwargs["monitor_audio_sample_rate"]
            )

        # Boot up hardware audio streaming devices on process startup
        audio_manager_start(audio_manager_kwargs)
        queue_out.put(("INFO", translations["realtime_is_ready"]))

        # Infinite loop handling streaming telemetry and dynamic payload commands
        while 1:
            time.sleep(0.1)
            # Periodically stream performance stats (latency and input amplitude volume metrics) to the UI
            if hasattr(callbacks, "audio") and hasattr(callbacks.audio, "performance") and hasattr(callbacks.audio, "volume"):
                queue_out.put(("STATUS", f"{translations['latency']}: {callbacks.audio.performance:.2f} ms | {translations['volume']}: {callbacks.audio.volume:.2f} dB"))

            try:
                # Check for asynchronous runtime configurations or lifestyle control commands
                cmd, callbacks_kwargs = config_queue.get_nowait()

                if cmd == "START":
                    callbacks_kwargs, audio_manager_kwargs = callbacks_kwargs
                    # Lazy instantiation if the process was kept alive but components were previously torn down
                    if callbacks is None:
                        callbacks = AudioCallbacks(**callbacks_kwargs)
                        audio_manager = callbacks.audio
                        audio_manager_start(audio_manager_kwargs)

                        queue_out.put(("INFO", translations["realtime_is_ready"]))
                elif cmd == "STOP":
                    if callbacks is not None:
                        # Safely stop physical low-level hardware IO stream interfaces
                        audio_manager.stop()

                        # Clean up telemetry metrics references
                        if hasattr(callbacks.audio, "performance"): del callbacks.audio.performance
                        if hasattr(callbacks.audio, "volume"): del callbacks.audio.volume

                        # Explicitly break reference trees to release underlying neural pipelines & weights
                        del callbacks.vc.vc_model.pipeline, callbacks.vc.vc_model
                        del audio_manager, callbacks

                        # Reset states completely
                        audio_manager = callbacks = None

                        # Garbage collect and flush VRAM to prevent memory leaks in shared application state
                        import gc
                        from main.library.utils import clear_gpu_cache

                        gc.collect()
                        clear_gpu_cache()

                        queue_out.put(("STOP", ""))
                elif cmd == "UPDATE_CONFIG":
                    if callbacks is None: continue

                    # Calculate overlapping structure size windows mapped against the target system sample rate
                    crossfade_frame = int(callbacks_kwargs.get("cross_fade_overlap_size", 0.1) * callbacks.input_sample_rate)
                    extra_frame = int(callbacks_kwargs.get("extra_convert_size", 0.5) * callbacks.input_sample_rate)

                    # Trigger memory reallocation on the underlying core if latency window sizes are altered
                    if (
                        callbacks.vc.crossfade_frame != crossfade_frame or
                        callbacks.vc.extra_frame != extra_frame
                    ):
                        # Force-release old arrays to prevent internal calculation leaks before reconfiguration
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

                    # Convert the dB floor threshold configuration parameter to standard linear amplitude values
                    callbacks.vc.vc_model.input_sensitivity = 10 ** (callbacks_kwargs.get("silent_threshold", -90) / 20)

                    # Voice Activity Detection (VAD) Engine Configuration Block
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

                    # Audio Clean Gate (TorchGate Noise Reduction) Block
                    clean_audio = callbacks_kwargs.get("clean_audio", False)
                    clean_strength = callbacks_kwargs.get("clean_strength", 0.5)

                    if clean_audio is False:
                        callbacks.vc.vc_model.tg = None
                    elif clean_audio and callbacks.vc.vc_model.tg is None:
                        from main.library.audio.noisereduce import TorchGate

                        callbacks.vc.vc_model.tg = (
                            TorchGate(
                                callbacks.vc.vc_model.pipeline.tgt_sr,
                                prop_decrease=clean_strength,
                            ).to(config.device)
                        )

                    if callbacks.vc.vc_model.tg is not None:
                        callbacks.vc.vc_model.tg.prop_decrease = clean_strength

                    # Pedalboard Effects Post-Processing Block
                    post_process = callbacks_kwargs.get("post_process", False)
                    kwargs = callbacks_kwargs.get("kwargs", {})

                    if post_process is False:
                        callbacks.vc.vc_model.board = None
                        callbacks.vc.vc_model.kwargs = None
                    elif post_process and callbacks.vc.vc_model.kwargs != kwargs:
                        new_board = callbacks.vc.vc_model.setup_pedalboard(**kwargs)
                        callbacks.vc.vc_model.board = new_board
                        callbacks.vc.vc_model.kwargs = kwargs.copy()

                    # Direct primitive assignment mapping for real-time pipeline hyper-parameters
                    callbacks.audio.f0_up_key = callbacks_kwargs.get("f0_up_key", 0)
                    callbacks.audio.index_rate = callbacks_kwargs.get("index_rate", 0.75)
                    callbacks.audio.protect = callbacks_kwargs.get("protect", 0.5)
                    callbacks.audio.rms_mix_rate = callbacks_kwargs.get("rms_mix_rate", 1)

                    # Autotune and pitch suggestion modules adjustments
                    callbacks.audio.f0_autotune = callbacks_kwargs.get("f0_autotune", False)
                    callbacks.audio.f0_autotune_strength = callbacks_kwargs.get("f0_autotune_strength", 1.0)
                    callbacks.audio.proposal_pitch = callbacks_kwargs.get("proposal_pitch", False)
                    callbacks.audio.proposal_pitch_threshold = callbacks_kwargs.get("proposal_pitch_threshold", 155.0)

                    # Hardware mixing pipeline digital gain structures
                    callbacks.audio.input_audio_gain = callbacks_kwargs.get("input_audio_gain", 1.0)
                    callbacks.audio.output_audio_gain = callbacks_kwargs.get("output_audio_gain", 1.0)
                    callbacks.audio.monitor_audio_gain = callbacks_kwargs.get("monitor_audio_gain", 1.0)

                    # Contextual embedding multi-layer blending structure configs
                    callbacks.audio.embedders_mix = callbacks_kwargs.get("embedders_mix", False)
                    callbacks.audio.embedders_mix_layers = callbacks_kwargs.get("embedders_mix_layers", 9)
                    callbacks.audio.embedders_mix_ratio = callbacks_kwargs.get("embedders_mix_ratio", 0.5)

                    callbacks.audio.use_phase_vocoder = callbacks_kwargs.get("use_phase_vocoder", True)

                    # Model Weight Reload / Architecture Swap Block
                    noise_scale = callbacks_kwargs.get("noise_scale", 0.35)
                    model_pth = callbacks_kwargs.get("model_path", callbacks.vc.vc_model.model_path)
                    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

                    if model_pth and callbacks.vc.vc_model.model_path != model_pth:
                        import torch
                        import torchaudio.transforms as tat

                        callbacks.vc.vc_model.model_path = model_pth
                        callbacks.vc.vc_model.pipeline.vc.setup(model_pth, noise_scale)
                        callbacks.vc.vc_model.pipeline.use_f0 = callbacks.vc.vc_model.pipeline.vc.use_f0
                        callbacks.vc.vc_model.pipeline.tgt_sr = callbacks.vc.vc_model.pipeline.vc.tgt_sr
                        callbacks.vc.vc_model.pipeline.version = callbacks.vc.vc_model.pipeline.vc.version

                        # Re-initialize the resampling layer since the target sample rate might have changed
                        callbacks.vc.vc_model.resample_out = tat.Resample(
                            orig_freq=callbacks.vc.vc_model.pipeline.tgt_sr,
                            new_freq=callbacks.vc.vc_model.output_sample_rate,
                            dtype=torch.float32
                        ).to(config.device)

                        if clean_audio:
                            from main.library.audio.noisereduce import TorchGate

                            callbacks.vc.vc_model.tg = (
                                TorchGate(
                                    callbacks.vc.vc_model.pipeline.tgt_sr,
                                    prop_decrease=clean_strength,
                                ).to(config.device)
                            )

                    # Speaker Identity ID Configuration
                    sid = callbacks_kwargs.get("sid", callbacks.vc.vc_model.pipeline.sid)
                    if callbacks.vc.vc_model.pipeline.sid != sid:
                        import torch
                        callbacks.vc.vc_model.pipeline.torch_sid = torch.tensor(
                            [sid], device=callbacks.vc.vc_model.pipeline.device, dtype=torch.int64
                        )

                    # FAISS Index Tracking Path Search Block
                    index_path = callbacks_kwargs.get("index_path", None)
                    if index_path:
                        if callbacks.vc.vc_model.index_path != index_path:
                            from main.library.utils import load_faiss_index

                            nprobe = callbacks_kwargs.get("nprobe", 1)
                            # Sanitize paths to handle literal quote configurations securely
                            index, index_reconstruct = load_faiss_index(
                                index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"),
                                nprobe=nprobe
                            )

                            callbacks.vc.vc_model.pipeline.index = index
                            if callbacks.vc.vc_model.pipeline.index.index is not None: callbacks.vc.vc_model.pipeline.index.index.nprobe = nprobe
                            callbacks.vc.vc_model.pipeline.big_tsr = index_reconstruct
                            callbacks.vc.vc_model.index_path = index_path
                    else:
                        callbacks.vc.vc_model.pipeline.index = None
                        callbacks.vc.vc_model.pipeline.big_tsr = None
                        callbacks.vc.vc_model.index_path = None

                    # Pitch Predictor and Feature Embedder Extraction Block
                    f0_method = callbacks_kwargs.get("f0_method", callbacks.vc.vc_model.pipeline.f0_method)
                    predictor_onnx = callbacks_kwargs.get("predictor_onnx", callbacks.vc.vc_model.pipeline.predictor.predictor_onnx)
                    embedders = callbacks_kwargs.get("embedder_model", callbacks.vc.vc_model.embedder_model)
                    embedders_mode = callbacks_kwargs.get("embedders_mode", callbacks.vc.vc_model.embedders_mode)
                    custom_embedders = callbacks_kwargs.get("embedder_model_custom", None)
                    embedder_model = (embedders if embedders != "custom" else custom_embedders)

                    # Ensure files and parameters match required asset types
                    from main.library.utils import check_assets
                    check_assets(f0_method, embedders, predictor_onnx, embedders_mode)

                    # Hot-swap the pitch predictor backend structures if options are updated
                    if (
                        callbacks.vc.vc_model.pipeline.vc.use_f0 and (
                            callbacks.vc.vc_model.pipeline.f0_method != f0_method or
                            callbacks.vc.vc_model.pipeline.predictor.predictor_onnx != predictor_onnx
                        )
                    ):
                        old_predictor = callbacks.vc.vc_model.pipeline.predictor
                        del old_predictor.pw, old_predictor.fcpe, old_predictor.djcm, old_predictor.penn, old_predictor.pesto, old_predictor.swift, old_predictor.rmvpe, old_predictor.crepe, old_predictor.mangio_penn, old_predictor.mangio_crepe
                        del old_predictor

                        callbacks.vc.vc_model.pipeline.predictor = callbacks.vc.vc_model.pipeline.setup_predictor(PIPELINE_SAMPLE_RATE, callbacks_kwargs.get("hop_length", callbacks.vc.vc_model.pipeline.predictor.hop_length), predictor_onnx)
                        callbacks.vc.vc_model.pipeline.f0_method = f0_method
                        callbacks.vc.vc_model.pipeline.predictor.predictor_onnx = predictor_onnx

                    # Hot-swap Content Vec or Hubert feature extractors if modified
                    if embedder_model:
                        if (
                            callbacks.vc.vc_model.embedder_model != embedder_model or
                            callbacks.vc.vc_model.embedders_mode != embedders_mode
                        ):
                            old_embedder = callbacks.vc.vc_model.pipeline.embedder
                            del old_embedder

                            callbacks.vc.vc_model.pipeline.embedder = callbacks.vc.vc_model.pipeline.setup_embedder(embedder_model, embedders_mode)
                            callbacks.vc.vc_model.embedder_model = embedder_model
                            callbacks.vc.vc_model.embedders_mode = embedders_mode

                    if (
                        callbacks.vc.vc_model.pipeline.vc.architecture == "SVC" and
                        callbacks.vc.vc_model.pipeline.vc.net_g.noise_scale != noise_scale
                    ): 
                        callbacks.vc.vc_model.pipeline.vc.net_g.noise_scale = noise_scale
                    
                    # Direct Disk Sound Recording Pipeline Controller
                    record_audio = callbacks_kwargs.get("record_audio", False)
                    if callbacks.vc.record_audio != record_audio:
                        callbacks.vc.record_audio = record_audio
                        callbacks.vc.record_audio_path = callbacks_kwargs.get("record_audio_path", None)
                        callbacks.vc.export_format = callbacks_kwargs.get("export_format", "wav")
                        callbacks.vc.setup_soundfile_record()
            except queue.Empty:
                pass
    except Exception as e:
        import traceback

        logger.error(e)
        logger.debug(traceback.format_exc())

        queue_out.put(("ERROR", translations["realtime_has_stop_with_error"]))

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
    input_audio_sample_rate,
    output_audio_sample_rate,
    monitor_audio_sample_rate,
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
    noise_scale,
    embedders_mix,
    embedders_mix_layers,
    embedders_mix_ratio,
    use_phase_vocoder,
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
    nprobe = 1
):
    """
    Prepares configurations, sanitizes input components, and spawns the real-time worker process.

    This acts as a UI generator that sends system state changes up to the interface layer,
    instantiates pipeline dictionary blocks, and coordinates cross-process tracking.
    """

    global running, realtime_process, ui_queue, config_queue, callbacks_kwargs
    running = True

    gr_info(translations["start_realtime"])

    # Configure interface visual element states
    _interactive_false = interactive_false.copy()
    _interactive_false["value"] = translations["stop_realtime_button"]

    yield (
        translations["start_realtime"], 
        interactive_false, 
        _interactive_false
    )

    # Audio IO hardware validation guardrails
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

    # Check and sanitize voice cloning target neural model path locations
    model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth
    embedder_model = (embedders if embedders != "custom" else custom_embedders)

    if (
        not model_pth or 
        not os.path.exists(model_pth) or 
        os.path.isdir(model_pth) or 
        not model_pth.endswith((".pth", ".onnx"))
    ):
        gr_warning(translations["provide_model"])

        yield (
            translations["provide_model"], 
            interactive_true, 
            interactive_false
        )
        return
    
    # Audio host API state management context switching for ASIO support
    input_is_asio = "ASIO" in input_audio_device
    output_is_asio = "ASIO" in output_audio_device
    monitor_is_asio = "ASIO" in monitor_output_device

    if input_is_asio or output_is_asio or monitor_is_asio:
        import main.library.audio.sounddevice as sd

        sd.terminate()
        sd.initialize()
    
    import main.app.core.ui as ui

    # Map selected hardware device strings back to direct OS index values
    input_device_id = ui.input_channels_map[input_audio_device][0]
    output_device_id = ui.output_channels_map[output_audio_device][0]
    output_monitor_id = ui.output_channels_map[monitor_output_device][0] if monitor else None
    # Rescale percentages from sliders down to raw linear gain metrics
    input_audio_gain /= 100.0
    output_audio_gain /= 100.0
    monitor_audio_gain /= 100.0

    # Calculate actual process block sizes based on window ms metrics against incoming sample rates
    chunk_size = int(chunk_size * input_audio_sample_rate / 1000 / 128)

    # Master structural pack for Voice Changer (VC) Callback engine configurations
    callbacks_kwargs = {
        "pass_through": False, 
        "read_chunk_size": chunk_size, 
        "cross_fade_overlap_size": cross_fade_overlap_size, 
        "input_sample_rate": input_audio_sample_rate, 
        "output_sample_rate": input_audio_sample_rate, 
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
        "noise_scale": noise_scale,
        "nprobe": nprobe,
        "embedders_mix": embedders_mix,
        "embedders_mix_layers": embedders_mix_layers,
        "embedders_mix_ratio": embedders_mix_ratio,
        "use_phase_vocoder": use_phase_vocoder,
        "record_audio": False,
        "record_audio_path": "",
        "export_format": "wav",
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

    # Master structural pack for the physical low-level host Soundcard device configuration
    audio_manager_kwargs = {
        "input_device_id": input_device_id,
        "output_device_id": output_device_id,
        "output_monitor_id": output_monitor_id,
        "exclusive_mode": exclusive_mode,
        "input_asio_channels": input_asio_channels,
        "output_asio_channels": output_asio_channels,
        "monitor_asio_channels": monitor_asio_channels,
        "chunk_size": chunk_size,
        "input_audio_sample_rate": input_audio_sample_rate,
        "output_audio_sample_rate": output_audio_sample_rate,
        "monitor_audio_sample_rate": monitor_audio_sample_rate
    }

    # Force using "spawn" multi-processing context to isolate variables securely across platforms (Windows/Linux)
    ctx = mp.get_context("spawn")
    if ui_queue is None or config_queue is None:
        ui_queue = ctx.Queue()
        config_queue = ctx.Queue()

    # Flush stale artifacts out of the UI messaging queue
    while not ui_queue.empty():
        try: 
            ui_queue.get_nowait()
        except queue.Empty: 
            break
    
    # Initialize background process structures or hot-reload via state queues if already active
    if realtime_process is None:
        realtime_process = ctx.Process(
            target=realtime_worker, 
            args=(ui_queue, config_queue, callbacks_kwargs, audio_manager_kwargs)
        )
        realtime_process.daemon = True
        realtime_process.start()
    elif realtime_process.is_alive():
        config_queue.put(("START", (callbacks_kwargs, audio_manager_kwargs)))

    # Main orchestration messaging polling loop between UI context and background workers
    while running and realtime_process.is_alive():
        try:
            msg_type, data = ui_queue.get(timeout=0.1)

            if msg_type == "STATUS": yield (data, interactive_false, interactive_true)
            elif msg_type == "ERROR":
                realtime_stop(None)

                yield (translations["realtime_has_stop_with_error"], interactive_true, interactive_false)
                return
            elif msg_type == "INFO": gr_info(data)
            elif msg_type == "STOP": break
        except queue.Empty:
            continue

    return (
        translations["realtime_has_stop"], 
        interactive_true, 
        interactive_false
    )

def change_config(value, key, if_kwargs=False):
    """
    Hot-swaps tracking parameter values inside the background worker configurations at runtime.

    Args:
        value (Any): Target configuration value structure to inject.
        key (str): Configuration key matching parameter configurations.
        if_kwargs (bool): Target dictionary layout router. If True, maps into Pedalboard effects.
    """

    global callbacks_kwargs

    if running and realtime_process is not None and realtime_process.is_alive():
        if if_kwargs and value is not None:
            callbacks_kwargs["kwargs"][key] = value
        elif value is not None:
            callbacks_kwargs[key] = value

        # Push mutations directly to the background configuration channel
        config_queue.put(("UPDATE_CONFIG", callbacks_kwargs))

def realtime_stop(stop_realtime):
    """
    Dispatches termination signals or forcefully kills running streaming child processes.
    """

    global running, realtime_process, ui_queue, config_queue

    if realtime_process is not None and realtime_process.is_alive():
        running = False

        if stop_realtime == translations["stop_realtime_button"]: 
            gr_info(translations["stop_realtime"])

            # Pause real-time processing
            config_queue.put(("STOP", callbacks_kwargs))
            gr_info(translations["realtime_has_stop"])
            time.sleep(1)

            _interactive_true = interactive_true.copy()
            _interactive_true["value"] = translations["terminate"]

            return (
                translations["realtime_has_stop"], 
                interactive_true, 
                _interactive_true
            )
        else:
            # Interrupt and terminate the real-time worker to completely free up VRAM.

            realtime_process.terminate()
            realtime_process.join()
            realtime_process = ui_queue = config_queue = None

            _interactive_false = interactive_false.copy()
            _interactive_false["value"] = translations["stop_realtime_button"]

            gr_info(translations["realtime_has_terminate"])

            return (
                translations["realtime_has_terminate"],
                interactive_true,
                _interactive_false
            )
    else:
        gr_warning(translations["realtime_not_found"])

        _interactive_false = interactive_false.copy()
        _interactive_false["value"] = translations["stop_realtime_button"]

        return (
            translations["realtime_not_found"], 
            interactive_true, 
            _interactive_false
        )

def soundfile_record_audio(
    record_button,
    record_audio_path = None,
    export_format = "wav"
):
    """
    Toggle the pipeline's recording function.
    """

    global callbacks_kwargs

    if running and realtime_process.is_alive():
        if record_button == translations["start_record"]:
            gr_info(translations["starting_record"])

            if not record_audio_path: # Setup fallbacks path mapping if explicit pathing parameters are absent
                record_audio_path = os.path.join(configs["audios_path"], "record_audio.wav")

            # Enable audio recording flags
            callbacks_kwargs["record_audio"] = True
            callbacks_kwargs["record_audio_path"] = record_audio_path
            callbacks_kwargs["export_format"] = export_format

            # Push configuration updates across the multi-processing IPC pipe
            config_queue.put(("UPDATE_CONFIG", callbacks_kwargs))
            return translations["stop_record"], None
        else:
            gr_info(translations["stopping_record"])

            # Unset recording structures
            callbacks_kwargs["record_audio"] = False
            callbacks_kwargs["record_audio_path"] = None
            callbacks_kwargs["export_format"] = None

            config_queue.put(("UPDATE_CONFIG", callbacks_kwargs))
            return translations["start_record"], record_audio_path

    gr_warning(translations["realtime_not_found"])
    return translations["start_record"], None