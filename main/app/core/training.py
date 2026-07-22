import os
import sys
import time
import shutil
import codecs
import datetime
import threading
import subprocess

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs, file_types, logger

def if_done(done, p):
    """
    Monitors a subprocess and updates a shared status flag upon completion.
    """

    while 1:
        # Check if the process is still running (poll returns None if alive)
        if p.poll() is None: time.sleep(0.5)
        else: break

    # Mark the process as completed
    done[0] = True

def log_read(done, name, start_time):
    """
    Generator that continuously yields new log entries matching specific keywords.

    Reads from the global 'app.log' file periodically and filters entries by 
    timestamp and process relevance.

    Yields:
        str: Concatenated log lines generated since the start_time.
    """

    log_file = os.path.join(configs["logs_path"], "app.log")

    def read_logs():
        """Helper to scan the log file and filter rows based on conditions."""

        logs = []
        # Open log safely with UTF-8 encoding
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Skip noise: ignore DEBUG logs, lines without matching keywords, or empty lines
                    if ("DEBUG" in line or not any(n in line for n in name) or line.strip() == ""): continue

                    # Parse log timestamp (assumes format: "YYYY-MM-DD HH:MM:SS.ffffff | LOG_MESSAGE")
                    timestamp = datetime.datetime.strptime(line.split("|")[0].strip(), "%Y-%m-%d %H:%M:%S.%f")

                    # Keep logs generated during or after the current operation
                    if timestamp >= start_time: logs.append(line)
                except ValueError:
                    # Gracefully skip lines that do not match the expected timestamp layout
                    continue

        return "".join(logs)

    # Main pooling loop for streaming logs
    while 1:
        yield read_logs()

        time.sleep(1)
        # Break out if the subprocess has finished tracking
        if done[0]:
            break

    # Yield final logs one last time to capture trailing entries
    yield read_logs()

def create_dataset(
    input_data,
    output_dirs,
    skip_seconds,
    skip_start_audios,
    skip_end_audios,
    separate,
    model_name, 
    reverb_model, 
    denoise_model,
    sample_rate,
    shifts, 
    batch_size, 
    overlap, 
    aggression,
    hop_length, 
    window_size,
    segments_size, 
    post_process_threshold,
    enable_tta,
    enable_denoise,
    high_end_process,
    enable_post_process,
    separate_reverb,
    clean_dataset,
    clean_strength
):
    """
    Executes the dataset creation process as an asynchronous external script.
    Launches a dedicated subprocess to structure audio data, remove silent chunks,
    isolate vocals/instruments, and clean up audio segments.

    Yields:
        str: Live execution console text pulled from application logs.
    """

    # Notify user interface that execution is starting
    gr_info(translations["start_create_dataset"])
    start_time = datetime.datetime.now()

    # Launch script as an independent detached background task
    p = subprocess.Popen([
        python,
        configs["create_dataset_path"],
        "--input_data", input_data,
        "--output_dirs", output_dirs,
        "--skip_seconds", str(skip_seconds),
        "--skip_start_audios", str(skip_start_audios),
        "--skip_end_audios", str(skip_end_audios),
        "--separate", str(separate),
        "--model_name", model_name,
        "--reverb_model", reverb_model,
        "--denoise_model", denoise_model,
        "--sample_rate", str(sample_rate),
        "--shifts", str(shifts),
        "--batch_size", str(batch_size),
        "--overlap", str(overlap),
        "--aggression", str(aggression),
        "--hop_length", str(hop_length),
        "--window_size", str(window_size),
        "--segments_size", str(segments_size),
        "--post_process_threshold", str(post_process_threshold),
        "--enable_tta", str(enable_tta),
        "--enable_denoise", str(enable_denoise),
        "--high_end_process", str(high_end_process),
        "--enable_post_process", str(enable_post_process),
        "--separate_reverb", str(separate_reverb),
        "--clean_dataset", str(clean_dataset),
        "--clean_strength", str(clean_strength),
    ])

    # Shared thread controller status
    done = [False]
    # Spawn thread to watch over process life cycle
    threading.Thread(target=if_done, args=(done, p)).start()

    # Stream relevant log channels to the UI component
    for log in log_read(done, ["create_dataset", "separate_music", "separator"], start_time):
        yield log

def create_reference(
    audio_path, 
    reference_name, 
    pitch_guidance, 
    version, 
    embedder_model, 
    embedders_mode, 
    f0_method, 
    predictor_onnx, 
    f0_up_key, 
    filter_radius, 
    f0_autotune, 
    f0_autotune_strength, 
    proposal_pitch, 
    proposal_pitch_threshold, 
    alpha=0.5,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5
):
    """
    Generates acoustic feature reference vectors from a target audio clip.

    Yields:
        str: Continuous execution stdout parsed through system logger pipelines.
    """

    gr_info(translations["start_create_reference"])
    start_time = datetime.datetime.now()
    # Fire subprocess executing the reference generator backend
    p = subprocess.Popen([
        python,
        configs["create_reference_path"],
        "--audio_path", audio_path,
        "--reference_name", reference_name,
        "--pitch_guidance", str(pitch_guidance),
        "--version", version,
        "--embedder_model", embedder_model,
        "--embedders_mode", embedders_mode,
        "--f0_method", f0_method,
        "--predictor_onnx", str(predictor_onnx),
        "--f0_up_key", str(f0_up_key),
        "--filter_radius", str(filter_radius),
        "--f0_autotune", str(f0_autotune),
        "--f0_autotune_strength", str(f0_autotune_strength),
        "--proposal_pitch", str(proposal_pitch),
        "--proposal_pitch_threshold", str(proposal_pitch_threshold),
        "--alpha", str(alpha),
        "--embedders_mix", str(embedders_mix),
        "--embedders_mix_layers", str(embedders_mix_layers),
        "--embedders_mix_ratio", str(embedders_mix_ratio)
    ])

    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, ["create_reference"], start_time):
        yield log

def preprocess(
    model_name, 
    sample_rate, 
    cpu_core, 
    cut_preprocess, 
    process_effects, 
    dataset, 
    clean_dataset, 
    clean_strength, 
    chunk_len=3.0, 
    overlap_len=0.3, 
    normalization_mode="none",
    architecture="RVC"
):
    """
    Execute the training data preprocessing process before performing extraction.

    Yields:
        str: Streaming update content blocks generated from standard output logs.
    """

    # Strip the string suffix ('k') and convert shorthand (e.g., "40k") to integer frequency (e.g., 40000)
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    # Validate that workspace naming context is not empty
    if not model_name: return gr_warning(translations["provide_name"])

    # Optional integrity verification checking if the source dataset contains valid non-empty files
    if configs.get("check_data", False):
        try:
            found = False
            if os.path.exists(dataset):
                for root, _, files in os.walk(dataset):
                    for f in files:
                        # Identify match against registered format tables and confirm file size > 0 bytes
                        if f.lower().endswith(tuple(file_types)) and os.path.getsize(os.path.join(root, f)) > 0:
                            found = True
                            break

                    if found: break
            if not found: return gr_warning(translations["not_found_data"])
        except Exception:
            return gr_warning(translations["not_found_data"])
    
    start_time = datetime.datetime.now()
    model_dir = os.path.join(configs["logs_path"], model_name)
    # Flush existing artifact folders to enforce clean workspace operations
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    # Initialize backend subprocess executing the audio slicing pipeline script
    p = subprocess.Popen([
        python,
        configs["preprocess_path"],
        "--model_name", model_name,
        "--dataset_path", dataset,
        "--sample_rate", str(sr),
        "--cpu_cores", str(cpu_core),
        "--cut_preprocess", cut_preprocess,
        "--process_effects", str(process_effects),
        "--clean_dataset", str(clean_dataset),
        "--clean_strength", str(clean_strength),
        "--chunk_len", str(chunk_len),
        "--overlap_len", str(overlap_len),
        "--normalization_mode", normalization_mode,
        "--architecture", architecture
    ])

    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    # Pre-create directory layout to hold output data streams safely
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, ["preprocess"], start_time):
        yield log

def extract(
    model_name, 
    version, 
    method, 
    pitch_guidance, 
    hop_length, 
    cpu_cores, 
    gpu, 
    sample_rate, 
    embedders, 
    custom_embedders, 
    predictor_onnx, 
    embedders_mode, 
    f0_autotune, 
    f0_autotune_strength, 
    hybrid_method, 
    alpha=0.5, 
    include_mutes=2,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    architecture = "RVC"
):
    """
    Extract features and pitch from the previously preprocessed data.

    Yields:
        str: Continuous execution stdout strings generated during processing runtime.
    """

    # Choose hybrid method if core choice is flagged as 'hybrid', else map native selection
    # Do the same validation check for custom embedding models overrides
    f0_method, embedder_model = (
        method if method != "hybrid" else hybrid_method, 
        embedders if embedders != "custom" else custom_embedders
    )

    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)

    # Workspace directory integrity verification before allowing extraction tasks
    if configs.get("check_data", False):
        try:
            # Confirm preprocess operations successfully output target slicing maps
            if not any(
                os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) 
                for f in os.listdir(os.path.join(model_dir, "sliced_audios"))
            ) or not any(
                os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) 
                for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))
            ): 
                return gr_warning(translations["not_found_data_preprocess"])
        except:
            return gr_warning(translations["not_found_data_preprocess"])
        
    start_time = datetime.datetime.now()
    # Launch feature analysis script as a background process node
    p = subprocess.Popen([
        python,
        configs["extract_path"],
        "--model_name", model_name,
        "--rvc_version", version,
        "--f0_method", f0_method,
        "--pitch_guidance", str(pitch_guidance),
        "--hop_length", str(hop_length),
        "--cpu_cores", str(cpu_cores),
        "--gpu", str(gpu),
        "--sample_rate", str(sr),
        "--embedder_model", embedder_model,
        "--predictor_onnx", str(predictor_onnx),
        "--embedders_mode", embedders_mode,
        "--f0_autotune", str(f0_autotune),
        "--f0_autotune_strength", str(f0_autotune_strength),
        "--alpha", str(alpha),
        "--include_mutes", str(include_mutes),
        "--embedders_mix", str(embedders_mix),
        "--embedders_mix_layers", str(embedders_mix_layers),
        "--embedders_mix_ratio", str(embedders_mix_ratio),
        "--architecture", architecture
    ])
    
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, ["extract", "embedding", "feature"], start_time):
        yield log

def create_index(
    model_name, 
    rvc_version, 
    index_algorithm,
    nprobe=1
):
    """
    Builds a vector search index file from extracted audio embeddings.

    Yields:
        str: Continuous feedback log data streams from index compilation steps.
    """

    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)

    # Ensure vector source map directories exist and contain valid raw material
    if configs.get("check_data", False):
        try:
            if not any(
                os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) 
                for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))
            ): 
                return gr_warning(translations["not_found_data_extract"])
        except:
            return gr_warning(translations["not_found_data_extract"])
    
    start_time = datetime.datetime.now()
    # Launch subprocess to index feature vectors
    p = subprocess.Popen([
        python, 
        configs["create_index_path"], 
        "--model_name", model_name, 
        "--rvc_version", rvc_version,
        "--index_algorithm", index_algorithm,
        "--nprobe", str(nprobe)
    ])

    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, ["create_index"], start_time):
        yield log

def training(
    model_name, 
    rvc_version, 
    save_every_epoch, 
    save_only_latest, 
    save_every_weights, 
    total_epoch, 
    sample_rate, 
    batch_size, 
    gpu, 
    pitch_guidance, 
    not_pretrain, 
    custom_pretrained, 
    pretrain_g, 
    pretrain_d, 
    clean_up, 
    cache, 
    model_author, 
    vocoder, 
    checkpointing, 
    deterministic, 
    benchmark, 
    optimizer, 
    custom_reference=False, 
    reference_name="", 
    multiscale_mel_loss=False,
    embedders="hubert_base",
    custom_embedders=None,
    cosine_annealing_lr=False,
    architecture="RVC"
):
    """
    Initializes and runs the main model training subprocess.
    Configures pre-trained models, structures file arguments, downloads checkpoints,
    and runs the optimization script while tracking process status.

    Yields:
        str: Segmented slice of the application's trailing execution stdout lines.
    """

    # SVC architecture always requires pitch guidance enabled
    if architecture == "SVC": pitch_guidance = True

    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join(configs["logs_path"], model_name)
    # Remove stale process ID files before launching a new session
    if os.path.exists(os.path.join(model_dir, "train_pid.txt")): 
        os.remove(os.path.join(model_dir, "train_pid.txt"))

    # Verify that the extraction directory contains files before starting training
    if configs.get("check_data", False):
        try:
            if not any(
                os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) 
                for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))
            ): 
                return gr_warning(translations["not_found_data_extract"])
        except:
            return gr_warning(translations["not_found_data_extract"])
    
    # Pre-trained base models initialization and download logic
    if not not_pretrain:
        if not custom_pretrained: 
            pretrain_dir = (
                configs["pretrained_v2_path"] 
                if rvc_version == 'v2' else 
                configs["pretrained_v1_path"]
            )

            # ROT13 obfuscated URL decoding for the HuggingFace pre-trained models repository
            download_version = codecs.decode(
                f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_", 
                "rot13"
            ) + f"{rvc_version}/"

            # Selection matrix mapping pitch guidance + sample rate to specific filenames
            pretrained_selector = {
                True: {
                    24000: ("f0G24k.pth", "f0D24k.pth"), 
                    32000: ("f0G32k.pth", "f0D32k.pth"), 
                    40000: ("f0G40k.pth", "f0D40k.pth"), 
                    44100: ("f0G44k.pth", "f0D44k.pth"),
                    48000: ("f0G48k.pth", "f0D48k.pth")
                }, 
                False: {
                    24000: ("G24k.pth", "D24k.pth"), 
                    32000: ("G32k.pth", "D32k.pth"), 
                    40000: ("G40k.pth", "D40k.pth"), 
                    44100: ("G44k.pth", "D44k.pth"),
                    48000: ("G48k.pth", "D48k.pth")
                }
            }

            pg2, pd2 = "", ""
            pg, pd = pretrained_selector[pitch_guidance][sr]

            # Construct targeted pre-trained model filenames based on architectural parameters
            if vocoder != 'Default': pg2, pd2 = pg2 + vocoder + "_", pd2 + vocoder + "_"
            if embedders not in ["hubert_base", "contentvec_base"]: pg2, pd2 = pg2 + embedders + "_", pd2 + embedders + "_"
            if architecture != "RVC": pg2, pd2 = architecture + "_" + pg2, architecture + "_" + pd2

            pg2, pd2 = pg2 + pg, pd2 + pd
            logger.debug("PG: " + pg2 + " PD: " + pd2)

            pretrained_G, pretrained_D = (
                os.path.join(
                    pretrain_dir,
                    pg2
                ), 
                os.path.join(
                    pretrain_dir,
                    pd2
                )
            )

            # Automatically fetch base files from Hugging Face if they are missing locally
            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    huggingface.HF_download_file(
                        "".join(
                            [
                                download_version, 
                                pg2
                            ]
                        ),
                        os.path.join(
                            pretrain_dir,
                            pg2
                        )
                    )
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    huggingface.HF_download_file(
                        "".join(
                            [
                                download_version, 
                                pd2
                            ]
                        ), 
                        os.path.join(
                            pretrain_dir,
                            pd2
                        )
                    )
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G = pretrained_D = None
        else:
            # Handle localized custom pre-trained models definitions
            if not pretrain_g: 
                return gr_warning(translations["provide_pretrained"].format(dg="G"))

            if not pretrain_d: 
                return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg2, pd2 = pretrain_g, pretrain_d
            pretrained_G, pretrained_D = (
                os.path.join(configs["pretrained_custom_path"], pg2) if not os.path.exists(pg2) else pg2, 
                os.path.join(configs["pretrained_custom_path"], pd2) if not os.path.exists(pd2) else pd2
            )

            if not os.path.exists(pretrained_G): 
                return gr_warning(translations["not_found_pretrain"].format(dg="G"))

            if not os.path.exists(pretrained_D): 
                return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: 
        pretrained_G = pretrained_D = None
        gr_warning(translations["not_use_pretrain"])

    # Resolve cross-reference paths if custom referencing mechanisms are toggled
    if custom_reference:
        embedder_model = embedders if embedders != "custom" else custom_embedders

        reference_path = os.path.join(
            configs["reference_path"], 
            "".join([
                reference_name, 
                "_", 
                rvc_version, 
                "_", 
                embedder_model, 
                "_", 
                str(pitch_guidance)
            ])
        )

        if not os.path.exists(reference_path):
            gr_warning(translations["not_found_reference"])

            custom_reference = False
            reference_path = None
    else: reference_path = None

    start_time = datetime.datetime.now()
    # Launch core neural network training module script
    p = subprocess.Popen([
        python,
        configs["train_path"],
        "--model_name", model_name,
        "--rvc_version", rvc_version,
        "--save_every_epoch", str(save_every_epoch),
        "--save_only_latest", str(save_only_latest),
        "--save_every_weights", str(save_every_weights),
        "--total_epoch", str(total_epoch),
        "--batch_size", str(batch_size),
        "--gpu", str(gpu),
        "--pitch_guidance", str(pitch_guidance),
        "--cleanup", str(clean_up),
        "--cache_data_in_gpu", str(cache),
        "--g_pretrained_path", str(pretrained_G),
        "--d_pretrained_path", str(pretrained_D),
        "--model_author", model_author,
        "--vocoder", vocoder,
        "--checkpointing", str(checkpointing),
        "--deterministic", str(deterministic),
        "--benchmark", str(benchmark),
        "--optimizer", optimizer,
        "--use_custom_reference", str(custom_reference),
        "--reference_path", str(reference_path),
        "--multiscale_mel_loss", str(multiscale_mel_loss),
        "--use_cosine_annealing_lr", str(cosine_annealing_lr),
        "--architecture", architecture,
        "--filelist_path", "None",
        "--config_save_path", "None",
        "--spec_dir", "None",
        "--eval_dir", "None",
        "--cache_spectrogram", str(True),
        "--save_the_pid", str(True),
        "--custom_training", str(False)
    ])

    done = [False]
    # Save active Process ID tracking variable values to a text file for external management controls
    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()
    # Capture outputs, formatting long streaming outputs to show only the 50 most recent lines
    for log in log_read(done, ["train", "synthesizers", "extract_model", "utils"], start_time):
        lines = log.splitlines()

        if len(lines) > 50: 
            log = "\n".join(lines[-50:])

        yield log