import os
import sys

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import config, translations, configs

def f0_gen(
    audio = None,
    predictor_onnx = False,
    f0_method = "rmvpe",
    pitch = 0,
    filter_radius = 3,
    f0_autotune = False,
    f0_autotune_strength = 1.0,
    proposal_pitch = False,
    proposal_pitch_threshold = 255.0,
    image_path = None,
    txt_path = None
):
    """
    Generates Fundamental Frequency (F0) pitch tracks and exports visualizations/data logs.

    This function is designed to run inside an isolated subprocess to prevent GPU memory leaks
    and framework deadlocks during intense neural network pitch estimation operations.

    Args:
        audio (Optional[str]): Path to the input source audio file.
        predictor_onnx (bool): Toggle to activate the optimized ONNX runtime backend.
        f0_method (str): Algorithm style to extract pitch (e.g., 'rmvpe', 'crepe', 'harvest').
        pitch (Union[int, float]): Semitones key offset value to shift the pitch up or down.
        filter_radius (int): Median filter radius applied for handling pitch tracking outliers.
        f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
        f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
        proposal_pitch (bool): Enable automatic pitch key shifting calculation based on median F0 alignment.
        proposal_pitch_threshold (float): The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.
        image_path (Optional[str]): Destination path for saving the plotted matplotlib PNG chart.
        txt_path (Optional[str]): Destination path for saving raw CSV format F0 timeline logs.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from main.library.utils import clear_gpu_cache
    from main.library.audio.audio import load_audio
    from main.library.predictors.Generator import Generator

    # Standardize sample rate processing down to standard model analysis frequency (16kHz)
    y = load_audio(audio, sample_rate=16000)
    # Initialize the core neural engine with strict frequency ceiling boundaries
    f0_generator = Generator(
        sample_rate=16000, 
        hop_length=160, 
        f0_min=configs.get("f0_min", 50), 
        f0_max=configs.get("f0_max", 1100), 
        alpha=0.5, 
        is_half=config.is_half, 
        device=config.device, 
        predictor_onnx=predictor_onnx
    )

    # Execute target pitch analysis extractor calculation workflow
    _, pitchf = f0_generator.calculator(
        x_pad=config.x_pad, 
        f0_method=f0_method, 
        x=y, 
        f0_up_key=pitch, 
        p_len=None, 
        filter_radius=filter_radius, 
        f0_autotune=f0_autotune, 
        f0_autotune_strength=f0_autotune_strength, 
        manual_f0=None, 
        proposal_pitch=proposal_pitch,
        proposal_pitch_threshold=proposal_pitch_threshold
    )

    F_temp = np.array(pitchf, dtype=np.float32)
    # Mask unvoiced frames containing absolute zero energy to avoid division by zero errors
    F_temp[F_temp == 0] = np.nan

    # Logarithmic pitch formula anchored on standard MIDI base tuning frequency (~8.18Hz)
    f0 = 1200 * np.log2(F_temp / 8.175798915643707)

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(pitchf):
            f.write(f"{i},{f0_value}\n")
    
    # Force manual dereferencing allocations to trigger proactive Python garbage collection
    del y, pitchf, F_temp, f0
    del f0_generator
    # Release VRAM context allocations on CUDA architectures
    clear_gpu_cache()

def f0_extract(
    audio = None, 
    f0_method = "rmvpe", 
    predictor_onnx = False,
    pitch = 0,
    filter_radius = 3,
    f0_autotune = False,
    f0_autotune_strength = 1.0,
    proposal_pitch = False,
    proposal_pitch_threshold = 255.0
):
    """
    Validates inputs and coordinates isolated multi-processing workers to safely extract audio F0 pitch attributes.

    Args:
        audio (Optional[str]): Path pointing toward target source audio file asset on disc.
        f0_method (str): Pitch tracing tracking choice routine logic structure. Default is "rmvpe".
        predictor_onnx (bool): Active execution deployment flag toggling ONNX acceleration frameworks.
        pitch (Union[int, float]): Value scaling baseline tuning transpositions.
        filter_radius (int): Frequency processing window frame filtering scale dimension size.
        f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
        f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
        proposal_pitch (bool): Enable automatic pitch key shifting calculation based on median F0 alignment.
        proposal_pitch_threshold (float): The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.

    Returns:
        List[Optional[str]]: A pair of absolute file paths matching [text_data_log_path, plotted_graph_image_path]. Returns [None, None] if input verification checks fail.
    """

    # Verify file sanity profiles prior to spending heavy system initialization cycles
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2
    
    import multiprocessing as mp

    from main.library.utils import check_assets
    # Validate weight checkpoint file architectures prior to process spawning forks
    check_assets(f0_method, "", predictor_onnx, "")

    # Automatically map out working storage output directories
    f0_path = os.path.join(configs["f0_path"], os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])
    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    # Use 'spawn' start method context strictly to ensure safety boundaries across PyTorch/CUDA environments
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=f0_gen, 
        args=(
            audio,
            predictor_onnx,
            f0_method,
            pitch,
            filter_radius,
            f0_autotune,
            f0_autotune_strength,
            proposal_pitch,
            proposal_pitch_threshold,
            image_path,
            txt_path
        )
    )

    process.start()
    process.join() # Hold context until targeted calculation pipeline successfully terminates

    gr_info(translations["extract_done"])
    return [txt_path, image_path]