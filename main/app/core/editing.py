import os
import sys
import random
import subprocess

sys.path.append(os.getcwd())

from main.app.variables import python, translations, configs
from main.app.core.ui import gr_info, gr_warning, process_output, replace_export_format

def audio_effects(
    input_path, 
    output_path, 
    resample, 
    resample_sr, 
    chorus_depth, 
    chorus_rate, 
    chorus_mix, 
    chorus_delay, 
    chorus_feedback, 
    distortion_drive, 
    reverb_room_size, 
    reverb_damping, 
    reverb_wet_level, 
    reverb_dry_level, 
    reverb_width, 
    reverb_freeze_mode, 
    pitch_shift, 
    delay_seconds, 
    delay_feedback, 
    delay_mix, 
    compressor_threshold, 
    compressor_ratio, 
    compressor_attack_ms, 
    compressor_release_ms, 
    limiter_threshold, 
    limiter_release, 
    gain_db, 
    bitcrush_bit_depth, 
    clipping_threshold, 
    phaser_rate_hz, 
    phaser_depth, 
    phaser_centre_frequency_hz, 
    phaser_feedback, 
    phaser_mix, 
    bass_boost_db, 
    bass_boost_frequency, 
    treble_boost_db, 
    treble_boost_frequency, 
    fade_in_duration, 
    fade_out_duration, 
    export_format, 
    chorus, 
    distortion, 
    reverb, 
    delay, 
    compressor, 
    limiter, 
    gain, 
    bitcrush, 
    clipping, 
    phaser, 
    treble_bass_boost, 
    fade_in_out, 
    audio_combination, 
    audio_combination_input, 
    main_vol, 
    combine_vol
):
    """Applies multiple professional audio DSP effects by invoking an external Python backend script.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Target path or directory for the output audio file.
        resample (bool): Toggle to resample the audio.
        resample_sr (int): Target sample rate for resampling.
        chorus_depth (float): Depth parameter for the chorus effect.
        chorus_rate (float): LFO modulation speed for chorus.
        chorus_mix (float): Dry/wet mix ratio for chorus.
        chorus_delay (int): Base delay time for chorus in milliseconds.
        chorus_feedback (float): Feedback amount for the chorus loop.
        distortion_drive (int): Input gain drive for saturation/distortion.
        reverb_room_size (float): Simulated room dimension size for reverb.
        reverb_damping (float): High-frequency absorption factor for reverb.
        reverb_wet_level (float): Wet (processed) signal level for reverb.
        reverb_dry_level (float): Dry (unprocessed) signal level for reverb.
        reverb_width (float): Stereo width representation for reverb.
        reverb_freeze_mode (bool): Reverb tail infinite sustain freeze toggle.
        pitch_shift (int): Number of semitones to shift the pitch.
        delay_seconds (float): Time interval between echo repeats in seconds.
        delay_feedback (float): Feedback gain decay loop for delay.
        delay_mix (float): Dry/wet mix percentage for delay.
        compressor_threshold (int): Decibel level above which compression activates.
        compressor_ratio (float): Input-to-output gain attenuation ratio.
        compressor_attack_ms (float): Reaction speed of compression onset in ms.
        compressor_release_ms (int): Recovery speed after signal drops in ms.
        limiter_threshold (int): Hard ceiling cap limit for output signal.
        limiter_release (int): Hard limiter release envelope time.
        gain_db (int): Master volume adjustment factor in decibels.
        bitcrush_bit_depth (int): Bit-depth reduction target for bitcrushing.
        clipping_threshold (int): Threshold ceiling for hard distortion clipping.
        phaser_rate_hz (float): LFO frequency rate for phaser sweeps.
        phaser_depth (float): Modulation sweep range depth for phaser.
        phaser_centre_frequency_hz (int): Center frequency of phaser notch filters.
        phaser_feedback (float): Resonance feedback value loop for phaser.
        phaser_mix (float): Blend ratio of original and phased signal.
        bass_boost_db (int): Low shelf equalizer filter boost value in dB.
        bass_boost_frequency (int): Cutoff frequency limit for low shelf filter.
        treble_boost_db (int): High shelf equalizer filter boost value in dB.
        treble_boost_frequency (int): Cutoff frequency limit for high shelf filter.
        fade_in_duration (float): Length of volume fade-in at start in seconds.
        fade_out_duration (float): Length of volume fade-out at end in seconds.
        export_format (str): Targeted extension container for audio output.
        chorus (bool): Master switch for chorus module activation.
        distortion (bool): Master switch for distortion module activation.
        reverb (bool): Master switch for reverb module activation.
        delay (bool): Master switch for delay module activation.
        compressor (bool): Master switch for dynamic compressor activation.
        limiter (bool): Master switch for peak limiter activation.
        gain (bool): Master switch for absolute gain volume scaling.
        bitcrush (bool): Master switch for bitcrush audio degradation effect.
        clipping (bool): Master switch for hard signal clipping distortion.
        phaser (bool): Master switch for multi-stage phaser filter effect.
        treble_bass_boost (bool): Master switch for dual shelf equalizer controls.
        fade_in_out (bool): Master switch for linear boundary volume fading.
        audio_combination (bool): Toggle to mix/merge two distinct audio streams.
        audio_combination_input (str): Path to the secondary audio file to blend.
        main_vol (int): Mix volume ratio multiplier for the primary stream.
        combine_vol (int): Mix volume ratio multiplier for the secondary stream.

    Returns:
        Optional[str]: Verified path pointing to the processed output file, or None if validation fails.
    """

    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    # Handle implicit directory inputs by assigning a default filename configuration
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_effects.{export_format}")
    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    output_path = process_output(output_path)
    
    gr_info(translations["start_apply_effect"])
    # Construct subprocess parameters dynamically to isolate core audio processing
    subprocess.run([
        python, 
        configs["audio_effects_path"], 
        "--input_path", input_path, 
        "--output_path", output_path, 
        "--resample", str(resample), 
        "--resample_sr", str(resample_sr), 
        "--chorus_depth", str(chorus_depth), 
        "--chorus_rate", str(chorus_rate), 
        "--chorus_mix", str(chorus_mix), 
        "--chorus_delay", str(chorus_delay), 
        "--chorus_feedback", str(chorus_feedback), 
        "--drive_db", str(distortion_drive), 
        "--reverb_room_size", str(reverb_room_size), 
        "--reverb_damping", str(reverb_damping), 
        "--reverb_wet_level", str(reverb_wet_level), 
        "--reverb_dry_level", str(reverb_dry_level), 
        "--reverb_width", str(reverb_width), 
        "--reverb_freeze_mode", str(reverb_freeze_mode), 
        "--pitch_shift", str(pitch_shift), 
        "--delay_seconds", str(delay_seconds), 
        "--delay_feedback", str(delay_feedback), 
        "--delay_mix", str(delay_mix), 
        "--compressor_threshold", str(compressor_threshold), 
        "--compressor_ratio", str(compressor_ratio), 
        "--compressor_attack_ms", str(compressor_attack_ms), 
        "--compressor_release_ms", str(compressor_release_ms), 
        "--limiter_threshold", str(limiter_threshold), 
        "--limiter_release", str(limiter_release), 
        "--gain_db", str(gain_db), 
        "--bitcrush_bit_depth", str(bitcrush_bit_depth), 
        "--clipping_threshold", str(clipping_threshold), 
        "--phaser_rate_hz", str(phaser_rate_hz), 
        "--phaser_depth", str(phaser_depth), 
        "--phaser_centre_frequency_hz", str(phaser_centre_frequency_hz), 
        "--phaser_feedback", str(phaser_feedback), 
        "--phaser_mix", str(phaser_mix), 
        "--bass_boost_db", str(bass_boost_db), 
        "--bass_boost_frequency", str(bass_boost_frequency), 
        "--treble_boost_db", str(treble_boost_db), 
        "--treble_boost_frequency", str(treble_boost_frequency), 
        "--fade_in_duration", str(fade_in_duration), 
        "--fade_out_duration", str(fade_out_duration), 
        "--export_format", export_format, 
        "--chorus", str(chorus), 
        "--distortion", str(distortion), 
        "--reverb", str(reverb), 
        "--pitchshift", str(pitch_shift != 0), 
        "--delay", str(delay), 
        "--compressor", str(compressor), 
        "--limiter", str(limiter), 
        "--gain", str(gain), 
        "--bitcrush", str(bitcrush), 
        "--clipping", str(clipping), 
        "--phaser", str(phaser), 
        "--treble_bass_boost", str(treble_bass_boost), 
        "--fade_in_out", str(fade_in_out), 
        "--audio_combination", str(audio_combination), 
        "--audio_combination_input", audio_combination_input, 
        "--main_volume", str(main_vol), 
        "--combination_volume", str(combine_vol)])

    gr_info(translations["success"])
    return replace_export_format(output_path, export_format)

def apply_voice_quirk(
    audio_path, 
    mode, 
    output_path, 
    export_format
):
    """
    Applies various fun/creative audio voice quirks and modifications natively using NumPy and Librosa.

    Args:
        audio_path (str): Source path of the input file.
        mode (str): Quirks dictionary key label string or matching mode index.
        output_path (str): Target filename destination template path.
        export_format (str): Desired final target file container type extension (e.g., 'wav', 'mp3').

    Returns:
        Optional[str]: Path pointing directly to the written audio artifact file, or None if validation fails.
    """

    if not audio_path or not os.path.exists(audio_path) or os.path.isdir(audio_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): 
        output_path = os.path.join(output_path, f"audio_quirk.{export_format}")

    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    output_path = process_output(output_path)
    
    gr_info(translations["start_apply_effect"])
    # Lazy imports to optimize framework initialization overhead
    import librosa
    import numpy as np
    import soundfile as sf

    def vibrato(y, sr, freq=5, depth=0.003):
        """Applies a frequency modulation (Vibrato) effect to the sound waveform array using a variable delay line."""

        # Calculate periodic sinus delay mapping allocations
        delay_modulation = depth * np.sin(2 * np.pi * freq * (np.arange(len(y)) / sr))
        # Interpolate spatial sampling lookup mappings securely
        return y[np.clip((np.arange(len(y)) + delay_modulation * sr).astype(int), 0, len(y) - 1)]

    # Load file with dynamic source target native sample rates representation intact
    y, sr = librosa.load(audio_path, sr=None)
    output_path = replace_export_format(output_path, export_format)

    # Convert named linguistic UI selections safely into explicit technical control integers
    mode = translations["quirk_choice"][mode]
    if mode == 0: mode = random.randint(1, 16) # Pick a random quirk

    if mode == 1: 
        y *= np.random.uniform(
            0.5, 
            0.8, 
            size=len(y)
        )
    elif mode == 2: 
        y = librosa.effects.pitch_shift(
            y=y + np.random.normal(0, 0.01, y.shape), 
            sr=sr, 
            n_steps=np.random.uniform(-1.5, -3.5)
        )
    elif mode == 3: 
        y = librosa.effects.time_stretch(
            librosa.effects.pitch_shift(
                y=y, 
                sr=sr, 
                n_steps=3
            ), 
            rate=1.2
        )
    elif mode == 4: 
        y = librosa.effects.time_stretch(
            librosa.effects.pitch_shift(
                y=y, 
                sr=sr, 
                n_steps=8
            ), 
            rate=1.3
        )
    elif mode == 5: 
        y = librosa.effects.time_stretch(
            librosa.effects.pitch_shift(
                y=y, 
                sr=sr, 
                n_steps=-3
            ), 
            rate=0.75
        )
    elif mode == 6: 
        y *= np.sin(np.linspace(0, np.pi * 20, len(y))) * 0.5 + 0.5
    elif mode == 7: 
        y = librosa.effects.time_stretch(
            vibrato(
                librosa.effects.pitch_shift(
                    y=y, 
                    sr=sr, 
                    n_steps=-4
                ), 
                sr, 
                freq=3, 
                depth=0.004
            ), 
            rate=0.85
        )
    elif mode == 8: 
        y = (y * 0.6) + np.pad(y, (sr // 2, 0), mode='constant')[:len(y)] * 0.4
    elif mode == 9: 
        y = librosa.effects.pitch_shift(
            y=y, 
            sr=sr, 
            n_steps=2
        ) + np.sin(
            np.linspace(0, np.pi * 20, len(y))
        ) * 0.02
    elif mode == 10: 
        y = vibrato(
            y, 
            sr, 
            freq=8, 
            depth=0.005
        )
    elif mode == 11: 
        y = librosa.effects.time_stretch(
            librosa.effects.pitch_shift(y=y, sr=sr, n_steps=4), 
            rate=1.25
        )
    elif mode == 12: 
        y = np.hstack([
            np.pad(f, (0, int(len(f)*0.3)), mode='edge') 
            for f in librosa.util.frame(y, frame_length=2048, hop_length=512).T
        ])
    elif mode == 13: 
        y = np.concatenate([
            y, 
            np.sin(
                2 * np.pi * np.linspace(0, 1, int(0.05 * sr))
            ) * 0.04
        ])
    elif mode == 14: 
        y += np.random.normal(
            0, 
            0.005, 
            len(y)
        )
    elif mode == 15:
        frame = int(sr * 0.8)
        chunks = [y[i:i + frame] for i in range(0, len(y), frame)]

        np.random.shuffle(chunks)
        y = np.concatenate(chunks)
    elif mode == 16:
        frame = int(sr * 0.3)

        for i in range(0, len(y), frame * 2):
            y[i:i+frame] = y[i:i+frame][::-1]

    sf.write(output_path, y, sr, format=export_format)
    del y, sr # Active garbage clearing to protect performance from large matrix leaks

    gr_info(translations["success"])
    return output_path