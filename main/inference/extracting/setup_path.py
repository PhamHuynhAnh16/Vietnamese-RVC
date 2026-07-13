import os

def setup_paths(exp_dir, version = None):
    """
    Sets up and creates the necessary workspace directories for feature extraction.

    Depending on whether an RVC/SVC `version` is provided, this function dynamically 
    routes path building for either structural feature embeddings or fundamental 
    frequency (F0) outputs.

    Args:
        exp_dir (str): Main experiment directory path on disk.
        version (str, optional): Target framework variation version (e.g., 'v1' or 'v2'). If provided, configures embedding paths. Defaults to None.

    Returns:
        tuple: Contains (wav_path, out_path) if version is provided, otherwise (wav_path, output_root1, output_root2).
    """

    # Define the common absolute folder holding standard 16kHz downsampled audio chunks
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if version:
        # Branch A: Setup workspace infrastructure for embedding extraction (e.g., v2_extracted)
        out_path = os.path.join(exp_dir, f"{version}_extracted")
        os.makedirs(out_path, exist_ok=True)

        return wav_path, out_path
    else:
        # Branch B: Setup workspace infrastructure for pitch estimation (coarse vs fine voiced pitch arrays)
        output_root1, output_root2 = (
            os.path.join(exp_dir, "f0"), 
            os.path.join(exp_dir, "f0_voiced")
        )

        # Ensure directories exist safely without throwing errors if they were previously created
        os.makedirs(output_root1, exist_ok=True) 
        os.makedirs(output_root2, exist_ok=True)

        return wav_path, output_root1, output_root2