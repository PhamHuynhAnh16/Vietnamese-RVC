import os
import tqdm
import requests

try:
    import wget
except:
    wget = None

def HF_download_file(url, output_path=None):
    """
    Downloads a file from Hugging Face, resolving blob links to raw file paths.

    Args:
        url (str): The Hugging Face file URL (supports standard or blob links).
        output_path (str, optional): Target file path or directory. If None,
          saves to the current working directory using the file's original name.
          Defaults to None.

    Returns:
        str: The absolute or relative path to the downloaded file.

    Raises:
        ValueError: If requests fails to download the file (non-200 status
          code).
    """

    # Step 1: Normalize Hugging Face URL structure to get the direct payload download link
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    # Step 2: Dynamically calculate the safe local destination path
    output_path = (
        os.path.basename(url) 
    ) if output_path is None else (
        os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path
    )

    # Step 3: Execute download operation
    if wget != None: 
        # Strategy A: Use optimized module fallback if available
        wget.download(
            url, 
            out=output_path
        )
    else:
        # Strategy B: Stream download natively via standard HTTP chunk retrieval
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            # Initialize terminal status progress bar tracker
            progress_bar = tqdm.tqdm(
                total=int(response.headers.get("content-length", 0)), 
                desc=os.path.basename(url), 
                ncols=100, 
                unit="byte", 
                leave=False
            )

            # Iteratively write data chunks directly to disk
            with open(output_path, "wb") as f:
                # Read chunks up to 10MB in memory buffer sequentially
                for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                    progress_bar.update(len(chunk))
                    f.write(chunk)

            progress_bar.close()
        else: 
            # Raise an operational exception if server returns an error code
            raise ValueError(f"Failed to download file, server responded with status code: {response.status_code}")

    return output_path