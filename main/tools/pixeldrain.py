import os
import requests

def pixeldrain(url, output_dir):
    """
    Downloads a file from Pixeldrain using its public API backend.

    Args:
        url (str): The public sharing URL of the Pixeldrain file (e.g.,
          https://pixeldrain.com/u/file_id).
        output_dir (str): The local directory path where the downloaded file
          should be stored.

    Returns:
        Optional[str]: The absolute or relative path to the downloaded file if
        successful, or None if the server returns a non-200 status code.

    Raises:
        RuntimeError: If any internal network connection or I/O processing operation fails.
    """

    try:
        # Step 1: Extract the unique file ID from the URL and query the direct file API endpoint
        response = requests.get(f"https://pixeldrain.com/api/file/{url.split('pixeldrain.com/u/')[1]}")

        if response.status_code == 200:
            # Build the clean target storage path
            file_path = os.path.join(
                output_dir, 
                # Step 2: Parse out the original filename from the 'Content-Disposition' header safely
                response.headers.get("Content-Disposition").split("filename=")[-1].strip('";')
            )

            # Step 3: Write the binary stream bytes directly into the local file
            with open(file_path, "wb") as newfile:
                newfile.write(response.content)

            return file_path
        # Return None explicitly if the resource is unavailable or has expired
        return None
    except Exception as e:
        raise RuntimeError(e)