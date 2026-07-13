import os
import sys
import requests

from bs4 import BeautifulSoup

def Mediafire_Download(url, output=None, filename=None):
    """
    Parses a MediaFire landing page to extract and stream the direct file download.

    Args:
        url (str): The MediaFire sharing URL.
        output (str, optional): The directory where the file should be saved.
          Defaults to the directory of the current script file.
        filename (str, optional): The custom name for the downloaded file.
          Defaults to the parsed folder name string from the URL.

    Returns:
        str: The full absolute or relative file path to the downloaded payload.

    Raises:
        RuntimeError: If scraping the download token fails or any network and
          I/O error occurs.
    """

    # Step 1: Assign default values for local storage targets if not provided
    if not filename: filename = url.split('/')[-2]
    if not output: output = os.path.dirname(os.path.realpath(__file__))

    output_file = os.path.join(output, filename)
    # Step 2: Initialize a structured requests session with a browser User-Agent header
    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    try:
        # Step 3: Fetch landing page and parse the HTML DOM to extract the raw target anchor link
        bs4 = BeautifulSoup(
            sess.get(url).content, 
            "html.parser"
        ).find(id="downloadButton").get("href")

        # Step 4: Stream the binary payload using chunk iterations
        with requests.get(
            bs4, 
            stream=True
        ) as r:
            r.raise_for_status()

            with open(output_file, "wb") as f:
                # Resolve content size metadata safely (fallback to 0 if missing)
                total_length = int(r.headers.get('content-length'))
                download_progress = 0
                # Read streamed buffer bytes progressively
                for chunk in r.iter_content(chunk_size=1024):
                    download_progress += len(chunk)
                    f.write(chunk)

                    # Step 5: Render terminal-friendly download status calculations dynamically
                    stdout = f"\r[{filename}]: {int(100 * download_progress / total_length)}% ({round(download_progress / 1024 / 1024, 2)}mb/{round(total_length / 1024 / 1024, 2)}mb)"
                    # Write inline output log directly to system stream
                    sys.stdout.write(stdout)
                    sys.stdout.flush()

        sys.stdout.write("\n")
        return output_file
    except Exception as e:
        raise RuntimeError(f"MediaFire Download operational failure: {e}")