import os
import requests
import tqdm


def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    if output_path is None: output_path = os.path.basename(url)
    else: output_path = os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path

    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        progress_bar = tqdm.tqdm(total=int(response.headers.get("content-length", 0)), ncols=100, unit="byte")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)

        progress_bar.close()
        return output_path
    else: raise ValueError(response.status_code)