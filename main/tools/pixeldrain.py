import os
import requests


def pixeldrain(url, output_dir):
    try:
        file_id = url.split("pixeldrain.com/u/")[1]
        response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")

        if response.status_code == 200:
            file_name = (response.headers.get("Content-Disposition").split("filename=")[-1].strip('";'))
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "wb") as newfile:
                newfile.write(response.content)
                return file_path
        else: return None
    except Exception as e:
        raise RuntimeError(e)