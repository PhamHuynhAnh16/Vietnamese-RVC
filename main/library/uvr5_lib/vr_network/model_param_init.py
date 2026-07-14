import json
import struct

class ModelParameters(object):
    """
    Handles loading, parsing, and normalizing model configurations from 
    either binary pickle files (.bin) or standard JSON files.
    """

    def __init__(
        self, 
        config_path="", 
        key_in_bin=None
    ):
        """
        Args:
            config_path (str, optional): Path to the configuration file (.json or .bin). Defaults to "".
            key_in_bin (Optional[str], optional): The specific key to extract if loading from a .bin pickle file. Defaults to None.
        """

        # Determine the file type by checking the extension
        if config_path.endswith(".bin"):
            target_content = None
            # Load binary configurations using pickle
            with open(config_path, "rb") as f:
                while 1:
                    key_len_bytes = f.read(4)
                    if not key_len_bytes: break

                    current = f.read(struct.unpack("I", key_len_bytes)[0]).decode('utf-8')
                    data_len = struct.unpack("I", f.read(4))[0]

                    if current == key_in_bin:
                        target_content = f.read(data_len)
                        break
                    else:
                        f.seek(data_len, 1)

            if target_content is None:
                raise KeyError(f"Configuration key '{key_in_bin}' not found inside safe binary database.")

            self.param = json.loads(target_content.decode('utf-8'))
        else:
            # Load standard textual configurations using json
            with open(config_path, "r", encoding="utf-8") as f:
                # Use the custom int_keys hook to ensure valid integer key lookups
                self.param = json.loads(f.read())

        # Ensure essential boolean stereo/processing flags exist; default to False if missing
        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            if k not in self.param:
                self.param[k] = False

        # Normalize key naming conventions: mirror 'n_bins' value over to 'bins' if it exists
        if "n_bins" in self.param:
            self.param["bins"] = self.param["n_bins"]