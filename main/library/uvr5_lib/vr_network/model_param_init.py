import json
import pickle

def int_keys(pairs):
    """
    Custom object pairs hook for JSON decoding that converts numeric string keys into integers.

    Args:
        pairs (List[Tuple[Any, Any]]): A list of key-value tuples extracted during JSON parsing.

    Returns:
        Dict[Any, Any]: A dictionary where string-digit keys are converted to integer keys.
    """

    result_dict = {}

    for key, value in pairs:
        # Check if the key is a string containing only digits, then cast it to int
        if isinstance(key, str) and key.isdigit(): key = int(key)
        result_dict[key] = value

    return result_dict

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
            # Load binary configurations using pickle
            with open(config_path, "rb") as f:
                data = pickle.load(f)
                # Extract the subset parameters using the specified binary dictionary key
                self.param = data[key_in_bin]
        else:
            # Load standard textual configurations using json
            with open(config_path, "r", encoding="utf-8") as f:
                # Use the custom int_keys hook to ensure valid integer key lookups
                self.param = json.loads(f.read(), object_pairs_hook=int_keys)

        # Ensure essential boolean stereo/processing flags exist; default to False if missing
        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            if k not in self.param:
                self.param[k] = False

        # Normalize key naming conventions: mirror 'n_bins' value over to 'bins' if it exists
        if "n_bins" in self.param:
            self.param["bins"] = self.param["n_bins"]