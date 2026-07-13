import os
import sys
import ast
import torch
import itertools
import collections

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import if_main_process, ddp_barrier
from main.library.speaker_diarization.features import register_checkpoint_hooks, mark_as_saver, mark_as_loader

@register_checkpoint_hooks
class CategoricalEncoder:
    """
    A bidirectional encoder designed to map categorical text labels to continuous 
    numerical indices and vice versa. It supports unknown label fallback, sequence 
    encoding/decoding, and distributed data parallel (DDP) synchronized loading.
    """

    VALUE_SEPARATOR = " => "
    EXTRAS_SEPARATOR = "================\n"

    def __init__(
        self, 
        starting_index=0, 
        **special_labels
    ):
        """
        Initializes mapping dictionaries and registers optional configurations.

        Args:
            starting_index (int, optional): The lowest boundary integer to begin index mapping. Defaults to 0.
            **special_labels (Any): Key-value pairs defining specialized token targets (e.g., unk_label="<unk>").
        """

        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = starting_index
        self.handle_special_labels(special_labels)

    def handle_special_labels(self, special_labels):
        """Parses arguments to register a designated out-of-vocabulary (OOV) token."""

        if "unk_label" in special_labels: self.add_unk(special_labels["unk_label"])

    def __len__(self):
        """Returns total distinct mapped categories."""

        return len(self.lab2ind)

    @classmethod
    def from_saved(cls, path):
        """
        Instantiates the class and instantly populates entries from a flat text record file.

        Args:
            path (str): File path location containing saved key-to-index dictionaries.

        Returns:
            CategoricalEncoder: An initialized instance populated with the loaded dataset vocabulary.
        """

        obj = cls()
        obj.load(path)
        return obj

    def update_from_iterable(self, iterable, sequence_input=False):
        """
        Iterates over entries or flat sequences to dynamically expand vocabulary fields.

        Args:
            iterable (Iterable[Any]): Target items or sequence arrays to process.
            sequence_input (bool, optional): Treat the iterator as a collection of sub-sequences to be flattened. Defaults to False.
        """

        label_iterator = itertools.chain.from_iterable(iterable) if sequence_input else iter(iterable)
        for label in label_iterator:
            self.ensure_label(label)

    def update_from_didataset(self, didataset, output_key, sequence_input=False):
        """
        Extracts and parses structural dictionary labels from custom pipeline datasets.

        Args:
            didataset (Any): Object implementing contextual key restrictions.
            output_key (str): Targeted target data key holding item labels.
            sequence_input (bool, optional): Specifies if values are sequences requiring flattening. Defaults to False.
        """

        with didataset.output_keys_as([output_key]):
            self.update_from_iterable(
                (data_point[output_key] for data_point in didataset), 
                sequence_input=sequence_input
            )

    def limited_labelset_from_iterable(
        self, 
        iterable, 
        sequence_input=False, 
        n_most_common=None, 
        min_count=1
    ):
        """
        Limits label selection to the top tokens filtering by threshold and occurrence frequency.

        Args:
            iterable (Iterable[Any]): Elements or series containing data targets.
            sequence_input (bool, optional): Flattens iterable streams if explicitly set. Defaults to False.
            n_most_common (Optional[int], optional): Maximum cap limit for selecting elements. Defaults to None.
            min_count (int, optional): Minimum frequency bound cutoff. Defaults to 1.

        Returns:
            collections.Counter: Frequency records for the input elements.
        """

        label_iterator = itertools.chain.from_iterable(iterable) if sequence_input else iter(iterable)
        counts = collections.Counter(label_iterator)

        for label, count in counts.most_common(n_most_common):
            if count < min_count: break
            self.add_label(label)

        return counts

    def load_or_create(
        self, 
        path, 
        from_iterables=[], 
        from_didatasets=[], 
        sequence_input=False, 
        output_key=None, 
        special_labels={}
    ):
        """
        Attempts to load a dictionary configuration checkpoint. If it doesn't exist, 
        builds it from source iterables across synchronized DDP processes.

        Args:
            path (str): Intended target load or backup serialization file path.
            from_iterables (List[Iterable[Any]], optional): Source tracking objects. Defaults to [].
            from_didatasets (List[Any], optional): Structural database objects. Defaults to [].
            sequence_input (bool, optional): Sequence context handling flag. Defaults to False.
            output_key (Optional[str], optional): Targeted evaluation keys inside data dictionaries. Defaults to None.
            special_labels (Dict[str, Any], optional): Custom dynamic labels dictionary. Defaults to {}.
        """

        try:
            if if_main_process():
                if not self.load_if_possible(path):
                    for iterable in from_iterables:
                        self.update_from_iterable(iterable, sequence_input)

                    for didataset in from_didatasets:
                        if output_key is None: raise ValueError
                        self.update_from_didataset(didataset, output_key, sequence_input)

                    self.handle_special_labels(special_labels)
                    self.save(path)
        finally:
            # Block sub-processes until the main thread completes vocabulary build operations
            ddp_barrier()
            self.load(path)

    def add_label(self, label):
        """
        Registers a unique label, matching it against the next available integer index.

        Args:
            label (Any): Unique category name or label token.

        Returns:
            int: The newly assigned integer index.
        """

        if label in self.lab2ind: raise KeyError(f"Label '{label}' is already registered in the encoder mapping definitions.")
        index = self._next_index()

        self.lab2ind[label] = index
        self.ind2lab[index] = label

        return index

    def ensure_label(self, label):
        """Saves a category or passes back the index value if it already exists."""

        if label in self.lab2ind: return self.lab2ind[label]
        else: return self.add_label(label)

    def insert_label(self, label, index):
        """Inserts a specific category label at a fixed target index location."""

        if label in self.lab2ind: raise KeyError(f"Label '{label}' already exists.")
        else: self.enforce_label(label, index)

    def enforce_label(self, label, index):
        """Binds a specific label to an index, shifting competing records if collisions occur."""

        index = int(index)
        # Clear old indices matching this exact label string reference
        if label in self.lab2ind:
            if index == self.lab2ind[label]: return
            else: del self.ind2lab[self.lab2ind[label]]

        # Detect collisions on the target index
        if index in self.ind2lab:
            saved_label = self.ind2lab[index]
            moving_other = True
        else: moving_other = False

        self.lab2ind[label] = index
        self.ind2lab[index] = label

        # Resolve index collisions by migrating displaced labels to the next open index
        if moving_other:
            new_index = self._next_index()
            self.lab2ind[saved_label] = new_index
            self.ind2lab[new_index] = saved_label

    def add_unk(self, unk_label="<unk>"):
        """Registers an out-of-vocabulary fallback default token identifier."""

        self.unk_label = unk_label
        return self.add_label(unk_label)

    def _next_index(self):
        """Scans indexes sequentially to find the first unassigned slot."""

        index = self.starting_index
        while index in self.ind2lab:
            index += 1

        return index

    def is_continuous(self):
        """Checks if the index definitions contain holes or gaps."""

        indices = sorted(self.ind2lab.keys())
        return self.starting_index in indices and all(j - i == 1 for i, j in zip(indices[:-1], indices[1:]))

    def encode_label(self, label, allow_unk=True):
        """Transforms an item string/token into its corresponding index target integer."""

        self._assert_len()

        try:
            return self.lab2ind[label]
        except KeyError:
            if hasattr(self, "unk_label") and allow_unk: return self.lab2ind[self.unk_label]
            else: raise KeyError(f"Label '{label}' not found, and no unknown fallback behavior is enabled.")

    def encode_label_torch(self, label, allow_unk=True):
        """Wraps the encoded label integer inside a 1D PyTorch LongTensor."""

        return torch.LongTensor([self.encode_label(label, allow_unk)])

    def encode_sequence(self, sequence, allow_unk=True):
        """Encodes an array list of string labels into an array list of index integers."""

        self._assert_len()
        return [self.encode_label(label, allow_unk) for label in sequence]

    def encode_sequence_torch(self, sequence, allow_unk=True):
        """Converts an array list of sequence labels directly into a PyTorch LongTensor."""

        return torch.LongTensor([self.encode_label(label, allow_unk) for label in sequence])

    def decode_torch(self, x):
        """Recursively parses indices out of PyTorch multi-dimensional arrays back into labels."""

        self._assert_len()
        decoded = []

        if x.ndim == 1:  
            for element in x:
                decoded.append(self.ind2lab[int(element)])
        else:
            for subtensor in x:
                decoded.append(self.decode_torch(subtensor))

        return decoded

    def decode_ndim(self, x):
        """Recursively walks nested iterables or lists to reconstruct raw data labels."""

        self._assert_len()
        try:
            decoded = []
            for subtensor in x:
                decoded.append(self.decode_ndim(subtensor))

            return decoded
        except TypeError:  
            # Triggers when item cannot be iterated, treating it as a basic integer key
            return self.ind2lab[int(x)]

    @mark_as_saver
    def save(self, path):
        """Serializes current category dictionaries to disk."""

        self._save_literal(path, self.lab2ind, self._get_extras())

    def load(self, path):
        """Loads entries into storage, overwriting all current configuration bounds."""

        lab2ind, ind2lab, extras = self._load_literal(path)
        self.lab2ind = lab2ind
        self.ind2lab = ind2lab
        self._set_extras(extras)

    @mark_as_loader
    def load_if_possible(self, path, end_of_epoch=False):
        """Gracefully loads checkpoints if they exist without breaking runtime workflows."""

        del end_of_epoch

        try:
            self.load(path)
        except FileNotFoundError:
            return False
        except (ValueError, SyntaxError):
            return False
        
        return True 

    def expect_len(self, expected_len):
        """Sets an expected vocabulary size constraint for runtime validation."""

        self.expected_len = expected_len

    def ignore_len(self):
        """Disables vocabulary size verification constraints."""

        self.expected_len = None

    def _assert_len(self):
        """Validates that current state sizing matches requested expectations."""

        if hasattr(self, "expected_len"):
            if self.expected_len is None: return
            if len(self) != self.expected_len: raise RuntimeError(f"Size mismatch: Expected {self.expected_len} items, got {len(self)}.")
        else:
            self.ignore_len()
            return

    def _get_extras(self):
        """Bundles supplementary structural class attributes for serialization processing."""

        extras = {"starting_index": self.starting_index}
        if hasattr(self, "unk_label"): extras["unk_label"] = self.unk_label

        return extras

    def _set_extras(self, extras):
        """Restores structural operational states from unmarshalled attributes dictionaries."""

        if "unk_label" in extras: self.unk_label = extras["unk_label"]
        self.starting_index = extras["starting_index"]

    @staticmethod
    def _save_literal(path, lab2ind, extras):
        """Writes plain text data structures with human-readable separators."""

        with open(path, "w", encoding="utf-8") as f:
            for label, ind in lab2ind.items():
                f.write(repr(label) + CategoricalEncoder.VALUE_SEPARATOR + str(ind) + "\n")

            f.write(CategoricalEncoder.EXTRAS_SEPARATOR)

            for key, value in extras.items():
                f.write(repr(key) + CategoricalEncoder.VALUE_SEPARATOR + repr(value) + "\n")

            f.flush()

    @staticmethod
    def _load_literal(path):
        """Parses custom flat map texts containing standard serialized Python literals."""

        lab2ind, ind2lab, extras = {}, {}, {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line == CategoricalEncoder.EXTRAS_SEPARATOR: break

                literal, ind = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR, 
                    maxsplit=1
                )

                label = ast.literal_eval(literal)
                lab2ind[label] = int(ind)
                ind2lab[ind] = label # Cast index as int to fix lookups in downstream decoders

            for line in f:
                literal_key, literal_value = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR, 
                    maxsplit=1
                )

                extras[ast.literal_eval(literal_key)] = ast.literal_eval(literal_value)
                
        return lab2ind, ind2lab, extras