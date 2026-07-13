import os
import sys
import inspect

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import fetch, run_on_main
from main.library.speaker_diarization.features import DEFAULT_TRANSFER_HOOKS, DEFAULT_LOAD_HOOKS

def get_default_hook(obj, default_hooks):
    """
    Resolves the appropriate handling hook function based on the object's Method Resolution Order (MRO).

    Args:
        obj (Any): Target object instance (e.g., nn.Module, Optimizer) to look up.
        default_hooks (Dict[Any, Callable]): Dictionary mapping class types to hook functions.

    Returns:
        Optional[Callable]: The resolved callback hook function if matched, otherwise None.
    """

    for cls in inspect.getmro(type(obj)):
        if cls in default_hooks: return default_hooks[cls]
        
    return None

class Pretrainer:
    """
    Manages downloading, local caching, and flexible state restoration for pre-trained models.

    Supports custom registration mapping paths, runtime evaluation filters, and hierarchical hook bindings.
    """

    def __init__(
        self, 
        loadables=None, 
        paths=None, 
        custom_hooks=None, 
        conditions=None
    ):
        """Initializes the Pretrainer system registries."""

        self.loadables = {}

        if loadables is not None: self.add_loadables(loadables)
        self.paths = {}

        if paths is not None: self.add_paths(paths)
        self.custom_hooks = {}

        if custom_hooks is not None: self.add_custom_hooks(custom_hooks)
        self.conditions = {}

        if conditions is not None: self.add_conditions(conditions)
        self.is_local = []

    def add_loadables(self, loadables):
        """Registers objects targeting weight restorations (e.g., models, optimizers)."""

        self.loadables.update(loadables)

    def add_paths(self, paths):
        """Maps loadable keys directly to their respective remote or local asset file locations."""

        self.paths.update(paths)

    def add_custom_hooks(self, custom_hooks):
        """Binds customized standalone logic callback handlers targeting unique recovery objects."""

        self.custom_hooks.update(custom_hooks)

    def add_conditions(self, conditions):
        """Sets conditional switches or dynamic callbacks deciding whether a key should be loaded."""

        self.conditions.update(conditions)

    @staticmethod
    def split_path(path):
        """
        Splits a full URI or file system path into a separate source root and file name target.

        Args:
            path (str): The complete resource identifier path string.

        Returns:
            Tuple[str, str]: A pair containing (source_directory_or_url, filename).
        """

        def split(src):
            if "/" in src: return src.rsplit("/", maxsplit=1)
            else: return "./", src

        return split(path)

    def collect_files(self, default_source=None):
        """
        Downloads or resolves the local cache file path for all active registered loadables.

        Ensures network fetching triggers on the main process thread first in distributed configurations.

        Args:
            default_source (Optional[str], optional): Fallback remote URL or directory root path. Defaults to None.

        Returns:
            Dict[str, str]: Resolved local file system path mappings.
        """

        loadable_paths = {}
        for name in self.loadables:
            if not self.is_loadable(name): continue
            save_filename = name + ".ckpt"

            if name in self.paths: 
                source, filename = self.split_path(
                    self.paths[name]
                )
            elif default_source is not None:
                filename = save_filename
                source = default_source
            else: raise ValueError(f"Failed to resolve fetch location for '{name}'. Provide an explicit path or define a default source route fallback.")

            fetch_kwargs = {
                "filename": filename, 
                "source": source
            }
            path = None

            def run_fetch(**kwargs):
                nonlocal path

                path = fetch(**kwargs)

            # Block multi-GPU worker initialization race conditions by querying solely via the master thread root
            run_on_main(
                run_fetch, 
                kwargs=fetch_kwargs, 
                post_func=run_fetch, 
                post_kwargs=fetch_kwargs
            )

            loadable_paths[name] = path
            self.paths[name] = str(path)
            self.is_local.append(name)

        return loadable_paths

    def is_loadable(self, name):
        """
        Evaluates whether a loadable component satisfies its defined structural conditions.

        Args:
            name (str): Key matching targeted parameter objects.
        """

        if name not in self.conditions: return True
        condition = self.conditions[name]

        if callable(condition): return condition()
        else: return bool(condition)

    def load_collected(self):
        """Triggers local file mapping verification before feeding components into structural loading hooks."""

        paramfiles = {}
        for name in self.loadables:
            if not self.is_loadable(name): continue

            if name in self.is_local: paramfiles[name] = self.paths[name]
            else: raise ValueError(f"Component file target '{name}' must be collected locally before running state loads.")

        self._call_load_hooks(paramfiles)

    def _call_load_hooks(self, paramfiles):
        """
        Dispatches parameters restoration targets to their best matching functional hooks pipeline.

        Prioritizes custom user-defined hooks, falls back to loose parameter transfers, then 
        standard system checkpoint recovery mechanisms, and errors out if no matching routine is discovered.
        """

        for name, obj in self.loadables.items():
            if not self.is_loadable(name): continue
            loadpath = paramfiles[name]

            # Priority 1: User defined standalone customized callback injections
            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath)
                continue

            # Priority 2: Safe partial state transfer bindings (e.g., fine-tuning partial layers matches)
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            
            if default_hook is not None:
                default_hook(obj, loadpath)
                continue

            # Priority 3: Standard matching layout checkpoint recovery loaders routines
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)

            if default_hook is not None:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch)
                continue

            raise RuntimeError(f"No compatible parameter loading hook mechanism found for object '{name}' of class type: {type(obj)}.")