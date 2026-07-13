import os
import copy
import hashlib
import subprocess

from collections import namedtuple
from collections.abc import Iterable

# Named tuple representing an edge in the Directed Acyclic Graph (DAG)
DagEdge = namedtuple(
    'DagEdge',
    [
        'downstream_node',
        'downstream_label',
        'upstream_node',
        'upstream_label',
        'upstream_selector',
    ],
)

class KwargReprNode:
    """A DAG node that is identified and hashed based on its arguments and upstream connections."""

    def __init__(self, incoming_edge_map, name, args=None, kwargs=None):
        """Initialize a KwargReprNode.

        Args:
            incoming_edge_map (dict): Mapping of downstream labels to upstream node connections.
            name (str): Name of the node operation.
            args (list, optional): Positional arguments. Defaults to None.
            kwargs (dict, optional): Keyword arguments. Defaults to None.
        """

        self._incoming_edge_map = incoming_edge_map
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}
        # The hash is pre-computed during initialization since nodes are immutable components of the graph
        self._hash = self._build_hash()

    def _get_upstream_hashes(self):
        """Collect hash components from all incoming edges to maintain structural integrity."""

        hashes = []
        # Iterate through every dependency to ensure any upstream change updates this node's identity
        for downstream_label, upstream_info in self.incoming_edge_map.items():
            upstream_node, upstream_label, upstream_selector = upstream_info
            # Collect unique identity aspects of the incoming connection
            hashes.extend([
                hash(downstream_label),
                hash(upstream_node),
                hash(upstream_label),
                hash(upstream_selector),
            ])

        return hashes

    def _get_inner_hash(self):
        """Generate a hash component from the local arguments and keyword arguments."""

        return get_hash({
            'args': self.args,
            'kwargs': self.kwargs,
        })

    def _build_hash(self):
        """Combine upstream dependency hashes and internal attribute hashes into a single unique integer."""

        hashes = self._get_upstream_hashes()
        hashes.append(self._get_inner_hash())

        return get_hash_int(hashes) # Convert the structural MD5 hex string into a native Python integer

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # Two nodes are identical if they belong to the same class and have matching graph-dependency hashes
        return isinstance(other, self.__class__) and hash(self) == hash(other)

    def __repr__(self):
        return self.long_repr()

    @property
    def incoming_edge_map(self):
        return self._incoming_edge_map

    @property
    def incoming_edges(self):
        """Get a list of DagEdge structures representing connections leading into this node."""

        return get_incoming_edges(self, self.incoming_edge_map)

    @property
    def short_hash(self):
        """Return a 12-character hex string representing the absolute hash of the node."""

        return format(abs(hash(self)), 'x')[:12]

    @property
    def short_repr(self):
        return self.name

    def long_repr(self, include_hash=True):
        """
        Format a detailed signature of the node, resembling a Python function call.

        Args:
            include_hash (bool): Whether to append the unique short hash token.
        """
        # Convert positional arguments into their standard string representations
        formatted = [repr(arg) for arg in self.args]

        # Order keyword arguments alphabetically to prevent hash mismatches caused by key shuffling
        formatted.extend(
            '{}={!r}'.format(key, self.kwargs[key])
            for key in sorted(self.kwargs)
        )

        # Build function-style string: e.g., "vflip(input_0)"
        result = '{}({})'.format(
            self.name,
            ', '.join(formatted),
        )

        if include_hash: # Append unique hash signature tracking when requested (useful for debugging complex graph routing)
            result += ' <{}>'.format(self.short_hash)

        return result

class Stream:
    """Represents a data wrapper/reference originating from a specific outlet (label) of a Node."""

    def __init__(
        self,
        upstream_node,
        upstream_label,
        node_types,
        upstream_selector=None,
    ):
        """
        Initialize a Stream object.

        Raises:
            TypeError: If the upstream node does not match the valid allowed types.
        """
        
        # Strict validation ensuring streams only originate from permitted nodes
        if not is_of_types(upstream_node, node_types): raise TypeError(f"Upstream node must be an instance of {node_types}, got {type(upstream_node).__name__}.")

        self.node = upstream_node
        self.label = upstream_label
        self.selector = upstream_selector

    def __hash__(self):
        return get_hash_int([
            hash(self.node),
            hash(self.label),
        ])

    def __eq__(self, other):
        return isinstance(other, Stream) and hash(self) == hash(other)

    def __repr__(self):
        # Exclude hash from internal node signature to keep output human-readable
        node_repr = self.node.long_repr(include_hash=False)

        # Append optional selectors like stream specifiers (e.g., ':v' or ':a') if active
        selector = ''
        if self.selector:
            selector = ':{}'.format(self.selector)

        return '{}[{!r}{}] <{}>'.format(
            node_repr,
            self.label,
            selector,
            self.node.short_hash,
        )

    def __getitem__(self, index):
        """
        Sub-select or extract parts of a stream (e.g., audio/video selector string).

        Raises:
            ValueError: If a selector option is already present on this stream.
            TypeError: If the index specifier is not a string type.
        """

        # Prevent chaining selectors twice (e.g., stream['v']['a'])
        if self.selector is not None: raise ValueError(f"Stream already has a selector pattern applied: {self.selector!r}.")
        # Bracket operators require a string parameter identifier
        if not isinstance(index, str): raise TypeError(f"Stream selector key must be a string descriptor, not {type(index).__name__}.")

        # Request the parent node to output a modified stream view containing the selector pattern
        return self.node.stream(
            label=self.label,
            selector=index,
        )

    @property
    def audio(self):
        """Shortcut property to select the audio layer ('a') of the current stream."""

        return self['a']

    @property
    def video(self):
        """Shortcut property to select the video layer ('v') of the current stream."""

        return self['v']

class Node(KwargReprNode):
    """An operation node within the DAG capable of verifying input thresholds and stream mappings."""

    @classmethod
    def check_input_len(cls, stream_map, min_inputs, max_inputs):
        """
        Validate whether the quantity of provided streams falls within acceptable bounds.

        Raises:
            ValueError: If the total inputs violate minimum or maximum boundaries.
        """

        stream_count = len(stream_map)

        # Check against lower bounds if configured
        if min_inputs is not None and stream_count < min_inputs: raise ValueError(f"Insufficient input streams provided. Expected at least {min_inputs}, got {stream_count}.")
        # Check against upper bounds if configured
        if max_inputs is not None and stream_count > max_inputs: raise ValueError(f"Too many input streams provided. Expected at most {max_inputs}, got {stream_count}.")

    @classmethod
    def check_input_types(cls, stream_map, incoming_stream_types):
        """
        Ensure all incoming streams match the target schema classifications.

        Raises:
            TypeError: If any input stream encounters a type validation failure.
        """

        for stream in stream_map.values():
            if not is_of_types(stream, incoming_stream_types): raise TypeError(f"Invalid stream type detected: {type(stream).__name__}. Expected one of: {incoming_stream_types}.")

    @classmethod
    def get_incoming_edge_map(cls, stream_map):
        """Convert a stream map configuration dictionary into a functional graph structure schema."""

        edge_map = {}
        # Deconstruct Stream object metadata to formulate direct dictionary connections
        for downstream_label, upstream in stream_map.items():
            edge_map[downstream_label] = (
                upstream.node,
                upstream.label,
                upstream.selector,
            )

        return edge_map

    def __init__(
        self,
        stream_spec,
        name,
        incoming_stream_types,
        outgoing_stream_type,
        min_inputs,
        max_inputs,
        args=None,
        kwargs=None,
    ):
        """Initialize a Node component while processing validation and parent setups."""

        self._outgoing_stream_type = outgoing_stream_type
        # Parse unstructured stream collections (lists, items, dicts) into explicit mappings
        stream_map = get_stream_map(stream_spec)

        # Perform topological sanity checks before instantiating node structures
        self.check_input_len(
            stream_map,
            min_inputs,
            max_inputs,
        )

        self.check_input_types(
            stream_map,
            incoming_stream_types,
        )

        # Initialize base KwargReprNode properties using the parsed incoming edge mapping configuration
        super().__init__(
            self.get_incoming_edge_map(stream_map),
            name,
            args=args,
            kwargs=kwargs,
        )

    def stream(self, label=None, selector=None):
        """Instantiate a structured downstream wrapper linked directly to this specific operational node."""

        return self._outgoing_stream_type(
            self,
            label,
            upstream_selector=selector,
        )

    def __getitem__(self, item):
        """Enable clean bracket slicing notation to fetch structured internal stream segments."""
    
        if isinstance(item, slice):
            return self.stream(
                label=item.start, # The 'start' token tracks the label index identifier
                selector=item.stop, # The 'stop' token tracks optional selector variants
            )

        return self.stream(label=item)

class FilterableStream(Stream):
    """A stream variant specifically designated for feeding into multimedia processing filters."""

    def __init__(
        self,
        upstream_node,
        upstream_label,
        upstream_selector=None,
    ):
        # Filterable streams can only originate from file inputs or existing filter blocks
        super().__init__(
            upstream_node,
            upstream_label,
            {InputNode, FilterNode},
            upstream_selector,
        )

class InputNode(Node):
    """Represents a primary external data source entry node (such as media files) in the graph."""

    def __init__(self, name, args=None, kwargs=None):
        super().__init__(
            stream_spec=None, # Inputs have no prior parent streams inside the DAG layout
            name=name,
            incoming_stream_types={},
            outgoing_stream_type=FilterableStream,
            min_inputs=0,
            max_inputs=0,
            args=args,
            kwargs=kwargs,
        )

    @property
    def short_repr(self):
        # Extract the underlying base filename for easy visual log inspections
        return os.path.basename(self.kwargs['filename'])

class FilterNode(Node):
    """Represents an processing filter stage modification inside the multimedia graph execution path."""

    def __init__(
        self,
        stream_spec,
        name,
        max_inputs=1,
        args=None,
        kwargs=None,
    ):
        super().__init__(
            stream_spec=stream_spec,
            name=name,
            incoming_stream_types={FilterableStream},
            outgoing_stream_type=FilterableStream,
            min_inputs=1,
            max_inputs=max_inputs,
            args=args,
            kwargs=kwargs,
        )

    def _get_filter(self, outgoing_edges):
        """Generate a formatted string string representing this filter component for CLI flags."""

        args = self.args
        kwargs = self.kwargs

        # Special handling for ffmpeg's split filters: the output count dictates its main parameter argument
        if self.name in ('split', 'asplit'):
            args = [len(outgoing_edges)]

        # Escape key/value pairings to ensure robust parsing by ffmpeg's complex graph layout engine
        escaped_args = [
            escape_chars(value, "\\'=:")
            for value in args
        ]

        # Escape key/value pairings to ensure robust parsing by ffmpeg's complex graph layout engine
        escaped_kwargs = {}
        for key, value in kwargs.items():
            escaped_kwargs[
                escape_chars(key, "\\'=:")
            ] = escape_chars(value, "\\'=:")

        params = []
        # Merge positional components into the main parameter collection
        params.extend(
            escape_chars(value, "\\'=:")
            for value in escaped_args
        )

        # Merge and sort keyword components into key=value strings
        params.extend(
            '{}={}'.format(key, escaped_kwargs[key])
            for key in sorted(escaped_kwargs)
        )

        # Construct the final initialization flag: name=param1:param2...
        result = escape_chars(self.name, "\\'=:")
        if params: result += '={}'.format(':'.join(params))

        # Perform comprehensive terminal-safe escaping for final graph injections
        return escape_chars(result, "\\'[],;")

class OutputNode(Node):
    """An end-point node indicating file target streams or rendering locations within the active graph."""

    def __init__(self, stream, name, args=None, kwargs=None):
        super().__init__(
            stream_spec=stream,
            name=name,
            incoming_stream_types={FilterableStream},
            outgoing_stream_type=OutputStream,
            min_inputs=1,
            max_inputs=None,
            args=args,
            kwargs=kwargs,
        )

    @property
    def short_repr(self):
        return os.path.basename(self.kwargs['filename'])

class OutputStream(Stream):
    """A terminal stream subtype tracking routing structures into endpoint configurations."""

    def __init__(
        self,
        upstream_node,
        upstream_label,
        upstream_selector=None,
    ):
        # Terminal streams route towards rendering components, global configurations, or merging adapters
        super().__init__(
            upstream_node,
            upstream_label,
            {OutputNode, GlobalNode, MergeOutputsNode},
            upstream_selector,
        )

class MergeOutputsNode(Node):
    """A specialized pipeline component combining an arbitrary cluster of distinct Output streams."""

    def __init__(self, streams, name):
        super().__init__(
            stream_spec=streams,
            name=name,
            incoming_stream_types={OutputStream},
            outgoing_stream_type=OutputStream,
            min_inputs=1,
            max_inputs=None,
        )

class GlobalNode(Node):
    """Applies global-scope flag directives targeting runtime execution blocks."""

    def __init__(self, stream, name, args=None, kwargs=None):
        super().__init__(
            stream_spec=stream,
            name=name,
            incoming_stream_types={OutputStream},
            outgoing_stream_type=OutputStream,
            min_inputs=1,
            max_inputs=1,
            args=args,
            kwargs=kwargs,
        )

def get_incoming_edges(downstream_node, incoming_edge_map):
    """Build structural DagEdge instances out of basic dictionary tracking formats."""

    edges = []
    for downstream_label, upstream_info in incoming_edge_map.items():
        upstream_node, upstream_label, upstream_selector = upstream_info
        # Convert decoupled tracking items into functional immutable namedtuple objects
        edges.append(
            DagEdge(
                downstream_node,
                downstream_label,
                upstream_node,
                upstream_label,
                upstream_selector,
            )
        )

    return edges

def get_outgoing_edges(upstream_node, outgoing_edge_map):
    """Synthesize downstream out-edge definitions derived from topological tracking indices."""

    edges = []
    # Iterate sorting key tags to maintain absolute consistency during parameter compiling
    for upstream_label, downstream_infos in sorted(outgoing_edge_map.items()):
        for downstream_info in downstream_infos:
            downstream_node, downstream_label, downstream_selector = downstream_info

            edges.append(
                DagEdge(
                    downstream_node,
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    downstream_selector # Mapped as downstream target variant selector reference
                )
            )

    return edges

def is_of_types(obj, types):
    """Validate if target object instances fit within custom typing collections."""

    return any(isinstance(obj, item) for item in types)

def get_stream_map(stream_spec):
    """
    Normalize input definitions, lists, or mapping tables into a reliable, uniform dictionary structure.

    Raises:
        TypeError: If the provided stream specifier type is unmappable.
    """

    if stream_spec is None:
        return {}

    # Wrap a standalone Stream into a default unified single-item dictionary mapping
    if isinstance(stream_spec, Stream):
        return {None: stream_spec}

    # Map sequential index listings directly to integer lookup definitions
    if isinstance(stream_spec, (list, tuple)):
        return dict(enumerate(stream_spec))

    # Return valid mappings directly
    if isinstance(stream_spec, dict):
        return stream_spec

    raise TypeError(f"Cannot parse stream specification from type: {type(stream_spec).__name__}.")

def get_stream_map_nodes(stream_map):
    """
    Extract operational core nodes from processed uniform stream structures.

    Raises:
        TypeError: If an element within the stream tracking map is not a valid Stream.
    """

    nodes = []
    for stream in stream_map.values():
        if not isinstance(stream, Stream):
            raise TypeError(f"Expected a Stream instance inside stream map, found: {type(stream).__name__}.")

        nodes.append(stream.node)

    return nodes

def get_stream_spec_nodes(stream_spec):
    """High-level translation parsing general input targets directly into a sequence of source nodes."""

    return get_stream_map_nodes(
        get_stream_map(stream_spec)
    )

def stream_operator(stream_classes=None, name=None):
    """Decorator factory helper to bind specific operations directly into targeted Stream instances."""

    stream_classes = stream_classes or {Stream}

    def decorator(func):
        # Override target context names using custom names when specified
        func_name = name or func.__name__
        # Dynamically attach the utility function method onto the specified Stream target structures
        for stream_class in stream_classes:
            setattr(stream_class, func_name, func)

        return func

    return decorator

def filter_operator(name=None):
    """Decorator targeting FilterableStream objects exclusively."""

    return stream_operator(
        stream_classes={FilterableStream},
        name=name,
    )

def output_operator(name=None):
    """Decorator targeting OutputStream objects exclusively."""

    return stream_operator(
        stream_classes={OutputStream},
        name=name,
    )

def input(filename, **kwargs):
    """
    Generate a clean, standardized multimedia file entry input node structure.

    Raises:
        ValueError: If both 'f' alias and 'format' keyword are provided.
    """

    kwargs['filename'] = filename
    # Extract short-hand format descriptors (e.g., input('file.mp4', f='mp4'))
    fmt = kwargs.pop('f', None)

    if fmt:
        # Enforce consistency to avoid colliding options
        if 'format' in kwargs: raise ValueError("Ambiguous formatting options: both 'f' alias and 'format' key were supplied.")
        kwargs['format'] = fmt

    # Return the open streaming line reference pointing to this source node instance
    return InputNode(
        input.__name__,
        kwargs=kwargs,
    ).stream()

@output_operator()
def global_args(stream, *args):
    """Inject unstructured custom parameter flags onto the parent multimedia stream target."""

    return GlobalNode(
        stream,
        global_args.__name__,
        args,
    ).stream()

@output_operator()
def overwrite_output(stream):
    """Append the implicit '-y' execution instruction to override present file paths."""

    return GlobalNode(
        stream,
        overwrite_output.__name__,
        ['-y'],
    ).stream()

@output_operator()
def merge_outputs(*streams):
    """Aggregate multi-stream branches back into a single operational interface node context."""

    return MergeOutputsNode(
        streams,
        merge_outputs.__name__,
    ).stream()

@filter_operator()
def output(*streams_and_filename, **kwargs):
    """
    Compile distinct processed layers onto an absolute file location definition.

    Raises:
        ValueError: If filename arguments are missing or invalid, or if file format options collide.
    """

    streams_and_filename = list(streams_and_filename)
    # Pop file strings from trailing argument values if a named filename kwarg is missing
    if 'filename' not in kwargs:
        if not isinstance(streams_and_filename[-1], str): raise ValueError("Output node declaration requires a destination file path string.")
        kwargs['filename'] = streams_and_filename.pop()

    fmt = kwargs.pop('f', None)

    if fmt:
        if 'format' in kwargs: raise ValueError("Ambiguous formatting options: both 'f' alias and 'format' key were supplied.")
        kwargs['format'] = fmt

    return OutputNode(
        streams_and_filename,
        output.__name__,
        kwargs=kwargs,
    ).stream()

def escape_chars(text, chars):
    """Cleanse runtime specific parameter string strings by prepending escape sequences."""

    text = str(text)
    # De-duplicate character targets
    chars = list(set(chars))

    # Backslashes must be escaped first before injecting additions to prevent recursive looping bugs
    if '\\' in chars:
        chars.remove('\\')
        chars.insert(0, '\\')

    for char in chars:
        text = text.replace(char, '\\' + char)

    return text

def recursive_repr(item):
    """Recursively formats structures into deterministic strings suitable for reliable hashing pipelines."""

    if isinstance(item, str):
        return item

    # Resolve arrays iteratively
    if isinstance(item, list):
        return '[{}]'.format(
            ', '.join(recursive_repr(x) for x in item)
        )

    # Alphabetize lookup dictionaries to keep state evaluations reliable
    if isinstance(item, dict):
        return '{{{}}}'.format(
            ', '.join(
                '{}: {}'.format(
                    recursive_repr(key),
                    recursive_repr(item[key]),
                )
                for key in sorted(item)
            )
        )

    return repr(item)

def get_hash(item):
    """Generate a unique MD5 hex signature for a python component structural state tracking layout."""

    value = recursive_repr(item).encode('utf-8')
    return hashlib.md5(value).hexdigest()

def get_hash_int(item):
    """Generate an integer-based signature equivalent extracted out of base hex data frames."""

    return int(get_hash(item), 16)

@output_operator()
def run(
    stream_spec,
    cmd='ffmpeg',
    capture_stdout=False,
    capture_stderr=False,
    input=None,
    quiet=False,
    overwrite_output=False,
    cwd=None,
):
    """
    Block-execute a multimedia stream graph directly within host system process utilities.

    Raises:
        RuntimeError: If the process tracking encounters non-zero error execution states.
    """

    # Spawns process structures using internal async engine setups
    process = run_async(
        stream_spec,
        cmd=cmd,
        pipe_stdin=input is not None,
        pipe_stdout=capture_stdout,
        pipe_stderr=capture_stderr,
        quiet=quiet,
        overwrite_output=overwrite_output,
        cwd=cwd,
    )

    # Perform synchronous block reads capturing standard runtime buffers
    out, err = process.communicate(input)
    retcode = process.poll()

    if retcode: raise RuntimeError(f"Process execution failed with exit code {retcode}. Error details: {err.decode() if err else ''}")
    return out, err

@output_operator()
def run_async(
    stream_spec,
    cmd='ffmpeg',
    pipe_stdin=False,
    pipe_stdout=False,
    pipe_stderr=False,
    quiet=False,
    overwrite_output=False,
    cwd=None
):
    """Spawns an asynchronous sub-process engine layer without blocking program execution paths."""

    # Build out final list array arguments mapping the whole graph setup
    args = compile(stream_spec, cmd=cmd, overwrite_output=overwrite_output)
    # Map target standard descriptor piping flags
    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout else None
    stderr_stream = subprocess.PIPE if pipe_stderr else None

    if quiet: # Redirect logs into internal blackholes when quiet mode is enabled
        stdout_stream = subprocess.DEVNULL
        stderr_stream = subprocess.STDOUT

    return subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream, cwd=cwd)

@output_operator()
def compile(stream_spec, cmd='ffmpeg', overwrite_output=False):
    """Translate active graph connections directly into valid terminal terminal array directives."""

    # Ensure command string definitions adapt into uniform lists
    if isinstance(cmd, str): cmd = [cmd]
    elif not isinstance(cmd, list): cmd = list(cmd)

    return cmd + get_args(stream_spec, overwrite_output=overwrite_output)

@output_operator()
def get_args(stream_spec, overwrite_output=False):
    """Construct full list configurations mapping out topological dependency flags sequentially."""

    nodes = get_stream_spec_nodes(stream_spec)
    # Sort active nodes via deep-first sorting to trace dependencies accurately
    sorted_nodes, outgoing_edge_maps = topo_sort(nodes)
    args, input_nodes, output_nodes, global_nodes, filter_nodes = [], [], [], [], []
    # Distribute heterogeneous node entries into specific classification lists
    for node in sorted_nodes:
        if isinstance(node, InputNode): input_nodes.append(node)
        elif isinstance(node, OutputNode): output_nodes.append(node)
        elif isinstance(node, GlobalNode): global_nodes.append(node)
        elif isinstance(node, FilterNode): filter_nodes.append(node)

    # Auto-index primitive file entry node names map sequentially: (node, None) -> "0", "1", ...
    stream_name_map = {(node, None): str(index) for index, node in enumerate(input_nodes)}
    for node in input_nodes:
        args.extend(get_input_args(node))

    # Process and build internal complex filter instructions string blocks
    filter_arg = get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map)
    if filter_arg: args.extend(['-filter_complex', filter_arg])

    # Append downstream export definitions
    for node in output_nodes:
        args.extend(get_output_args(node, stream_name_map))

    # Append final global flag directives
    for node in global_nodes:
        args.extend(list(node.args))

    # Inject global overwrite flags directly at the end when requested
    if overwrite_output: args.append('-y')
    return args

def get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map):
    """Assemble complex multiline filter graph configurations safely merged with semicolon delimiters."""

    allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map)
    return ';'.join([get_filter_spec(node, outgoing_edge_maps[node], stream_name_map) for node in filter_nodes])

def allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map):
    """
    Sequentially assign distinct alphanumeric labels (e.g., [s0]) onto multi-branch outputs.

    Raises:
        ValueError: If an individual branch breaks validation by binding duplicate edge rules.
    """

    stream_count = 0

    for upstream_node in filter_nodes:
        outgoing_edge_map = outgoing_edge_maps[upstream_node]

        for upstream_label, downstreams in sorted(outgoing_edge_map.items()):
            # If multiple targets request a label without an explicit split filter, raise an issue
            if len(downstreams) > 1: raise ValueError(f"Stream split conflict: Label {upstream_label!r} of node {upstream_node.name} is mapped to multiple downstream endpoints without an explicit split filter.")

            # Assign internal stream identity tags sequentially (e.g., s0, s1, s2...)
            stream_name_map[(upstream_node, upstream_label)] = 's{}'.format(stream_count)
            stream_count += 1

def get_filter_spec(node, outgoing_edge_map, stream_name_map):
    """Format inputs, operations, and intermediate targets into structural filter specification tokens."""

    incoming_edges = node.incoming_edges
    outgoing_edges = get_outgoing_edges(node, outgoing_edge_map)

    inputs = [format_input_stream_name(stream_name_map, edge) for edge in incoming_edges] # Format incoming tokens: e.g., "[0:v]" or "[s0]"
    outputs = [format_output_stream_name(stream_name_map, edge) for edge in outgoing_edges] # Format destination tokens: e.g., "[s1]"

    return '{}{}{}'.format(''.join(inputs), node._get_filter(outgoing_edges), ''.join(outputs))

def get_input_args(input_node):
    """
    Generate CLI command-line parameters dedicated to initializing specific source files.

    Raises:
        ValueError: If the current operational node structure type mismatches input specifications.
    """

    if input_node.name != input.__name__: raise ValueError(f"Expected input execution context block, but encountered node: {input_node.name}")

    kwargs = copy.copy(input_node.kwargs)
    filename = kwargs.pop('filename')
    fmt = kwargs.pop('format', None)
    video_size = kwargs.pop('video_size', None)
    args = []

    # Map format flags sequentially
    if fmt: args.extend(['-f', fmt])
    # Map video resolution array tuples down into string formats
    if video_size: args.extend(['-video_size', '{}x{}'.format(video_size[0], video_size[1])])

    # Convert auxiliary parameters into standard command options
    args.extend(convert_kwargs_to_cmd_line_args(kwargs))
    # Target file paths are always introduced after their respective metadata modifiers
    args.extend(['-i', filename])

    return args

def get_output_args(node, stream_name_map):
    """
    Generate final execution mappings and specific compression/bitrate arguments targeting outputs.

    Raises:
        ValueError: If node schemas violate expected conventions or lose input connections entirely.
    """

    if node.name != output.__name__: raise ValueError(f"Expected output execution context block, but encountered node: {node.name}")
    if not node.incoming_edges: raise ValueError(f"Output node {node.long_repr()} has no valid incoming streams to export.")

    args = []
    # Compile stream routing definitions into explicit map declarations
    for edge in node.incoming_edges:
        stream_name = format_input_stream_name(stream_name_map, edge, is_final_arg=True)
        # Avoid explicit mapping declarations for plain standalone root streams
        if stream_name != '0' or len(node.incoming_edges) > 1: args.extend(['-map', stream_name])

    kwargs = copy.copy(node.kwargs)
    filename = kwargs.pop('filename')

    # Convert generic keywords into explicit formatting keys
    if 'format' in kwargs: args.extend(['-f', kwargs.pop('format')])
    if 'video_bitrate' in kwargs: args.extend(['-b:v', str(kwargs.pop('video_bitrate'))])
    if 'audio_bitrate' in kwargs: args.extend(['-b:a', str(kwargs.pop('audio_bitrate'))])
    # Process resolution parameters
    if 'video_size' in kwargs:
        video_size = kwargs.pop('video_size')
        # Map structural arrays down into size tokens
        if not isinstance(video_size, str) and isinstance(video_size, Iterable): video_size = '{}x{}'.format(video_size[0], video_size[1])

        args.extend(['-video_size', video_size])

    # Unroll remaining kwargs configurations
    args.extend(convert_kwargs_to_cmd_line_args(kwargs))
    args.append(filename)

    return args

def convert_kwargs_to_cmd_line_args(kwargs):
    """Convert arbitrary key-value pairs into explicit terminal hyphen prefix flags."""

    args = []

    for key in sorted(kwargs):
        value = kwargs[key]
        # Handle list/iterable argument sets mapped under identical flag names
        if isinstance(value, Iterable) and not isinstance(value, str):
            for item in value:
                args.append('-{}'.format(key))
                if item is not None: args.append(str(item))

            continue

        # Map basic standard parameters
        args.append('-{}'.format(key))
        if value is not None: args.append(str(value))

    return args

def format_input_stream_name(stream_name_map, edge, is_final_arg=False):
    """Translate edge data maps into exact bracket syntax indices required by the terminal engine execution."""

    prefix = stream_name_map[(edge.upstream_node, edge.upstream_label)]
    suffix = ''

    # Append optional sub-selectors like stream indices or tracks
    if edge.upstream_selector: suffix = ':{}'.format(edge.upstream_selector)
    # Bare unbracketed stream tokens are preferred for raw mapped input roots
    if is_final_arg and isinstance(edge.upstream_node, InputNode): return '{}{}'.format(prefix, suffix)

    # Standard stream tokens require bracket enclosures inside complex filters
    return '[{}{}]'.format(prefix, suffix)

def format_output_stream_name(stream_name_map, edge):
    """Format downstream reference identifiers for use within complex filter strings."""

    return '[{}]'.format(stream_name_map[(edge.upstream_node, edge.upstream_label)])

def topo_sort(downstream_nodes):
    """
    Perform a depth-first search Topological Sort to establish execution order.

    Raises:
        RuntimeError: If a cyclic graph relationship (loop) is identified.
    """

    marked_nodes = set()  # Tracks nodes currently on the active stack path to spot cycles
    sorted_nodes = []     # Stores the final resolved execution stack sequentially
    outgoing_edge_maps = {}  # Tracks reverse mappings pointing upstream nodes to downstreams

    def visit(
        upstream_node,
        upstream_label,
        downstream_node,
        downstream_label,
        downstream_selector=None
    ):
        # A cycle is found if an upstream node matches an item on the current stack path
        if upstream_node in marked_nodes: raise RuntimeError(f"Cyclic dependency detected! The graph contains a loop involving node: {upstream_node.name}")

        # Update the reverse outgoing mapping layout data frames
        if downstream_node is not None:
            outgoing_edge_map = outgoing_edge_maps.setdefault(
                upstream_node,
                {}
            )

            outgoing_edges = outgoing_edge_map.setdefault(
                upstream_label,
                []
            )

            outgoing_edges.append((downstream_node, downstream_label, downstream_selector))

        # Skip nodes that have already been fully resolved and sorted
        if upstream_node in sorted_nodes: return
        # Add the node to the active stack path
        marked_nodes.add(upstream_node)

        # Recursively visit all upstream dependencies first
        for edge in upstream_node.incoming_edges:
            visit(edge.upstream_node, edge.upstream_label, edge.downstream_node, edge.downstream_label, edge.upstream_selector)

        # Remove the node from the active stack path and commit it to the sorted list
        marked_nodes.remove(upstream_node)
        sorted_nodes.append(upstream_node)

    # Convert the raw array input parameters into initial pending items
    pending_nodes = [(node, None) for node in downstream_nodes]
    # Process all pending root nodes
    while pending_nodes:
        upstream_node, upstream_label = pending_nodes.pop()
        visit(upstream_node, upstream_label, None, None)

    return sorted_nodes, outgoing_edge_maps