import os
import copy
import hashlib
import subprocess

from collections import namedtuple
from collections.abc import Iterable

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

class DagNode:
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    @property
    def short_repr(self):
        raise NotImplementedError

    @property
    def incoming_edge_map(self):
        raise NotImplementedError

class KwargReprNode(DagNode):
    def __init__(self, incoming_edge_map, name, args=None, kwargs=None):
        self._incoming_edge_map = incoming_edge_map
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}

        self._hash = self._build_hash()

    def _get_upstream_hashes(self):
        hashes = []

        for downstream_label, upstream_info in self.incoming_edge_map.items():
            upstream_node, upstream_label, upstream_selector = upstream_info

            hashes.extend([
                hash(downstream_label),
                hash(upstream_node),
                hash(upstream_label),
                hash(upstream_selector),
            ])

        return hashes

    def _get_inner_hash(self):
        return get_hash({
            'args': self.args,
            'kwargs': self.kwargs,
        })

    def _build_hash(self):
        hashes = self._get_upstream_hashes()
        hashes.append(self._get_inner_hash())

        return get_hash_int(hashes)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, self.__class__) and hash(self) == hash(other)

    def __repr__(self):
        return self.long_repr()

    @property
    def incoming_edge_map(self):
        return self._incoming_edge_map

    @property
    def incoming_edges(self):
        return get_incoming_edges(self, self.incoming_edge_map)

    @property
    def short_hash(self):
        return format(abs(hash(self)), 'x')[:12]

    @property
    def short_repr(self):
        return self.name

    def long_repr(self, include_hash=True):
        formatted = [repr(arg) for arg in self.args]

        formatted.extend(
            '{}={!r}'.format(key, self.kwargs[key])
            for key in sorted(self.kwargs)
        )

        result = '{}({})'.format(
            self.name,
            ', '.join(formatted),
        )

        if include_hash:
            result += ' <{}>'.format(self.short_hash)

        return result

class Stream:
    def __init__(
        self,
        upstream_node,
        upstream_label,
        node_types,
        upstream_selector=None,
    ):
        if not is_of_types(upstream_node, node_types):
            raise TypeError

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
        node_repr = self.node.long_repr(include_hash=False)

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
        if self.selector is not None:
            raise ValueError

        if not isinstance(index, str):
            raise TypeError

        return self.node.stream(
            label=self.label,
            selector=index,
        )

    @property
    def audio(self):
        return self['a']

    @property
    def video(self):
        return self['v']

class Node(KwargReprNode):
    @classmethod
    def check_input_len(cls, stream_map, min_inputs, max_inputs):
        stream_count = len(stream_map)

        if min_inputs is not None and stream_count < min_inputs:
            raise ValueError

        if max_inputs is not None and stream_count > max_inputs:
            raise ValueError

    @classmethod
    def check_input_types(cls, stream_map, incoming_stream_types):
        for stream in stream_map.values():
            if not is_of_types(stream, incoming_stream_types):
                raise TypeError

    @classmethod
    def get_incoming_edge_map(cls, stream_map):
        edge_map = {}

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
        self._outgoing_stream_type = outgoing_stream_type

        stream_map = get_stream_map(stream_spec)

        self.check_input_len(
            stream_map,
            min_inputs,
            max_inputs,
        )

        self.check_input_types(
            stream_map,
            incoming_stream_types,
        )

        super().__init__(
            self.get_incoming_edge_map(stream_map),
            name,
            args=args,
            kwargs=kwargs,
        )

    def stream(self, label=None, selector=None):
        return self._outgoing_stream_type(
            self,
            label,
            upstream_selector=selector,
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.stream(
                label=item.start,
                selector=item.stop,
            )

        return self.stream(label=item)

class FilterableStream(Stream):
    def __init__(
        self,
        upstream_node,
        upstream_label,
        upstream_selector=None,
    ):
        super().__init__(
            upstream_node,
            upstream_label,
            {InputNode, FilterNode},
            upstream_selector,
        )

class InputNode(Node):
    def __init__(self, name, args=None, kwargs=None):
        super().__init__(
            stream_spec=None,
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
        return os.path.basename(self.kwargs['filename'])

class FilterNode(Node):
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
        args = self.args
        kwargs = self.kwargs

        if self.name in ('split', 'asplit'):
            args = [len(outgoing_edges)]

        escaped_args = [
            escape_chars(value, "\\'=:")
            for value in args
        ]

        escaped_kwargs = {}

        for key, value in kwargs.items():
            escaped_kwargs[
                escape_chars(key, "\\'=:")
            ] = escape_chars(value, "\\'=:")

        params = []

        params.extend(
            escape_chars(value, "\\'=:")
            for value in escaped_args
        )

        params.extend(
            '{}={}'.format(key, escaped_kwargs[key])
            for key in sorted(escaped_kwargs)
        )

        result = escape_chars(self.name, "\\'=:")
        if params: result += '={}'.format(':'.join(params))

        return escape_chars(result, "\\'[],;")

class OutputNode(Node):
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
    def __init__(
        self,
        upstream_node,
        upstream_label,
        upstream_selector=None,
    ):
        super().__init__(
            upstream_node,
            upstream_label,
            {OutputNode, GlobalNode, MergeOutputsNode},
            upstream_selector,
        )

class MergeOutputsNode(Node):
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
    edges = []

    for downstream_label, upstream_info in incoming_edge_map.items():
        upstream_node, upstream_label, upstream_selector = upstream_info

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
    edges = []

    for upstream_label, downstream_infos in sorted(outgoing_edge_map.items()):
        for downstream_info in downstream_infos:
            downstream_node, downstream_label, downstream_selector = downstream_info

            edges.append(
                DagEdge(
                    downstream_node,
                    downstream_label,
                    upstream_node,
                    upstream_label,
                    downstream_selector,
                )
            )

    return edges

def is_of_types(obj, types):
    return any(isinstance(obj, item) for item in types)

def get_stream_map(stream_spec):
    if stream_spec is None:
        return {}

    if isinstance(stream_spec, Stream):
        return {None: stream_spec}

    if isinstance(stream_spec, (list, tuple)):
        return dict(enumerate(stream_spec))

    if isinstance(stream_spec, dict):
        return stream_spec

    raise TypeError

def get_stream_map_nodes(stream_map):
    nodes = []

    for stream in stream_map.values():
        if not isinstance(stream, Stream):
            raise TypeError

        nodes.append(stream.node)

    return nodes

def get_stream_spec_nodes(stream_spec):
    return get_stream_map_nodes(
        get_stream_map(stream_spec)
    )

def stream_operator(stream_classes=None, name=None):
    stream_classes = stream_classes or {Stream}

    def decorator(func):
        func_name = name or func.__name__

        for stream_class in stream_classes:
            setattr(stream_class, func_name, func)

        return func

    return decorator

def filter_operator(name=None):
    return stream_operator(
        stream_classes={FilterableStream},
        name=name,
    )

def output_operator(name=None):
    return stream_operator(
        stream_classes={OutputStream},
        name=name,
    )

def input(filename, **kwargs):
    kwargs['filename'] = filename

    fmt = kwargs.pop('f', None)

    if fmt:
        if 'format' in kwargs:
            raise ValueError

        kwargs['format'] = fmt

    return InputNode(
        input.__name__,
        kwargs=kwargs,
    ).stream()

@output_operator()
def global_args(stream, *args):
    return GlobalNode(
        stream,
        global_args.__name__,
        args,
    ).stream()

@output_operator()
def overwrite_output(stream):
    return GlobalNode(
        stream,
        overwrite_output.__name__,
        ['-y'],
    ).stream()

@output_operator()
def merge_outputs(*streams):
    return MergeOutputsNode(
        streams,
        merge_outputs.__name__,
    ).stream()

@filter_operator()
def output(*streams_and_filename, **kwargs):
    streams_and_filename = list(streams_and_filename)

    if 'filename' not in kwargs:
        if not isinstance(streams_and_filename[-1], str):
            raise ValueError

        kwargs['filename'] = streams_and_filename.pop()

    fmt = kwargs.pop('f', None)

    if fmt:
        if 'format' in kwargs:
            raise ValueError

        kwargs['format'] = fmt

    return OutputNode(
        streams_and_filename,
        output.__name__,
        kwargs=kwargs,
    ).stream()

def escape_chars(text, chars):
    text = str(text)

    chars = list(set(chars))

    if '\\' in chars:
        chars.remove('\\')
        chars.insert(0, '\\')

    for char in chars:
        text = text.replace(char, '\\' + char)

    return text

def recursive_repr(item):
    if isinstance(item, str):
        return item

    if isinstance(item, list):
        return '[{}]'.format(
            ', '.join(recursive_repr(x) for x in item)
        )

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
    value = recursive_repr(item).encode('utf-8')

    return hashlib.md5(value).hexdigest()

def get_hash_int(item):
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

    out, err = process.communicate(input)
    retcode = process.poll()

    if retcode: raise RuntimeError(err)
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
    args = compile(stream_spec, cmd=cmd, overwrite_output=overwrite_output)

    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout else None
    stderr_stream = subprocess.PIPE if pipe_stderr else None

    if quiet:
        stdout_stream = subprocess.DEVNULL
        stderr_stream = subprocess.STDOUT

    return subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream, cwd=cwd)

@output_operator()
def compile(stream_spec, cmd='ffmpeg', overwrite_output=False):
    if isinstance(cmd, str): cmd = [cmd]
    elif not isinstance(cmd, list): cmd = list(cmd)

    return cmd + get_args(stream_spec, overwrite_output=overwrite_output)

@output_operator()
def get_args(stream_spec, overwrite_output=False):
    nodes = get_stream_spec_nodes(stream_spec)
    sorted_nodes, outgoing_edge_maps = topo_sort(nodes)
    args, input_nodes, output_nodes, global_nodes, filter_nodes = [], [], [], [], []

    for node in sorted_nodes:
        if isinstance(node, InputNode): input_nodes.append(node)
        elif isinstance(node, OutputNode): output_nodes.append(node)
        elif isinstance(node, GlobalNode): global_nodes.append(node)
        elif isinstance(node, FilterNode): filter_nodes.append(node)

    stream_name_map = {(node, None): str(index) for index, node in enumerate(input_nodes)}
    for node in input_nodes:
        args.extend(get_input_args(node))

    filter_arg = get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map)

    if filter_arg: args.extend(['-filter_complex', filter_arg])

    for node in output_nodes:
        args.extend(get_output_args(node, stream_name_map))

    for node in global_nodes:
        args.extend(get_global_args(node))

    if overwrite_output: args.append('-y')
    return args

def get_global_args(node):
    return list(node.args)

def get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map):
    allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map)
    return ';'.join([get_filter_spec(node, outgoing_edge_maps[node], stream_name_map) for node in filter_nodes])

def allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map):
    stream_count = 0

    for upstream_node in filter_nodes:
        outgoing_edge_map = outgoing_edge_maps[upstream_node]

        for upstream_label, downstreams in sorted(outgoing_edge_map.items()):
            if len(downstreams) > 1: raise ValueError

            stream_name_map[(upstream_node, upstream_label)] = 's{}'.format(stream_count)
            stream_count += 1

def get_filter_spec(node, outgoing_edge_map, stream_name_map):
    incoming_edges = node.incoming_edges
    outgoing_edges = get_outgoing_edges(node, outgoing_edge_map)

    inputs = [format_input_stream_name(stream_name_map, edge) for edge in incoming_edges]
    outputs = [format_output_stream_name(stream_name_map, edge) for edge in outgoing_edges]

    return '{}{}{}'.format(''.join(inputs), node._get_filter(outgoing_edges), ''.join(outputs))

def get_input_args(input_node):
    if input_node.name != input.__name__: raise ValueError

    kwargs = copy.copy(input_node.kwargs)
    filename = kwargs.pop('filename')
    fmt = kwargs.pop('format', None)
    video_size = kwargs.pop('video_size', None)
    args = []

    if fmt: args.extend(['-f', fmt])
    if video_size: args.extend(['-video_size', '{}x{}'.format(video_size[0], video_size[1])])

    args.extend(convert_kwargs_to_cmd_line_args(kwargs))
    args.extend(['-i', filename])

    return args

def get_output_args(node, stream_name_map):
    if node.name != output.__name__: raise ValueError
    if not node.incoming_edges: raise ValueError

    args = []

    for edge in node.incoming_edges:
        stream_name = format_input_stream_name(stream_name_map, edge, is_final_arg=True)
        if stream_name != '0' or len(node.incoming_edges) > 1: args.extend(['-map', stream_name])

    kwargs = copy.copy(node.kwargs)
    filename = kwargs.pop('filename')

    if 'format' in kwargs: args.extend(['-f', kwargs.pop('format')])
    if 'video_bitrate' in kwargs: args.extend(['-b:v', str(kwargs.pop('video_bitrate'))])
    if 'audio_bitrate' in kwargs: args.extend(['-b:a', str(kwargs.pop('audio_bitrate'))])

    if 'video_size' in kwargs:
        video_size = kwargs.pop('video_size')
        if not isinstance(video_size, str) and isinstance(video_size, Iterable): video_size = '{}x{}'.format(video_size[0], video_size[1])

        args.extend(['-video_size', video_size])

    args.extend(convert_kwargs_to_cmd_line_args(kwargs))
    args.append(filename)

    return args

def convert_kwargs_to_cmd_line_args(kwargs):
    args = []

    for key in sorted(kwargs):
        value = kwargs[key]

        if isinstance(value, Iterable) and not isinstance(value, str):
            for item in value:
                args.append('-{}'.format(key))
                if item is not None: args.append(str(item))

            continue

        args.append('-{}'.format(key))
        if value is not None: args.append(str(value))

    return args

def format_input_stream_name(stream_name_map, edge, is_final_arg=False):
    prefix = stream_name_map[(edge.upstream_node, edge.upstream_label)]
    suffix = ''

    if edge.upstream_selector: suffix = ':{}'.format(edge.upstream_selector)
    if is_final_arg and isinstance(edge.upstream_node, InputNode): return '{}{}'.format(prefix, suffix)

    return '[{}{}]'.format(prefix, suffix)

def format_output_stream_name(stream_name_map, edge):
    return '[{}]'.format(stream_name_map[(edge.upstream_node, edge.upstream_label)])

def topo_sort(downstream_nodes):
    marked_nodes = set()
    sorted_nodes = []
    outgoing_edge_maps = {}

    def visit(
        upstream_node,
        upstream_label,
        downstream_node,
        downstream_label,
        downstream_selector=None
    ):
        if upstream_node in marked_nodes: raise RuntimeError

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

        if upstream_node in sorted_nodes: return
        marked_nodes.add(upstream_node)

        for edge in upstream_node.incoming_edges:
            visit(edge.upstream_node, edge.upstream_label, edge.downstream_node, edge.downstream_label, edge.upstream_selector)

        marked_nodes.remove(upstream_node)
        sorted_nodes.append(upstream_node)

    pending_nodes = [(node, None) for node in downstream_nodes]

    while pending_nodes:
        upstream_node, upstream_label = pending_nodes.pop()
        visit(upstream_node, upstream_label, None, None)

    return sorted_nodes, outgoing_edge_maps