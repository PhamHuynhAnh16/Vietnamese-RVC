import os
import sys
import math
import bisect
import operator
import collections.abc

from _thread import get_ident
from functools import reduce, wraps
from itertools import chain, repeat, starmap

sys.path.append(os.getcwd())

from main.app.variables import logger

def recursive_repr(func):
    """
    Decorator to prevent infinite recursion in __repr__ methods.
    
    If a container references itself, this replaces the recursive call with '...'.
    """

    repr_running = set()

    @wraps(func)
    def wrapper(self):
        key = id(self), get_ident()
        if key in repr_running: return '...'

        repr_running.add(key)

        try:
            return func(self)
        finally:
            repr_running.discard(key)

    return wrapper

class SortedList(collections.abc.MutableSequence):
    """A mutable sequence that automatically maintains its elements in sorted order."""

    def __init__(self, iterable=None, load=1000):
        """
        Initialize the SortedList.

        Args:
            iterable: An optional iterable of items to initialize the list with.
            load: The load factor controlling the maximum sublist size before splitting.
        """

        self._len, self._maxes, self._lists, self._index = 0, [], [], []
        self._load, self._twice, self._half = load, load * 2, load >> 1
        if iterable is not None: self.update(iterable)

    def clear(self):
        """Remove all elements from the SortedList."""

        self._len = 0

        del self._maxes[:]
        del self._lists[:]
        del self._index[:]

    def add(self, val):
        """
        Insert a new element into the SortedList maintaining the sorted order.

        Args:
            val: The value to be added.
        """

        _maxes, _lists = self._maxes, self._lists

        if _maxes:
            # Find the appropriate sublist index based on maximum sublist values
            pos = bisect.bisect_right(_maxes, val)

            if pos == len(_maxes):
                # Element is larger than all current elements; append to the last sublist
                pos -= 1
                _maxes[pos] = val
                _lists[pos].append(val)
            else:
                # Insert inside the designated sublist maintaining order
                bisect.insort(_lists[pos], val)

            self._expand(pos)
        else:
            # First element initialization
            _maxes.append(val)
            _lists.append([val])

        self._len += 1

    def _expand(self, pos):
        """
        Split a sublist if it exceeds the load limit threshold.

        Args:
            pos: Position index of the sublist to balance.
        """

        _lists, _index = self._lists, self._index
        # Check if the sublist size violates the load factor ceiling
        if len(_lists[pos]) > self._twice:
            _maxes, _load = self._maxes, self._load
            half = _lists[pos][_load:]
            # Split the heavy sublist into two equal halves
            _lists[pos] = _lists[pos][:_load]
            _maxes[pos] = _lists[pos][-1]

            _maxes.insert(pos + 1, half[-1])
            _lists.insert(pos + 1, half)
            # Invalidate index tree since structure changed
            del _index[:]
        else:
            # Update positional index counters if tree index is active
            if len(_index) > 0:
                child = self._offset + pos

                while child > 0:
                    _index[child] += 1
                    child = (child - 1) >> 1

                _index[0] += 1

    def update(self, iterable):
        """
        Batch insert multiple values from an iterable into the SortedList.

        Args:
            iterable: Collection of elements to be merged.
        """

        _maxes, _lists = self._maxes, self._lists
        values = sorted(iterable)

        if _maxes:
            # If incoming dataset is large, merge entirely and rebuild to minimize overhead
            if len(values) * 4 >= self._len:
                values.extend(chain.from_iterable(_lists))
                values.sort()
                self.clear()
            else:
                # Step-by-step element injection for smaller datasets
                _add = self.add

                for val in values:
                    _add(val)

                return

        # Distribute sorted items into fresh sublists based on load factor chunking
        _load, _index = self._load, self._index
        _lists.extend(values[pos:(pos + _load)] for pos in range(0, len(values), _load))
        _maxes.extend(sublist[-1] for sublist in _lists)

        self._len = len(values)
        del _index[:]

    def __contains__(self, val):
        """Check if a value exists in the SortedList."""
        _maxes = self._maxes
        if not _maxes: return False

        # Locate candidate sublist via binary search
        pos = bisect.bisect_left(_maxes, val)
        if pos == len(_maxes): return False

        # Query sublist for presence of element
        _lists = self._lists
        idx = bisect.bisect_left(_lists[pos], val)
        return _lists[pos][idx] == val

    def discard(self, val):
        """Remove a value if it is present in the container."""

        _maxes = self._maxes

        if not _maxes: return
        pos = bisect.bisect_left(_maxes, val)

        if pos == len(_maxes): return
        _lists = self._lists

        idx = bisect.bisect_left(_lists[pos], val)
        if _lists[pos][idx] == val: self._delete(pos, idx)

    def remove(self, val):
        """
        Remove a value from the container; raises ValueError if not found.

        Args:
            val: Element to eliminate.
        """

        _maxes = self._maxes
        if not _maxes: raise ValueError(f"remove(x): x not in SortedList (list is empty)")

        pos = bisect.bisect_left(_maxes, val)
        if pos == len(_maxes): raise ValueError(f"remove(x): x not in SortedList (value exceeds max bound)")

        _lists = self._lists
        idx = bisect.bisect_left(_lists[pos], val)

        if _lists[pos][idx] == val: self._delete(pos, idx)
        else: raise ValueError(f"remove(x): x not in SortedList")

    def _delete(self, pos, idx):
        """
        Internal operation to delete an element at a specific sublist coordinates.

        Args:
            pos: Sublist container index.
            idx: Local list index element.
        """

        _maxes, _lists, _index = self._maxes, self._lists, self._index

        lists_pos = _lists[pos]
        del lists_pos[idx]

        self._len -= 1
        len_lists_pos = len(lists_pos)
        # Check if sublist size remains stable within bounds
        if len_lists_pos > self._half:
            _maxes[pos] = lists_pos[-1]
            # Cascade decrements up the positional tree index
            if len(_index) > 0:
                child = self._offset + pos

                while child > 0:
                    _index[child] -= 1
                    child = (child - 1) >> 1

                _index[0] -= 1
        elif len(_lists) > 1:
            # Merge shallow sublist with adjacent neighbors to maintain density
            if pos == 0: pos += 1

            prev = pos - 1
            _lists[prev].extend(_lists[pos])
            _maxes[prev] = _lists[prev][-1]

            del _maxes[pos]
            del _lists[pos]
            del _index[:]

            self._expand(prev)
        elif len_lists_pos: _maxes[pos] = lists_pos[-1]
        else:
            del _maxes[pos]
            del _lists[pos]
            del _index[:]

    def _loc(self, pos, idx):
        """
        Convert a localized (sublist, element) coordinate to a global linear index.

        Args:
            pos: Sublist target offset.
            idx: Localized array element position.
        """

        if pos == 0: return idx

        _index = self._index
        if len(_index) == 0: self._build_index()

        total = 0
        pos += self._offset
        # Traverse upwards through tree structure accumulators
        while pos:
            if not (pos & 1): total += _index[pos - 1]
            pos = (pos - 1) >> 1

        return total + idx

    def _pos(self, idx):
        """
        Convert a global sequence index into localized (sublist, element) index.

        Args:
            idx: Global positional index integer.
        """

        _len, _lists = self._len, self._lists
        # Standardizing negative indexing parameters
        if idx < 0:
            last_len = len(_lists[-1])
            if (-idx) <= last_len: return len(_lists) - 1, last_len + idx

            idx += _len
            if idx < 0: raise IndexError(f"SortedList index out of range (negative index overflow: {idx - _len})")
        elif idx >= _len: raise IndexError(f"SortedList index out of range: {idx}")
        # Short-circuit optimize mapping targets landing in the primary sublist
        if idx < len(_lists[0]): return 0, idx

        _index = self._index
        if len(_index) == 0: self._build_index()
        # Binary search downwards through the segmented tracking index
        pos = 0
        len_index = len(_index)
        child = (pos << 1) + 1

        while child < len_index:
            index_child = _index[child]

            if idx < index_child: pos = child
            else:
                idx -= index_child
                pos = child + 1

            child = (pos << 1) + 1

        return (pos - self._offset, idx)

    def _build_index(self):
        """Construct the structural indexing tree layout mapping offsets to segment counts."""

        row0 = list(map(len, self._lists))

        if len(row0) == 1:
            self._index[:] = row0
            self._offset = 0
            return

        head = iter(row0)
        tail = iter(head)
        row1 = list(starmap(operator.add, zip(head, tail)))

        if len(row0) & 1: row1.append(row0[-1])

        if len(row1) == 1:
            self._index[:] = row1 + row0
            self._offset = 1
            return

        size = 2 ** (int(math.log(len(row1) - 1, 2)) + 1)
        row1.extend(repeat(0, size - len(row1)))
        tree = [row0, row1]
        # Generate aggregated index level structures
        while len(tree[-1]) > 1:
            head = iter(tree[-1])
            tail = iter(head)

            row = list(starmap(operator.add, zip(head, tail)))
            tree.append(row)

        reduce(operator.iadd, reversed(tree), self._index)
        self._offset = size * 2 - 1

    def _slice(self, slc):
        """
        Normalize Python slicing specifications matching current state limitations.

        Args:
            slc: Native slice configuration.
        """

        start, stop, step = slc.start, slc.stop, slc.step
        if step == 0: raise ValueError("slice step cannot be zero")

        if step is None: step = 1

        if step > 0:
            if start is None: start = 0
            if stop is None: stop = len(self)
            elif stop < 0: stop += len(self)
        else:
            if start is None: start = len(self) - 1
            if stop is None: stop = -1
            elif stop < 0: stop += len(self)

        if start < 0: start += len(self)

        if step > 0:
            if start < 0: start = 0
            elif start > len(self): start = len(self)
            if stop < 0: stop = 0
            elif stop > len(self): stop = len(self)
        else:
            if start < 0: start = -1
            elif start >= len(self): start = len(self) - 1
            if stop < 0: stop = -1
            elif stop > len(self): stop = len(self)

        return start, stop, step

    def __delitem__(self, idx):
        """Remove element or slice from container at designated reference index."""

        if isinstance(idx, slice):
            start, stop, step = self._slice(idx)
            # Heavy structural truncation optimize: rewrite container instead of individual deletes
            if ((step == 1) and (start < stop) and ((stop - start) * 8 >= self._len)):
                values = self[:start]
                if stop < self._len: values += self[stop:]

                self.clear()
                self.update(values)

                return

            indices = range(start, stop, step)
            if step > 0: indices = reversed(indices)
            _pos, _delete = self._pos, self._delete

            for index in indices:
                pos, idx = _pos(index)
                _delete(pos, idx)
        else:
            pos, idx = self._pos(idx)
            self._delete(pos, idx)

    def __getitem__(self, idx):
        """Retrieve element or slice segment based on requested index mapping."""

        _lists = self._lists

        if isinstance(idx, slice):
            start, stop, step = self._slice(idx)
            if step == 1 and start < stop:
                if start == 0 and stop == self._len: return self.as_list()
                start_pos, start_idx = self._pos(start)

                if stop == self._len:
                    stop_pos = len(_lists) - 1
                    stop_idx = len(_lists[stop_pos])
                else: stop_pos, stop_idx = self._pos(stop)

                if start_pos == stop_pos: return _lists[start_pos][start_idx:stop_idx]

                # Stitch slices traversing boundary lines together
                prefix = _lists[start_pos][start_idx:]
                middle = _lists[(start_pos + 1):stop_pos]

                result = reduce(operator.iadd, middle, prefix)
                result += _lists[stop_pos][:stop_idx]

                return result

            if step == -1 and start > stop:
                result = self[(stop + 1):(start + 1)]
                result.reverse()
                return result

            indices = range(start, stop, step)
            return list(self[index] for index in indices)
        else:
            pos, idx = self._pos(idx)
            return _lists[pos][idx]

    def _check_order(self, idx, val):
        """
        Verify sorting property guarantees are maintained around a specific index index position.

        Args:
            idx: Positional index verified.
            val: Evaluation entity tracking candidates.
        """

        _lists, _len = self._lists, self._len
        pos, loc = self._pos(idx)
        if idx < 0: idx += _len

        # Verify left neighbor relation integrity
        if idx > 0:
            idx_prev = loc - 1
            pos_prev = pos

            if idx_prev < 0:
                pos_prev -= 1
                idx_prev = len(_lists[pos_prev]) - 1

            if _lists[pos_prev][idx_prev] > val: raise ValueError(f"Order violation: element at index {idx} ({val}) cannot be less than index {idx - 1} ({_lists[pos_prev][idx_prev]})")

        # Verify right neighbor relation integrity
        if idx < (_len - 1):
            idx_next = loc + 1
            pos_next = pos

            if idx_next == len(_lists[pos_next]):
                pos_next += 1
                idx_next = 0

            if _lists[pos_next][idx_next] < val: raise ValueError(f"Order violation: element at index {idx} ({val}) cannot be greater than index {idx + 1} ({_lists[pos_next][idx_next]})")

    def __setitem__(self, index, value):
        """Modify sequence items while strictly maintaining ordering layout rules."""

        _maxes, _lists, _pos = self._maxes, self._lists, self._pos
        _check_order = self._check_order

        if isinstance(index, slice):
            start, stop, step = self._slice(index)
            indices = range(start, stop, step)

            if step != 1:
                if not hasattr(value, '__len__'): value = list(value)

                indices = list(indices)
                if len(value) != len(indices): raise ValueError(f"attempt to assign sequence of size {len(value)} to extended slice of size {len(indices)}")

                log = []
                _append = log.append
                # Stage modification targets transactions
                for idx, val in zip(indices, value):
                    pos, loc = _pos(idx)
                    _append((idx, _lists[pos][loc], val))

                    _lists[pos][loc] = val
                    if len(_lists[pos]) == (loc + 1): _maxes[pos] = val

                # Rollback operations if monotonic ordering contracts break
                try:
                    for idx, oldval, newval in log:
                        _check_order(idx, newval)
                except ValueError:
                    for idx, oldval, newval in log:
                        pos, loc = _pos(idx)
                        _lists[pos][loc] = oldval

                        if len(_lists[pos]) == (loc + 1): _maxes[pos] = oldval

                    raise
            else:
                # Slice modifications requiring sequence rebuilding properties
                if not hasattr(value, '__getitem__'): value = list(value)
                ordered = all(value[pos - 1] <= value[pos] for pos in range(1, len(value)))
                if not ordered: raise ValueError("Assigned slice values must be sorted monotonically")

                if start == 0 or len(value) == 0: pass
                else:
                    if self[start - 1] > value[0]: raise ValueError(f"Slice value start boundary violation: sequence element at {start - 1} ({self[start - 1]}) > new element ({value[0]})")

                if stop == len(self) or len(value) == 0: pass
                else:
                    if self[stop] < value[-1]: raise ValueError(f"Slice value end boundary violation: sequence element at {stop} ({self[stop]}) < new element ({value[-1]})")

                del self[index]
                _insert = self.insert

                for idx, val in enumerate(value):
                    _insert(start + idx, val)
        else:
            pos, loc = _pos(index)
            _check_order(index, value)

            _lists[pos][loc] = value
            if len(_lists[pos]) == (loc + 1):  _maxes[pos] = value

    def __iter__(self):
        """Return an iterator over the SortedList."""

        return chain.from_iterable(self._lists)

    def __reversed__(self):
        """Return a reverse iterator over the SortedList."""

        _lists = self._lists
        start = len(_lists) - 1

        iterable = (reversed(_lists[pos]) for pos in range(start, -1, -1))
        return chain.from_iterable(iterable)

    def __len__(self):
        """Return the total number of elements in the SortedList."""

        return self._len

    def bisect_left(self, val):
        """Find the index where val should be inserted to maintain sorted order (leftmost)."""

        _maxes = self._maxes
        if not _maxes: return 0

        pos = bisect.bisect_left(_maxes, val)
        if pos == len(_maxes): return self._len

        idx = bisect.bisect_left(self._lists[pos], val)
        return self._loc(pos, idx)

    def bisect(self, val):
        """Alias for bisect_right."""

        return self.bisect_right(val)

    def bisect_right(self, val):
        """Find the index where val should be inserted to maintain sorted order (rightmost)."""

        _maxes = self._maxes
        if not _maxes: return 0

        pos = bisect.bisect_right(_maxes, val)
        if pos == len(_maxes): return self._len
        idx = bisect.bisect_right(self._lists[pos], val)

        return self._loc(pos, idx)

    def count(self, val):
        """Return the number of occurrences of a value."""

        _maxes = self._maxes
        if not _maxes: return 0

        pos_left = bisect.bisect_left(_maxes, val)
        if pos_left == len(_maxes): return 0

        _lists = self._lists
        idx_left = bisect.bisect_left(_lists[pos_left], val)
        pos_right = bisect.bisect_right(_maxes, val)

        if pos_right == len(_maxes): return self._len - self._loc(pos_left, idx_left)
        idx_right = bisect.bisect_right(_lists[pos_right], val)

        if pos_left == pos_right:
            return idx_right - idx_left

        right = self._loc(pos_right, idx_right)
        left = self._loc(pos_left, idx_left)

        return right - left

    def copy(self):
        """Return a shallow copy of the SortedList."""

        return SortedList(self, load=self._load)

    def __copy__(self):
        """Return a shallow copy of the SortedList."""

        return self.copy()

    def append(self, val):
        """Append a value; raises ValueError if it breaks sorted order restrictions."""

        _maxes, _lists = self._maxes, self._lists

        if not _maxes:
            _maxes.append(val)
            _lists.append([val])

            self._len = 1
            return

        pos = len(_lists) - 1
        if val < _lists[pos][-1]: raise ValueError(f"append(x): x ({val}) cannot be less than the largest element ({_lists[pos][-1]})")

        _maxes[pos] = val
        _lists[pos].append(val)

        self._len += 1
        self._expand(pos)

    def extend(self, values):
        """Extend list elements; raises ValueError if incoming data breaks sorted properties."""

        _maxes, _lists, _load = self._maxes, self._lists, self._load

        if not isinstance(values, list):
            values = list(values)

        if any(values[pos - 1] > values[pos] for pos in range(1, len(values))):
            raise ValueError("extend(iterable): items in iterable must be sorted monotonically")

        offset = 0

        if _maxes:
            if values[0] < _lists[-1][-1]: raise ValueError(f"extend(iterable): first item ({values[0]}) cannot be less than the largest container element ({_lists[-1][-1]})")

            if len(_lists[-1]) < self._half:
                _lists[-1].extend(values[:_load])
                _maxes[-1] = _lists[-1][-1]
                offset = _load

        len_lists = len(_lists)

        for idx in range(offset, len(values), _load):
            _lists.append(values[idx:(idx + _load)])
            _maxes.append(_lists[-1][-1])

        _index = self._index

        if len_lists == len(_lists):
            len_index = len(_index)

            if len_index > 0:
                len_values = len(values)
                child = len_index - 1

                while child:
                    _index[child] += len_values
                    child = (child - 1) >> 1

                _index[0] += len_values
        else: del self._index[:]

        self._len += len(values)

    def insert(self, idx, val):
        """Insert a value at a numerical position; raises ValueError if order is compromised."""

        _maxes, _lists, _len = self._maxes, self._lists, self._len

        if idx < 0: idx += _len
        if idx < 0: idx = 0
        if idx > _len: idx = _len

        if not _maxes:
            _maxes.append(val)
            _lists.append([val])

            self._len = 1
            return

        if idx == 0:
            if val > _lists[0][0]: raise ValueError(f"insert(0, x): x ({val}) cannot be greater than the first element ({_lists[0][0]})")
            else:
                _lists[0].insert(0, val)
                self._expand(0)

                self._len += 1
                return

        if idx == _len:
            pos = len(_lists) - 1

            if _lists[pos][-1] > val: raise ValueError(f"insert({_len}, x): x ({val}) cannot be less than the last element ({_lists[pos][-1]})")
            else:
                _lists[pos].append(val)
                _maxes[pos] = _lists[pos][-1]

                self._expand(pos)
                self._len += 1
                return

        pos, idx = self._pos(idx)
        idx_before = idx - 1

        if idx_before < 0:
            pos_before = pos - 1
            idx_before = len(_lists[pos_before]) - 1
        else: pos_before = pos

        before = _lists[pos_before][idx_before]

        if before <= val <= _lists[pos][idx]:
            _lists[pos].insert(idx, val)
            self._expand(pos)
            self._len += 1
        else: raise ValueError(f"insert({idx}, x): x ({val}) violates sorted properties bounds")

    def pop(self, idx=-1):
        """Remove and return item at index (default last element)."""

        if (idx < 0 and -idx > self._len) or (idx >= self._len): raise IndexError(f"pop index out of range: {idx}")

        pos, idx = self._pos(idx)
        val = self._lists[pos][idx]
        self._delete(pos, idx)

        return val

    def index(self, val, start=None, stop=None):
        """Return first index index occurrence of value within target slice bounds."""

        _len, _maxes = self._len, self._maxes
        if not _maxes: raise ValueError(f"index(x): x not in SortedList (list is empty)")

        if start is None: start = 0
        if start < 0: start += _len
        if start < 0: start = 0
        if stop is None: stop = _len
        if stop < 0: stop += _len
        if stop > _len: stop = _len

        if stop <= start: raise ValueError(f"index(x): valid slice range empty (start={start} >= stop={stop})")

        stop -= 1
        pos_left = bisect.bisect_left(_maxes, val)
        if pos_left == len(_maxes): raise ValueError(f"index(x): x not in SortedList (exceeds maximum elements)")

        _lists = self._lists
        idx_left = bisect.bisect_left(_lists[pos_left], val)

        if _lists[pos_left][idx_left] != val: raise ValueError(f"index(x): x not in SortedList")
        left = self._loc(pos_left, idx_left)

        if start <= left:
            if left <= stop: return left
        else:
            right = self.bisect_right(val) - 1
            if start <= right: return start

        raise ValueError(f"index(x): x found at index {left}, but it is out of the specified bounds [{start}:{stop+1}]")

    def as_list(self):
        """Flatten segmented representation returning objects in a single continuous layout list."""

        return reduce(operator.iadd, self._lists, [])

    def __add__(self, that):
        """Concatenate SortedList with an external sequence, producing a new SortedList instance."""

        values = self.as_list()
        values.extend(that)

        return SortedList(values)

    def __iadd__(self, that):
        """Inplace addition expansion sequence updates."""

        self.update(that)
        return self

    def __mul__(self, that):
        """Multiply SortedList items generating duplicated sequence sets into a new object instance."""

        values = self.as_list() * that
        return SortedList(values)

    def __imul__(self, that):
        """Inplace multiplication allocation update processing."""

        values = self.as_list() * that
        self.clear()
        self.update(values)
        return self

    def __eq__(self, that):
        """Check if structural sequence elements are strictly identical."""

        return ((self._len == len(that)) and all(lhs == rhs for lhs, rhs in zip(self, that)))

    def __ne__(self, that):
        """Check if structural sequence elements are non-identical."""

        return ((self._len != len(that)) or any(lhs != rhs for lhs, rhs in zip(self, that)))

    def __lt__(self, that):
        """Less than inequality comparator checks."""

        return ((self._len <= len(that)) and all(lhs < rhs for lhs, rhs in zip(self, that)))

    def __le__(self, that):
        """Less-or-equal inequality comparator checks."""

        return ((self._len <= len(that)) and all(lhs <= rhs for lhs, rhs in zip(self, that)))

    def __gt__(self, that):
        """Greater than inequality comparator checks."""

        return ((self._len >= len(that)) and all(lhs > rhs for lhs, rhs in zip(self, that)))

    def __ge__(self, that):
        """Greater-or-equal inequality comparator checks."""

        return ((self._len >= len(that)) and all(lhs >= rhs for lhs, rhs in zip(self, that)))

    @recursive_repr
    def __repr__(self):
        """Return a string representation of the object."""

        return '{0}({1})'.format(self.__class__.__name__, repr(self.as_list()))

    def _check(self):
        """Perform comprehensive integrity checks on the internal state structure of the SortedList."""

        try:
            assert self._load >= 4
            assert self._half == (self._load >> 1)
            assert self._twice == (self._load * 2)

            if self._maxes == []:
                assert self._lists == []
                return

            assert len(self._maxes) > 0 and len(self._lists) > 0
            assert all(sublist[pos - 1] <= sublist[pos] for sublist in self._lists for pos in range(1, len(sublist)))

            for pos in range(1, len(self._lists)):
                assert self._lists[pos - 1][-1] <= self._lists[pos][0]

            assert len(self._maxes) == len(self._lists)
            assert all(self._maxes[pos] == self._lists[pos][-1] for pos in range(len(self._maxes)))
            assert all(len(sublist) <= self._twice for sublist in self._lists)
            assert all(len(self._lists[pos]) >= self._half for pos in range(0, len(self._lists) - 1))
            assert self._len == sum(len(sublist) for sublist in self._lists)

            if len(self._index):
                assert len(self._index) == self._offset + len(self._lists)
                assert self._len == self._index[0]

                def test_offset_pos(pos):
                    from_index = self._index[self._offset + pos]
                    return from_index == len(self._lists[pos])

                assert all(test_offset_pos(pos) for pos in range(len(self._lists)))

                for pos in range(self._offset):
                    child = (pos << 1) + 1

                    if self._index[pos] == 0: assert child >= len(self._index)
                    elif child + 1 == len(self._index): assert self._index[pos] == self._index[child]
                    else: assert self._index[pos] == (self._index[child] + self._index[child + 1])
        except:
            import traceback

            # Log execution traceback state logs for analysis
            logger.error(traceback.format_exc())
            logger.debug(self._len, self._load, self._half, self._twice)
            logger.debug(self._index)
            logger.debug(self._maxes)
            logger.debug(self._lists)

            raise