import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from main.tools.sortedcontainers import SortedList

AUTO_ROUND_TIME = False
SEGMENT_PRECISION = 1e-6

class Timeline:
    """
    Represents a collection of unique, ordered time segments.

    Provides high-level geometric operations on time intervals such as union,
    intersection (cropping), gaps detection, and overlapping lookups.
    """

    @classmethod
    def from_df(cls, df, uri = None):
        """
        Creates a Timeline instance from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing a 'segment' column.
            uri: Optional unique resource identifier for the timeline.

        Returns:
            A new Timeline instance containing the extracted segments.
        """

        return cls(segments=list(df['segment']), uri=uri)

    def __init__(
        self, 
        segments = None, 
        uri = None
    ):
        """
        Initializes a Timeline with a collection of segments.

        Args:
            segments: An iterable collection of Segment objects.
            uri: Optional unique resource identifier.
        """

        if segments is None: segments = ()
        # Filter out empty/falsy segments and ensure uniqueness using a set
        segments_set = set([segment for segment in segments if segment])

        self.segments_set_ = segments_set
        # Keep segments in a sorted data structure for efficient lookups
        self.segments_list_ = SortedList(segments_set)
        # Store all boundary points (starts and ends) sorted linearly
        self.segments_boundaries_ = SortedList((
            boundary 
            for segment in segments_set 
            for boundary in segment
        ))
        self.uri = uri

    def __len__(self):
        """Returns the number of segments in the timeline."""

        return len(self.segments_set_)

    def __nonzero__(self):
        """Python 2 compatibility wrapper for boolean evaluation."""

        return self.__bool__()

    def __bool__(self):
        """Returns True if the timeline contains at least one segment."""

        return len(self.segments_set_) > 0

    def __iter__(self):
        """Iterates over the sorted segments."""

        return iter(self.segments_list_)

    def __getitem__(self, k):
        """Retrieves a segment by its index from the sorted list."""

        return self.segments_list_[k]

    def __eq__(self, other):
        """Checks if two timelines contain the exact same set of segments."""

        return self.segments_set_ == other.segments_set_

    def __ne__(self, other):
        """Checks if two timelines are different."""

        return self.segments_set_ != other.segments_set_

    def index(self, segment):
        """Returns the sorted index position of a given segment."""

        return self.segments_list_.index(segment)

    def add(self, segment):
        """
        Adds a new valid segment to the timeline if not already present.

        Args:
            segment: The Segment instance to be added.
        """

        segments_set_ = self.segments_set_
        if segment in segments_set_ or not segment: return self

        # Update set and sorted list structures
        segments_set_.add(segment)
        self.segments_list_.add(segment)

        # Update boundaries
        segments_boundaries_ = self.segments_boundaries_
        segments_boundaries_.add(segment.start)
        segments_boundaries_.add(segment.end)

        return self

    def remove(self, segment):
        """
        Removes a segment from the timeline. Raises no error if missing.

        Args:
            segment: The Segment instance to remove.
        """

        segments_set_ = self.segments_set_
        if segment not in segments_set_: return self

        # Remove from data structures
        segments_set_.remove(segment)
        self.segments_list_.remove(segment)

        # Remove its corresponding boundaries
        segments_boundaries_ = self.segments_boundaries_
        segments_boundaries_.remove(segment.start)
        segments_boundaries_.remove(segment.end)

        return self

    def discard(self, segment):
        """Alias for remove() to safely discard a segment."""

        return self.remove(segment)

    def __ior__(self, timeline):
        """In-place union operator (|=). Updates the timeline directly."""

        return self.update(timeline)

    def update(self, timeline):
        """
        Merges another timeline into this instance (In-place Union).

        Args:
            timeline: The timeline to merge from.
        """

        segments_set = self.segments_set_
        segments_set |= timeline.segments_set_
        # Re-build sorted lists based on the updated set
        self.segments_list_ = SortedList(segments_set)
        self.segments_boundaries_ = SortedList((
            boundary 
            for segment in segments_set 
            for boundary in segment
        ))

        return self

    def __or__(self, timeline):
        """Union operator (|). Returns a new Timeline combination."""

        return self.union(timeline)

    def union(self, timeline):
        """
        Returns a new Timeline containing segments from both timelines.

        Args:
            timeline: The other timeline to unite with.
        """

        return Timeline(
            segments=self.segments_set_ | timeline.segments_set_, 
            uri=self.uri
        )

    def co_iter(self, other):
        """
        Yields pairs of intersecting segments between this timeline and another.

        Args:
            other: The other timeline to evaluate intersections against.
        """

        for segment in self.segments_list_:
            # Create a zero-duration dummy segment at the end boundary to optimize range queries
            temp = Segment(start=segment.end, end=segment.end)

            # Look up candidate segments from the other timeline within range
            for other_segment in other.segments_list_.irange(maximum=temp):
                if segment.intersects(other_segment): yield segment, other_segment

    def crop_iter(self, support, mode = 'intersection', returns_mapping = False):
        """
        Iterates over parts of segments that overlap with a support frame.

        Args:
            support: A Segment or Timeline acting as the cropping window.
            mode: 'intersection' (slices overlaps), 'loose' (keeps any intersecting), or 'strict' (keeps only fully contained segments).
            returns_mapping: If True, yields (original_segment, cropped_segment).

        Raises:
            ValueError: If an unsupported mode is provided.
            TypeError: If support is not a Segment or Timeline instance.
        """

        if mode not in {'loose', 'strict', 'intersection'}: raise ValueError("Mode must be 'loose', 'strict', or 'intersection'")
        if not isinstance(support, (Segment, Timeline)): raise TypeError("Support must be a Segment or Timeline instance")

        # Handle fallback if support is a single Segment
        if isinstance(support, Segment):
            support = Timeline(
                segments=([support] if support else []), 
                uri=self.uri
            )

            for yielded in self.crop_iter(
                support, 
                mode=mode, 
                returns_mapping=returns_mapping
            ):
                yield yielded

            return

        # Collapse overlapping regions in support to avoid redundant processing
        support = support.support()

        if mode == 'loose':
            for segment, _ in self.co_iter(support):
                yield segment

            return

        if mode == 'strict':
            for segment, other_segment in self.co_iter(support):
                if segment in other_segment: yield segment

            return

        for segment, other_segment in self.co_iter(support):
            mapped_to = segment & other_segment # Logical AND calculates overlap
            if not mapped_to: continue

            if returns_mapping: yield segment, mapped_to
            else: yield mapped_to

    def crop(self, support, mode = 'intersection', returns_mapping = False):
        """
        Crops the timeline according to a support boundary.

        Args:
            support: Segment or Timeline indicating the target crop zone.
            mode: Slicing configuration ('intersection', 'loose', 'strict').
            returns_mapping: If True, returns a mapping dict along with the Timeline.
        """

        if mode == 'intersection' and returns_mapping:
            segments, mapping = [], {}
            
            for segment, mapped_to in self.crop_iter(
                support, 
                mode='intersection', 
                returns_mapping=True
            ):
                segments.append(mapped_to)
                # Map cropped segment back to its original segment(s)
                mapping[mapped_to] = mapping.get(mapped_to, list()) + [segment]

            return Timeline(segments=segments, uri=self.uri), mapping

        return Timeline(segments=self.crop_iter(support, mode=mode), uri=self.uri)

    def overlapping(self, t):
        """Returns a list of all segments covering timestamp `t`."""

        return list(self.overlapping_iter(t))

    def overlapping_iter(self, t):
        """Yields segments that cover the target timestamp `t`."""

        for segment in self.segments_list_.irange(maximum=Segment(start=t, end=t)):
            if segment.overlaps(t): yield segment

    def get_overlap(self):
        """
        Finds self-intersecting regions within this timeline.

        Returns:
            A new Timeline representing the consolidated internal overlapping areas.
        """

        overlaps_tl = Timeline(uri=self.uri)
        # Self cross-iteration to look for internal collisions
        for s1, s2 in self.co_iter(self):
            if s1 == s2: continue

            overlaps_tl.add(s1 & s2)

        return overlaps_tl.support()

    def extrude(self, removed, mode = 'intersection'):
        """
        Removes specific segments or timelines from the current timeline (Inversion/Subtraction).

        Args:
            removed: The segments/timeline to excise.
            mode: Cropping strategy mode logic.
        """

        if isinstance(removed, Segment): removed = Timeline([removed])
        # Invert the crop modes to switch extraction perspectives
        if mode == "loose": mode = "strict"
        elif mode == "strict": mode = "loose"

        # Crop self against the gaps (non-removed regions) of the removed input
        return self.crop(
            removed.gaps(
                support=Timeline(
                    [self.extent()], 
                    uri=self.uri
                )
            ), 
            mode=mode
        )

    def __str__(self):
        """Returns a clean multi-line string representation of the timeline segments."""

        n = len(self.segments_list_)
        string = "["

        for i, segment in enumerate(self.segments_list_):
            string += str(segment)
            string += "\n " if i + 1 < n else ""

        string += "]"
        return string

    def __repr__(self):
        """Returns the formal developer representation of the Timeline."""

        return "<Timeline(uri=%s, segments=%s)>" % (self.uri, list(self.segments_list_))

    def __contains__(self, included):
        """Checks containment of a Segment or an entire Timeline."""

        if isinstance(included, Segment): return included in self.segments_set_
        elif isinstance(included, Timeline): return self.segments_set_.issuperset(included.segments_set_)
        else: raise TypeError("Can only check containment for Segment or Timeline objects")

    def empty(self):
        """Returns an empty copy of this timeline holding the same URI."""

        return Timeline(uri=self.uri)

    def covers(self, other):
        """Verifies if this timeline entirely wraps around another timeline's coverage."""

        gaps = self.gaps(support=other.extent())
        # If any of our gaps overlap with the other timeline, we don't fully cover it
        for _ in gaps.co_iter(other):
            return False

        return True

    def copy(self, segment_func = None):
        """Returns a copy of the timeline, optionally processing segments through a function.

        Args:
            segment_func: Optional callable to modify segments during copying.
        """

        if segment_func is None: return Timeline(segments=self.segments_list_, uri=self.uri)
        return Timeline(segments=[segment_func(s) for s in self.segments_list_], uri=self.uri)

    def extent(self):
        """Returns the absolute span of the timeline (from very beginning to the absolute end)."""

        if self.segments_set_:
            segments_boundaries_ = self.segments_boundaries_
            return Segment(start=segments_boundaries_[0], end=segments_boundaries_[-1])

        return Segment(start=0.0, end=0.0)

    def support_iter(self, collar = 0.0):
        """
        Fuses adjacent/overlapping segments together into continuous blocks.

        Args:
            collar: Minimum time separation distance required to keep segments separate.
        """

        if not self: return

        new_segment = self.segments_list_[0]

        for segment in self:
            possible_gap = segment ^ new_segment # Symmetric difference tracking

            # If they touch, overlap, or the gap is smaller than the collar, merge them
            if not possible_gap or possible_gap.duration < collar: new_segment |= segment
            else:
                yield new_segment
                new_segment = segment

        yield new_segment

    def support(self, collar = 0.):
        """
        Consolidates the timeline into non-overlapping continuous segments.

        Args:
            collar: Gap padding allowance threshold.
        """

        return Timeline(segments=self.support_iter(collar), uri=self.uri)

    def duration(self):
        """Calculates total unique active time duration across all merged segments."""

        return sum(s.duration for s in self.support_iter())

    def gaps_iter(self, support = None):
        """
        Discovers unallocated slots (empty time gaps) within a specified window.

        Args:
            support: Boundary scope for gap mapping. Defaults to total timeline extent.
        """

        if support is None: support = self.extent()
        if not isinstance(support, (Segment, Timeline)): raise TypeError("Support must be a Segment or Timeline instance")

        if isinstance(support, Segment):
            end = support.start
            # Find holes by tracking steps between continuous blocks
            for segment in self.crop(support, mode='intersection').support():
                gap = Segment(start=end, end=segment.start)
                if gap: yield gap

                end = segment.end

            # Check trailing space gap
            gap = Segment(start=end, end=support.end)
            if gap: yield gap
        elif isinstance(support, Timeline):
            for segment in support.support():
                for gap in self.gaps_iter(support=segment):
                    yield gap

    def gaps(self, support = None):
        """Returns a Timeline composed entirely of the discovered empty spaces."""

        return Timeline(segments=self.gaps_iter(support=support), uri=self.uri)

    def segmentation(self):
        """Subdivides the timeline into atomic chunks based on all present boundary combinations."""

        support = self.support()
        timestamps = set([])

        # Collect every boundary point
        for (start, end) in self:
            timestamps.add(start)
            timestamps.add(end)

        timestamps = sorted(timestamps)
        if len(timestamps) == 0: return Timeline(uri=self.uri)

        segments = []
        start = timestamps[0]
        # Form atomic chunks between sequential boundaries
        for end in timestamps[1:]:
            segment = Segment(start=start, end=end)
            # Keep chunk if it matches an active timeline section
            if segment and support.overlapping(segment.middle): segments.append(segment)
            start = end

        return Timeline(segments=segments, uri=self.uri)

    def _iter_uem(self):
        """Internal generator yielding UEM formatted mapping lines."""

        uri = self.uri if self.uri else "<NA>"

        for segment in self:
            yield f"{uri} 1 {segment.start:.3f} {segment.end:.3f}\n"

    def to_uem(self):
        """Converts timeline data into standard string block matching UEM file format specifications."""

        return "".join([line for line in self._iter_uem()])

    def write_uem(self, file):
        """
        Writes the UEM formatted timeline attributes into a file buffer descriptor.

        Args:
            file: File-like object supporting text write operations.
        """

        for line in self._iter_uem():
            file.write(line)

    def _repr_png_(self):
        """Rich notebook rendering placeholder layout function."""

        return None

class Segment:
    """Represents a continuous interval of time bounded by a start and end point."""

    def __init__(self, start, end):
        """Initializes a Segment object."""

        self.start = start
        self.end = end

    @staticmethod
    def set_precision(ndigits = None):
        """
        Sets the global precision limit rules applied to Segment parameters.

        Args:
            ndigits: Digits resolution parameter count. Passing None disables rounding constraints.
        """

        global AUTO_ROUND_TIME, SEGMENT_PRECISION

        if ndigits is None:
            AUTO_ROUND_TIME = False
            SEGMENT_PRECISION = 1e-6
        else:
            AUTO_ROUND_TIME = True
            SEGMENT_PRECISION = 10 ** (-ndigits)

    def __bool__(self):
        """Evaluates segment validity. Returns False if duration falls below resolution precision."""

        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def __post_init__(self):
        """Applies dynamic runtime rounding filters if AUTO_ROUND_TIME configuration flag is set."""

        if AUTO_ROUND_TIME:
            object.__setattr__(self, 'start', int(self.start / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)
            object.__setattr__(self, 'end', int(self.end / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)

    @property
    def duration(self):
        """Returns the total duration of the segment. Returns 0 if invalid."""

        return self.end - self.start if self else 0.

    @property
    def middle(self):
        """Finds the mathematical midpoint coordinate of the segment instance."""

        return .5 * (self.start + self.end)

    def __iter__(self):
        """Allows unpacking or looping via a (start, end) format structure."""

        yield self.start
        yield self.end

    def copy(self):
        """Returns an identical duplicate clone of the original segment."""

        return Segment(start=self.start, end=self.end)

    def __contains__(self, other):
        """Checks whether this segment completely encapsulates the other object parameter segment."""

        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Logical intersection mapping operator (&) calculating overlapping range areas."""

        return Segment(start=max(self.start, other.start), end=min(self.end, other.end))

    def intersects(self, other):
        """Checks if there is any shared overlap area matching standard bounds constraints."""

        return (
            self.start < other.start and other.start < self.end - SEGMENT_PRECISION
        ) or (
            self.start > other.start and self.start < other.end - SEGMENT_PRECISION
        ) or (
            self.start == other.start
        )

    def overlaps(self, t):
        """Checks if a particular timestamp location point falls anywhere within current frame limits."""

        return self.start <= t and self.end >= t

    def __or__(self, other):
        """Logical union hull merge operator (|) computing bounding range wrapper bounds."""

        if not self: return other
        if not other: return self

        return Segment(
            start=min(self.start, other.start), 
            end=max(self.end, other.end)
        )

    def __xor__(self, other):
        """Symmetric difference operator (^) discovering internal gap separator boundaries."""

        if (not self) or (not other): raise ValueError("Both segments must be valid to evaluate XOR")

        return Segment(
            start=min(self.end, other.end), 
            end=max(self.start, other.start)
        )

    def _str_helper(self, seconds):
        """Internal timedelta display encoder string conversion assistant."""

        from datetime import timedelta

        negative = seconds < 0
        td = timedelta(seconds=abs(seconds))

        hours, remainder = divmod(td.seconds + 86400 * td.days, 3600)
        minutes, seconds = divmod(remainder, 60)

        return '%s%02d:%02d:%02d.%03d' % ('-' if negative else ' ', hours, minutes, seconds, td.microseconds / 1000)

    def __str__(self):
        """Returns user legible standard timestamp boundary summary block layout string."""

        if self: return '[%s --> %s]' % (self._str_helper(self.start), self._str_helper(self.end))
        return '[]'

    def __repr__(self):
        """Developer inspection representation string layout handler structure."""

        return '<Segment(%g, %g)>' % (self.start, self.end)

    def _repr_png_(self):
        """Rich notebook visualization support hook entry vector."""

        return None

class SlidingWindow:
    """Generates sequential window segments sliding across a temporal continuum."""

    def __init__(
        self, 
        duration=0.030, 
        step=0.010, 
        start=0.000, 
        end=None
    ):
        """
        Initializes a SlidingWindow configuration.

        Args:
            duration: Width span size matching frame metrics allocations.
            step: Advancement offset resolution step.
            start: Tracking window offset coordinates floor.
            end: Upper bounds execution range roof ceiling target marker.
        """

        if duration <= 0: raise ValueError("Duration must be strictly positive")
        self.__duration = duration
        if step <= 0: raise ValueError("Step step increments must be positive integers")
        
        self.__step = step
        self.__start = start

        if end is None: self.__end = np.inf
        else:
            if end <= start: raise ValueError("End bounds must surpass tracking start index")
            self.__end = end

        self.__i = -1 # Internal iteration registry tracking counter variable

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def step(self):
        return self.__step

    @property
    def duration(self):
        return self.__duration

    def closest_frame(self, t):
        """Finds the nearest discrete step frame index mapping adjacent to target `t` location."""

        return int(np.rint((t - self.__start - .5 * self.__duration) / self.__step))

    def samples(self, from_duration, mode = 'strict'):
        """
        Calculates expected total chunk allocations capacity fitting into given input length profile rules.

        Args:
            from_duration: Total available evaluation duration length tracker.
            mode: Fitting evaluation structure ('strict', 'loose', 'center').
        """

        if mode == 'strict': return int(np.floor((from_duration - self.duration) / self.step)) + 1
        elif mode == 'loose': return int(np.floor((from_duration + self.duration) / self.step))
        elif mode == 'center': return int(np.rint((from_duration / self.step)))

    def crop(self, focus, mode = 'loose', fixed = None, return_ranges = False):
        """
        Crops sliding steps indices window frame matrices bounding into structural regions parameters map trackers.

        Args:
            focus: Target tracking framework region criteria layer details.
            mode: Intersection mapping configuration selector.
            fixed: Fixed constraint window limit variables specifications rules.
            return_ranges: Toggles returning coordinate arrays raw nested limits.
        """

        if not isinstance(focus, (Segment, Timeline)): raise TypeError("Focus must be a Segment or Timeline instance")

        if isinstance(focus, Timeline):
            if fixed is not None: raise ValueError("Fixed setting is unavailable for Timeline inputs")

            if return_ranges:
                ranges = []

                for i, s in enumerate(focus.support()):
                    rng = self.crop(s, mode=mode, fixed=fixed, return_ranges=True)

                    if i == 0 or rng[0][0] > ranges[-1][1]: ranges += rng
                    else: ranges[-1][1] = rng[0][1] # Merge contiguous block bounds

                return ranges

            # Consolidate linear index coordinates arrays directly together
            return np.unique(
                np.hstack([
                    self.crop(
                        s, 
                        mode=mode, 
                        fixed=fixed, 
                        return_ranges=False
                    ) 
                    for s in focus.support()
                ])
            )

        # Implementation mapping parameters calculations across modes
        if mode == 'loose':
            i = int(np.ceil((focus.start - self.duration - self.start) / self.step))

            if fixed is None:
                j = int(np.floor((focus.end - self.start) / self.step))
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='loose')
                rng = (i, i + n)
        elif mode == 'strict':
            i = int(np.ceil((focus.start - self.start) / self.step))

            if fixed is None:
                j = int(np.floor((focus.end - self.duration - self.start) / self.step))
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='strict')
                rng = (i, i + n)
        elif mode == 'center':
            i = self.closest_frame(focus.start)

            if fixed is None:
                j = self.closest_frame(focus.end)
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='center')
                rng = (i, i + n)
        else: raise ValueError("Invalid mode: choose 'loose', 'strict', or 'center'")

        if return_ranges: return [list(rng)]
        return np.array(range(*rng), dtype=np.int64)

    def segmentToRange(self, segment):
        """Legacy camelCase alias wrapper for segment_to_range."""

        return self.segment_to_range(segment)

    def segment_to_range(self, segment):
        """Maps a physical segment onto internal array space coordinates tuple pairs (start_index, count)."""

        return self.closest_frame(segment.start), int(segment.duration / self.step) + 1

    def rangeToSegment(self, i0, n):
        """Legacy camelCase alias wrapper for range_to_segment."""

        return self.range_to_segment(i0, n)

    def range_to_segment(self, i0, n):
        """Reconstructs continuous spatial metrics mapping dynamic context targets back to structural block classes."""

        start = self.__start + (i0 - .5) * self.__step + .5 * self.__duration

        if i0 == 0: start = self.start
        return Segment(start, start + (n * self.__step))

    def samplesToDuration(self, nSamples):
        """Legacy camelCase alias wrapper for samples_to_duration."""

        return self.samples_to_duration(nSamples)

    def samples_to_duration(self, n_samples):
        """Calculates numerical span runtime dimensions mapping sample step limits."""

        return self.range_to_segment(0, n_samples).duration

    def durationToSamples(self, duration):
        """Legacy camelCase alias wrapper for duration_to_samples."""

        return self.duration_to_samples(duration)

    def duration_to_samples(self, duration):
        """Converts raw window duration metrics into scalar step increments parameters array counts."""

        return self.segment_to_range(Segment(0, duration))[1]

    def __getitem__(self, i):
        """Generates the exact Segment boundary window matching step sequence position integer location `i`."""

        start = self.__start + i * self.__step
        if start >= self.__end: return None

        return Segment(start=start, end=start + self.__duration)

    def next(self):
        return self.__next__()

    def __next__(self):
        """Advances the internal counter index to return the next consecutive Segment frame sequence slice block."""

        self.__i += 1
        window = self[self.__i]

        if window: return window
        else: raise StopIteration()

    def __iter__(self):
        """Resets tracking sequence states pointers to restart sequential iterable navigation loops."""

        self.__i = -1
        return self

    def __len__(self):
        """Determines total available processing windows allocation segments limits matching configuration frames."""

        if np.isinf(self.__end): raise ValueError("Infinite timelines parameters cannot evaluate finite len allocations counters.")
        i = self.closest_frame(self.__end)

        # Proactively look ahead to scan edge window offsets boundaries accurately
        while (self[i]):
            i += 1

        length = i
        return length

    def copy(self):
        """Generates a separate cloned copy reflecting current settings structure."""

        return self.__class__(
            duration=self.duration, 
            step=self.step, 
            start=self.start, 
            end=self.end
        )

    def __call__(self, support, align_last = False):
        """
        Slices a specific support timeline target map using configured sliding step sequences limits.

        Args:
            support: Target area bounds context.
            align_last: Forces generation of an extra adjusted window snapped directly to boundary limits.
        """

        if isinstance(support, Timeline): segments = support
        elif isinstance(support, Segment): segments = Timeline(segments=[support])
        else: raise TypeError("Support must be a Segment or Timeline instance")

        for segment in segments:
            if segment.duration < self.duration: continue
            # Evaluate slices iteratively
            for s in SlidingWindow(
                duration=self.duration, 
                step=self.step, 
                start=segment.start, 
                end=segment.end
            ):
                if s in segment:
                    yield s
                    last = s

            # Handle fractional alignment padding if toggled active
            if align_last and last.end < segment.end: yield Segment(start=segment.end - self.duration, end=segment.end)