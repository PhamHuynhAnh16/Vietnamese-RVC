import os
import sys
import torch

import numpy as np
import torch.utils.data as tdata

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.inference.training.mel_processing import spectrogram_torch
from main.inference.training.utils import load_filepaths_and_text, load_wav_to_torch

class TextAudioLoader(tdata.Dataset):
    """
    A PyTorch Dataset that loads audio filepaths, transcriptions, and optional structural 
    pitch labels for voice conversion training pipelines.
    """

    def __init__(
        self, 
        hparams, 
        spec_dirs=None,
        cache_spectrogram=True,
        pitch_guidance=True, 
    ):
        """
        Initializes the dataset configuration and triggers file filtering.
        """

        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.filter_length = hparams.filter_length
        self.sample_rate = hparams.sample_rate
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.cache_spectrogram = cache_spectrogram
        self.pitch_guidance = pitch_guidance
        self.spec_dirs = spec_dirs
        self._filter()

    def _filter(self):
        """
        Filters out dataset entries that do not comply with sequence length bounds 
        and pre-computes rough estimate audio length dimensions for bucket sorting.
        """

        audiopaths_and_text_new, lengths = [], []

        for item in self.audiopaths_and_text:
            audiopath = item[0]
            text = item[1]
            # Validate structural bounds constraints on text token length
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append(item)
                # Calculate estimated frame lengths based on raw disk file size
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))

        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        """
        Parses and casts a speaker identifier securely into a target LongTensor block.

        Args:
            sid: Raw speaker identifier.

        Returns:
            torch.LongTensor: Long Tensor wrapped single-element indexing vector.
        """

        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError:
            sid = torch.LongTensor([0]) # Fallback defaults to speaker 0 on conversion errors

        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        """
        Constructs a runtime training sample dictionary item tuple by unpacking path keys, 
        reading label buffers, and slicing timeline frame misalignments.

        Args:
            audiopath_and_text (List[str]): Metadata items line parsed from file manifests.

        Returns:
            Tuple[torch.Tensor, ...]: Packed feature elements for training steps.
        """

        extra = audiopath_and_text[2:]
        pitch, pitchf, sid = None, None, None

        if self.pitch_guidance: pitch, pitchf, sid = extra
        else: sid = extra[0]

        spec, wav = self.get_audio(audiopath_and_text[0])
        dv = self.get_sid(sid)

        phone, pitch, pitchf = self.get_labels(
            audiopath_and_text[1],
            pitch=pitch,
            pitchf=pitchf
        )

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        # Enforce exact axis sequence alignment across features due to processing edge offsets
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec, wav, phone = spec[:, :len_min], wav[:, :len_wav], phone[:len_min, :]
            if self.pitch_guidance: pitch, pitchf = pitch[:len_min], pitchf[:len_min]

        outputs = [spec, wav, phone, dv]
        if self.pitch_guidance: outputs[3:3] = [pitch, pitchf]

        return tuple(outputs)

    def get_labels(self, phone, pitch=None, pitchf=None):
        """
        Loads pre-extracted phonetic hidden features and pitch configurations from numpy files.

        Args:
            phone (str): Filepath to phoneme representation array.
            pitch (str, optional): Filepath to integer pitch arrays. Defaults to None.
            pitchf (str, optional): Filepath to float continuous pitch arrays. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Torched label features.
        """

        # Feature vectors are typically extracted at double hop sizes, upsample via repeating
        phone = np.repeat(np.load(phone), 2, axis=0)
        n_num = min(phone.shape[0], 900) # Hard cap sample sequence ceiling block constraints

        return (
            torch.FloatTensor(phone[:n_num, :]), 
            torch.LongTensor(np.load(pitch)[:n_num]) if pitch else None, 
            torch.FloatTensor(np.load(pitchf)[:n_num]) if pitchf else None
        )

    def get_audio(self, filename):
        """
        Loads a raw audio waveform and computes/loads its associated spectrogram.

        Args:
            filename (str): Audio source filename path.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Squeezed Spectrogram and raw normalized waveform.
        """

        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.sample_rate: 
            raise ValueError(translations["sr_does_not_match"].format(sample_rate=sample_rate, sample_rate2=self.sample_rate))

        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.spec_dirs: spec_filename = os.path.join(self.spec_dirs, os.path.basename(spec_filename))

        def get_spectrogram(audio_norm):
            return spectrogram_torch(
                audio_norm, 
                self.filter_length, 
                self.hop_length, 
                self.win_length, 
                center=False
            ).squeeze(0)

        # Handle persistent disk caching mechanics for spectrogram structures
        if not self.cache_spectrogram:
            spec = get_spectrogram(audio_norm)
        elif os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=True)
            except Exception:
                spec = get_spectrogram(audio_norm)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = get_spectrogram(audio_norm)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)

        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollate:
    """
    Collate function to dynamic-pad and bundle variable-length sequence elements 
    into a uniformly padded tensor batch.
    """

    def __init__(self, return_ids=False, pitch_guidance=True):
        self.return_ids = return_ids
        self.pitch_guidance = pitch_guidance

    def __call__(self, batch):
        """
        Sorts the batch by decreasing structural timeline frame lengths, 
        allocates padded tracking tensors, and performs linear target mask copy injections.
        """

        # Sort batch by decreasing sequence length to satisfy sequence padding constraints down the pipe
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)
        spec_lengths, wave_lengths = torch.LongTensor(len(batch)), torch.LongTensor(len(batch))
        spec_padded, wave_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max([x[0].size(1) for x in batch])), torch.FloatTensor(len(batch), 1, max([x[1].size(1) for x in batch]))
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths, phone_padded = torch.LongTensor(len(batch)), torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        phone_padded.zero_()

        if self.pitch_guidance:
            pitch_padded, pitchf_padded = torch.LongTensor(len(batch), max_phone_len), torch.FloatTensor(len(batch), max_phone_len)
            pitch_padded.zero_()
            pitchf_padded.zero_()

        sid = torch.LongTensor(len(batch))
        # Iteratively inject sample representations into allocating memory views
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]

            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            if self.pitch_guidance:
                pitch = row[3]
                pitch_padded[i, : pitch.size(0)] = pitch
                pitchf = row[4]
                pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5 if self.pitch_guidance else 3]

        outputs = [phone_padded, phone_lengths, spec_padded, spec_lengths, wave_padded, wave_lengths, sid]
        if self.pitch_guidance: outputs[2:2] = [pitch_padded, pitchf_padded]

        return tuple(outputs)

class DistributedBucketSampler(tdata.distributed.DistributedSampler):
    """
    A distributed sampler that bundles together elements of similar length profiles 
    into variable sequence length clusters ("buckets") to reduce computing resource padding overheads.
    """

    def __init__(
        self, 
        dataset, 
        batch_size, 
        boundaries, 
        num_replicas=None, 
        rank=None, 
        shuffle=True
    ):
        super().__init__(
            dataset, 
            num_replicas=num_replicas, 
            rank=rank, 
            shuffle=shuffle
        )
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        """
        Bins element indices based on historical duration bounds constraints.
        """

        buckets = [[] for _ in range(len(self.boundaries) - 1)]

        for i in range(len(self.lengths)):
            idx_bucket = self._bisect(self.lengths[i])
            if idx_bucket != -1: buckets[idx_bucket].append(i)

        # Drop empty buckets to avoid zero division step bugs
        for i in range(len(buckets) - 1, -1, -1):  
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []

        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size

            # Pad size calculation calculations to match exact distributed rank dimensions cleanly
            num_samples_per_bucket.append(
                len_bucket + ((total_batch_size - (len_bucket % total_batch_size)) % total_batch_size)
            )

        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices, batches = [], []

        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]

            # Balance lists by padding index loops uniformly where remaining values exist
            rem = self.num_samples_per_bucket[i] - len_bucket
            ids_bucket = (
                ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]
            )[self.rank :: self.num_replicas]

            # Split up individual bucket entries into clean mini-batch size blocks
            for j in range(len(ids_bucket) // self.batch_size):
                batches.append([bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]])

        # Shuffle the global batch presentation layout if required
        if self.shuffle: batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples

        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        """
        Binary search structure lookup utility to trace matching bucket intervals.
        """

        if hi is None: hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2

            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]: return mid
            elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
            else: return self._bisect(x, mid + 1, hi)
        else: return -1

    def __len__(self):
        return self.num_samples // self.batch_size

def get_training_dataloader(config, spec_dir, cache_spectrogram, pitch_guidance = True, architecture = "RVC", batch_size = 8, n_gpus = 1, rank = 0):
    """
    Factory helper constructor instantiated to provide configured data loading pipelines.

    Args:
        config (object): Hparams configuration wrapper namespace block.
        spec_dir (str): Target folder path directory containing cached components.
        cache_spectrogram (bool): Toggle option controlling disk write/read cycles.
        pitch_guidance (bool): Toggle option tracking pitch lines. Defaults to True.
        architecture (str): Model framework context variant. Defaults to "RVC".
        batch_size (int): Mini-batch constraint count per device worker. Defaults to 8.
        n_gpus (int): Total active computational processing nodes. Defaults to 1.
        rank (int): Host identifier index reference tracking. Defaults to 0.

    Returns:
        DataLoader: Fully configured PyTorch data loader structure.
    """

    train_dataset = TextAudioLoader(config.data, spec_dirs=spec_dir, cache_spectrogram=cache_spectrogram, pitch_guidance=pitch_guidance)

    train_loader = DataLoader(
        train_dataset, 
        num_workers=4, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=1 if architecture != "SVC" else batch_size,
        collate_fn=TextAudioCollate(pitch_guidance=pitch_guidance), 
        batch_sampler=DistributedBucketSampler(
            train_dataset, 
            batch_size, 
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900], 
            num_replicas=n_gpus, 
            rank=rank, 
            shuffle=True
        ) if architecture != "SVC" else None, 
        persistent_workers=True, 
        prefetch_factor=8
    )

    return train_loader