import os
import sys
import faiss
import torch

sys.path.append(os.getcwd())

from main.app.variables import translations, logger

class IndexWrapper:
    """
    A lightweight wrapper for loading, converting, and searching FAISS indexes using PyTorch tensors.

    Supports:
    - Reading a FAISS index
    - Converting vectors to tensors
    - Performing brute-force L2 distance search with PyTorch
    """

    def __init__(self, index_path, nprobe = 1, device="cuda", is_half=False, faiss_cpu=False, clamp=1e-8):
        """
        Initializes the IndexWrapper with paths, hardware options, and calculation parameters.

        Args:
            index_path (str): File system path to the serialized `.index` FAISS file.
            nprobe (int): Number of clusters to query during Voronoi cells lookup. Defaults to 1.
            device (str): Destination hardware string ("cuda" or "cpu"). Defaults to "cuda".
            is_half (bool): If True, uses FP16 precision instead of FP32. Defaults to False.
            faiss_cpu (bool): Explicitly forces the lookup strategy to rely on CPU. Defaults to False.
            clamp (float): Lower constraint threshold value to avoid zero/negative distances. Defaults to 1e-8.
        """

        self.index_path = index_path
        self.nprobe = nprobe
        self.device = device
        self.clamp = clamp
        # In-memory structural database placeholders
        self.index = None
        self.big_npy = None
        self.b_norms = None
        self.big_tensor = None
        # Data type definition mapping
        self.dtype = torch.float16 if is_half else torch.float32
        # Dynamically map the main search execution routing function path
        self.search = self._search_cpu if faiss_cpu else self._search_gpu
    
    def read_index(self):
        """
        Loads the FAISS index file from storage and extracts underlying target vector arrays.

        Returns:
            Tuple containing:
                - index (faiss.Index or None): Initialized FAISS index struct.
                - big_npy (np.ndarray or None): Matrix of full dataset vectors.
        """

        if self.index_path != "" and os.path.exists(self.index_path):
            try:
                # Load serialized structural mapping records from disk
                index = faiss.read_index(self.index_path)
                # Reconstruct full flat float arrays representing all registered indices
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        # Set search probe cluster configuration depth limits if parsing was successful.
        # Basically, it is only useful when you perform the search using the CPU.
        if index is not None: index.nprobe = self.nprobe

        self.index = index
        self.big_npy = big_npy

        return index, big_npy
    
    def read_index_tensor(self):
        """
        Initializes structural index resources and maps feature arrays directly to VRAM.

        Returns:
            Tuple containing:
                - big_tensor (torch.Tensor or None): High speed VRAM feature table tensor.
                - b_norms (torch.Tensor or None): Precomputed horizontal squared L2 norm vectors.
        """

        self.read_index()

        if self.index is None or self.big_npy is None: self.big_tensor, self.b_norms = None, None
        else:
            # Transfer flat vector arrays to GPU for accelerated matrix ops
            self.big_tensor = torch.from_numpy(self.big_npy).to(self.dtype).to(self.device).contiguous()
            # Precompute the squared L2 norm matrix to optimize distance calculation processes.
            self.b_norms = (self.big_tensor ** 2).sum(dim=-1, keepdim=True).T.contiguous()

        return self.big_tensor, self.b_norms
    
    def setup_cpu(self, query, k=8):
        """
        Emergency fallback setup triggered when VRAM bounds are exceeded.
        Frees up GPU spaces and routes operations back to host resources.

        Args:
            query (torch.Tensor): Vector query input data tensor.
            k (int): Number of top items to fetch. Defaults to 8.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Search metrics score outputs from CPU execution fallback.
        """

        logger.warning(translations["index_warn"])
        # Permanently shift standard execution mapping to safe CPU paths
        self.search = self._search_cpu
        self.b_norms = None

        # Dynamically call garbage collections to purge stranded allocations
        from main.library.utils import clear_gpu_cache
        clear_gpu_cache()

        return self._search_cpu(query, k)

    def _search_gpu(self, query, k=8):
        """
        Executes fast batched vector distance queries natively on VRAM devices 
        utilizing algebraic expansions

        Args:
            query (torch.Tensor): Query vectors matrix tensor.
            k (int): Total near neighbors tracking depth. Defaults to 8.

        Returns:
            Tuple containing:
                - distances (torch.Tensor): Calculated smallest scalar distances matrix.
                - indices (torch.Tensor): Index positions pointing to closest target vectors.
        """

        with torch.inference_mode():
            try:
                # 1. Compute squared norms of the input queries: ||A||^2
                q_norm = (query ** 2).sum(dim=-1, keepdim=True)

                # 2. Perform matrix multiplication trick: distances = beta * ||B||^2 + alpha * (A @ B^T)
                # Compiles down to a single optimized fused GPU calculation block
                distances = torch.addmm(self.b_norms, query, self.big_tensor.T, alpha=-2.0, beta=1.0)

                # 3. Complete expansion: distances = (||B||^2 - 2AB^T) + ||A||^2
                distances.add_(q_norm)
                distances.clamp_(min=self.clamp) # Clamp to handle rounding anomalies

                # 4. Extract smallest elements by utilizing TopK over negative fields
                scores, indices = torch.topk(-distances, k=k, dim=-1)
                return -scores, indices
            except (torch.OutOfMemoryError, RuntimeError):
                # Gracefully intercept hardware runtime execution errors and trigger CPU routing routines
                return self.setup_cpu(query, k)

    def _search_cpu(self, query, k=8):
        """
        Standard fallback execution method processing similarity queries over host system CPU spaces.

        Args:
            query (torch.Tensor): Vector query input tensor.
            k (int): Target neighbor retrieval matching limit. Defaults to 8.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Score results array and associated mapping indices locations.
        """

        # Execute traditional index structure searching out of NumPy array references
        score, ix = self.index.search(query.cpu().numpy(), k)

        # Cast results back to native environment storage targets
        return torch.from_numpy(score).to(self.dtype).to(self.device), torch.from_numpy(ix).to(self.device).long()