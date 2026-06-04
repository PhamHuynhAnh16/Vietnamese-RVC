import os
import sys
import faiss
import torch

sys.path.append(os.getcwd())

from main.app.variables import translations, logger

class IndexWrapper:
    def __init__(self, index_path, nprobe = 1, device="cuda", is_half=False, faiss_cpu=False, clamp=1e-8):
        self.index_path = index_path
        self.nprobe = nprobe
        self.device = device
        self.clamp = clamp
        self.index = None
        self.big_npy = None
        self.b_norms = None
        self.big_tensor = None

        self.dtype = torch.float16 if is_half else torch.float32
        self.search = self._search_cpu if faiss_cpu else self._search_gpu
    
    def read_index(self):
        if self.index_path != "" and os.path.exists(self.index_path):
            try:
                index = faiss.read_index(self.index_path)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        if index is not None: index.nprobe = self.nprobe
        self.index = index
        self.big_npy = big_npy

        return index, big_npy
    
    def read_index_tensor(self):
        self.read_index()

        if self.index is None or self.big_npy is None: self.big_tensor, self.b_norms = None, None
        else:
            self.big_tensor = torch.from_numpy(self.big_npy).to(self.dtype).to(self.device).contiguous()
            self.b_norms = (self.big_tensor ** 2).sum(dim=-1, keepdim=True).T.contiguous()

        return self.big_tensor, self.b_norms
    
    def setup_cpu(self, query, k=8):
        logger.warning(translations["index_warn"])

        self.search = self._search_cpu
        self.b_norms = None

        from main.library.utils import clear_gpu_cache
        clear_gpu_cache()

        return self._search_cpu(query, k)

    def _search_gpu(self, query, k=8):
        with torch.inference_mode():
            try:
                q_norm = (query ** 2).sum(dim=-1, keepdim=True)

                distances = torch.addmm(self.b_norms, query, self.big_tensor.T, alpha=-2.0, beta=1.0)
                distances.add_(q_norm)
                distances.clamp_(min=self.clamp)

                scores, indices = torch.topk(-distances, k=k, dim=-1)
                return -scores, indices
            except (torch.OutOfMemoryError, RuntimeError):
                return self.setup_cpu(query, k)

    def _search_cpu(self, query, k=8):
        score, ix = self.index.search(query.cpu().numpy(), k)
        return torch.from_numpy(score).to(self.dtype).to(self.device), torch.from_numpy(ix).to(self.device).long()
