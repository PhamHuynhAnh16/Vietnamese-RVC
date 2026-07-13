import os
import sys
import faiss
import argparse

import numpy as np

from multiprocessing import cpu_count
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, configs

def parse_arguments():
    """
    Parses command-line arguments configuring the FAISS index pipeline.

    Returns:
        argparse.Namespace: Object containing validated operational configurations.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--create_index", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--index_algorithm", type=str, default="Auto")
    parser.add_argument("--nprobe", type=int, default=1)

    return parser.parse_args()

def main():
    """
    Executes core vector gathering, clustering optimization, and FAISS index model export pipeline.
    """

    args = parse_arguments()
    # Resolve target asset directories and variables
    exp_dir = os.path.join(configs["logs_path"], args.model_name)
    version, index_algorithm, nprobe = args.rvc_version, args.index_algorithm, args.nprobe
    # Construct execution configuration parameters map for logging
    log_data = {
        translations['modelname']: args.model_name, 
        translations['model_path']: exp_dir, 
        translations['training_version']: version, 
        translations['index_algorithm_info']: index_algorithm,
        translations['nprobe']: nprobe
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    try:
        npys = []
        feature_dir = os.path.join(exp_dir, f"{version}_extracted")
        model_name = os.path.basename(exp_dir)

        # Scan and aggregate extracted feature vector arrays
        for name in sorted(os.listdir(feature_dir)):
            npys.append(np.load(os.path.join(feature_dir, name)))

        # Unify collection lists into one continuous multi-dimensional matrix stack
        big_npy = np.concatenate(npys, axis=0)
        big_npy_idx = np.arange(big_npy.shape[0])
        # Randomize vector index arrangements to eliminate localized sample ordering bias
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        # Downsample extremely massive collections through k-means centroid quantization
        if big_npy.shape[0] > 2e5 and (index_algorithm == "Auto" or index_algorithm == "KMeans"): 
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000, 
                    verbose=True, 
                    batch_size=256 * cpu_count(), 
                    compute_labels=False, 
                    init="random"
                ).fit(big_npy).cluster_centers_
            )

        # Export aggregated features baseline matrix
        np.save(
            os.path.join(exp_dir, "total_fea.npy"), 
            big_npy
        )
        # Dynamically compute optimal Inverted File (IVF) cell partitioning scale counts
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)

        # Phase 1: Build and Train structural baseline Index
        index_trained = faiss.index_factory(
            256 if version == "v1" else 768, # Resolve target dimension widths (v1 features use 256, v2 architectures use 768)
            f"IVF{n_ivf},Flat"
        )

        index_ivf_trained = faiss.extract_index_ivf(index_trained)
        index_ivf_trained.nprobe = nprobe
        # Fit vector quantization spaces
        index_trained.train(big_npy)

        faiss.write_index(
            index_trained, 
            os.path.join(
                exp_dir, 
                f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf_trained.nprobe}_{model_name}_{version}.index"
            )
        )

        # Phase 2: Build, Train, and Populate Added-Vectors Index
        index_added = faiss.index_factory(
            256 if version == "v1" else 768, 
            f"IVF{n_ivf},Flat"
        )

        index_ivf_added = faiss.extract_index_ivf(index_added)
        index_ivf_added.nprobe = nprobe
        index_added.train(big_npy)

        # Incrementally stream clustered matrices sequentially into index vector tables
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index_added.add(
                big_npy[i : i + batch_size_add]
            )

        index_filepath_added = os.path.join(
            exp_dir, 
            f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf_added.nprobe}_{model_name}_{version}.index"
        )

        faiss.write_index(
            index_added, 
            index_filepath_added
        )

        logger.info(f"{translations['save_index']} '{index_filepath_added}'")
    except Exception as e:
        logger.error(f"{translations['create_index_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__": main()