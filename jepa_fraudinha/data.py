import argparse
import json
import multiprocessing
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


class FilesToCreate(TypedDict):
    type: str
    idx: NDArray[np.int_]


class SaveCompressedFileJob(FilesToCreate):
    output: Path
    vectors: NDArray[np.float32]
    missing_mask: NDArray[np.float32]
    labels: NDArray[np.float32]


def save_compressed_file(job: SaveCompressedFileJob) -> None:
    np.savez_compressed(
        job["output"] / f"{job['type']}.npz",
        job["vectors"],
        job["missing_mask"],
        job["labels"],
        job["idx"]
    )


def prepare_data(args: argparse.Namespace) -> None:
    input = Path(args.input)
    output = Path(args.output)
    val_size = args.val_size
    test_size = args.test_size
    train_size = args.train_size
    seed = args.seed

    output.mkdir(parents=True, exist_ok=True)

    with input.open("r") as f:
        records = json.load(f)

    labels_list = []
    vectors_list = []
    missing_mask_list = []

    for record in records:
        clean_vector = []
        missing_mask = []

        for vector in record["vector"]:
            if vector == -1:
                clean_vector.append(0.0)
                missing_mask.append(1.0)
            else:
                clean_vector.append(vector)
                missing_mask.append(0.0)

        missing_mask_list.append(missing_mask)
        vectors_list.append(clean_vector)
        labels_list.append(
                1 if record["label"] == "fraud" else 0
        )

    vectors = np.array(
            vectors_list,
            dtype=np.float32
    )

    missing_mask = np.array(
            missing_mask_list,
            dtype=np.float32
    )

    labels = np.array(
            labels_list,
            dtype=np.float32
    )

    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        train_size=train_size,
        stratify=labels,
        random_state=seed
    )

    val_ratio_inside_temp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio_inside_temp,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    files_to_create: list[SaveCompressedFileJob] = [{
        "type": "train",
        "idx": train_idx,
        "output": output,
        "vectors": vectors,
        "missing_mask": missing_mask,
        "labels": labels,
    }, {
        "type": "val",
        "idx": val_idx,
        "output": output,
        "vectors": vectors,
        "missing_mask": missing_mask,
        "labels": labels,
    }, {
        "type": "test",
        "idx": test_idx,
        "output": output,
        "vectors": vectors,
        "missing_mask": missing_mask,
        "labels": labels,
    }]
    processes = min(len(files_to_create), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(save_compressed_file, files_to_create)

    stats = {
        "total": int(len(labels)),
        "features": int(vectors.shape[1]),
        "fraud_count": int(labels.sum()),
        "legit_count": int(len(labels) - labels.sum()),
        "fraud_rate": float(labels.mean()),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "missing_count_by_feature": 
            missing_mask.sum(axis=0).astype(int).tolist()
    }

    with (output / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
