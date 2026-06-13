import os
from os.path import join

import data.reflect_dataset as datasets


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "testdata_CEILNET_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "objects",
        "save_subdir": "SIR2_objects",
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "postcard",
        "save_subdir": "SIR2_postcard",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "wild",
        "save_subdir": "SIR2_wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
}


def build_eval_loader(
    opt,
    data_root,
    dataset_key,
    size=None,
    max_long_edge=None,
):
    spec = EVAL_DATASETS[dataset_key]
    effective_long_edge = (
        max_long_edge
        if max_long_edge is not None
        else spec.get("max_long_edge")
    )
    dataset = datasets.CEILTestDataset(
        join(data_root, spec["path"]),
        size=size,
        max_long_edge=effective_long_edge,
    )
    loader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )
    return spec, loader


def image_name(data, index):
    value = data.get("fn", str(index))
    if isinstance(value, (list, tuple)):
        value = value[0]
    return os.path.splitext(os.path.basename(str(value)))[0]
