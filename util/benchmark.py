import logging
import os, sys
import types
from datetime import datetime
import pandas as pd
from .data import load_dataset
from .evaluate import compute_scores
from .evaluate_cluster import compute_cluster_scores
from .base import BaseSynthesizer
import warnings
warnings.filterwarnings(action='ignore')



def compute_benchmark(synthesizer, dataset_name):
    """
    compute last scores of Clustering and scores of Supervised Learning
    """
    train, test, meta, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)
    synthesized = synthesizer(train, test, meta, dataset_name, categoricals, ordinals)
    
    scores = compute_scores(train, test, synthesized, meta)
    if 'likelihood' in meta["problem_type"]:
        return scores
    scores_cluster = compute_cluster_scores(train, test, synthesized, meta)
    return scores, scores_cluster


def benchmark(synthesizer, syn_arg, dataset):
    """
    Args:
        synthesizer : synthesizer to test
        syn_arg:       argument of synthesizer 
        dataset:      data to test
    Return:
        compute last scores of Clustering and scores of Supervised Learning
    """
    synthesizer = synthesizer(**syn_arg).fit_sample
    return compute_benchmark(synthesizer, dataset)
