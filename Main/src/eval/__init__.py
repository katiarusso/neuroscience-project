"""Evaluation utilities used by every modelling notebook.

The neuron-level aggregation contract from `WORKFLOW.md §6.3` is implemented
here as `aggregate_probs_to_neuron` and `neuron_level_score`. Importing from
this module guarantees that every model run uses the exact same metric
definition.
"""
