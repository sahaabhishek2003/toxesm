"""
ToxESM - Peptide Toxicity Prediction Package

This package provides functionality to predict peptide toxicity
using ESM-2 embeddings and machine learning models.
"""

from .predict import run_prediction

__all__ = ["run_prediction"]
