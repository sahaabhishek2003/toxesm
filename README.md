# ToxESM

Transformer-based peptide toxicity prediction using ESM-2 embeddings and XGBoost.

---

## Overview

ToxESM is a command-line tool for predicting peptide toxicity from amino acid sequences.  
It combines protein language model embeddings (ESM-2) with a trained XGBoost classifier.

The tool is designed to be simple, fast, and easy to use for researchers in bioinformatics.

---

## Features

- Uses ESM-2 embeddings for sequence representation
- XGBoost-based toxicity classification
- Accepts FASTA input files
- Outputs results in CSV format
- Simple command-line interface

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/sahaabhishek2003/toxesm.git
cd toxesm
pip install .
