# ============================================================
# IMPORTS
# ============================================================
import torch
import esm
import pandas as pd
import joblib
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONSTANTS
# ============================================================
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================
def load_model():
    base_path = os.path.dirname(__file__)

    model_path = os.path.join(base_path, "model", "XGB_based_toxicity_model.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError("XGBoost model file not found.")

    return joblib.load(model_path)


# ============================================================
# LOAD ESM MODEL
# ============================================================
def load_esm():
    print("Loading ESM-2 model...")

    try:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    except Exception as e:
        raise RuntimeError("Failed to load ESM-2 model. Check internet connection.") from e

    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, batch_converter, device


# ============================================================
# READ FASTA
# ============================================================
def read_fasta(file_path):
    sequences = []
    headers = []

    try:
        with open(file_path, "r") as f:
            seq = ""
            header = None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    if seq:
                        sequences.append(seq)
                        headers.append(header)
                    header = line
                    seq = ""
                else:
                    seq += line

            if seq:
                sequences.append(seq)
                headers.append(header)

    except Exception as e:
        raise RuntimeError(f"Error reading input file: {file_path}") from e

    if not sequences:
        raise ValueError("No valid sequences found in input file.")

    return sequences, headers


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================
def generate_embeddings(sequences, esm_model, batch_converter, device, batch_size=16):

    print("Generating embeddings...")

    cleaned_sequences = []
    valid_data = []

    for i, seq in enumerate(sequences):
        clean_seq = "".join([aa for aa in seq.upper() if aa in VALID_AA])
        cleaned_sequences.append(clean_seq)

        if 2 <= len(clean_seq) <= 50:
            valid_data.append((i, clean_seq))

    embeddings = {}

    for i in tqdm(range(0, len(valid_data), batch_size), desc="Embedding", ncols=80):
        batch = valid_data[i:i + batch_size]
        batch_input = [("seq", seq) for _, seq in batch]

        _, _, tokens = batch_converter(batch_input)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[33])

        token_embeddings = results["representations"][33]

        for j, (idx, seq) in enumerate(batch):
            emb = token_embeddings[j, 1:len(seq)+1].mean(0)
            embeddings[idx] = emb.cpu().numpy()

    return embeddings, cleaned_sequences


# ============================================================
# MAIN FUNCTION
# ============================================================
def run_prediction(input_file, output_file="toxesm_results.csv"):

    print("Reading input sequences...")
    sequences, headers = read_fasta(input_file)

    total = len(sequences)
    print(f"Total sequences: {total}")

    print("Loading XGBoost model...")
    model = load_model()

    esm_model, batch_converter, device = load_esm()

    embeddings, cleaned_sequences = generate_embeddings(
        sequences, esm_model, batch_converter, device
    )

    results = []

    print("Running predictions...")

    for i, (seq, header) in enumerate(zip(sequences, headers)):

        clean_seq = cleaned_sequences[i]

        try:
            if len(clean_seq) < 2 or len(clean_seq) > 50 or i not in embeddings:
                results.append({
                    "Header": header,
                    "Sequence": clean_seq,
                    "Prediction": "Invalid",
                    "Score": "NA"
                })
                continue

            X = pd.DataFrame([embeddings[i]])

            final_score = model.predict_proba(X)[0][1]
            final_label = "Toxic" if final_score >= 0.5 else "Non-Toxic"

            results.append({
                "Header": header,
                "Sequence": clean_seq,
                "Prediction": final_label,
                "Score": round(float(final_score), 6)
            })

        except Exception as e:
            results.append({
                "Header": header,
                "Sequence": clean_seq,
                "Prediction": "Error",
                "Score": str(e)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\nSummary:")
    counts = df["Prediction"].value_counts()

    for key in ["Toxic", "Non-Toxic", "Invalid", "Error"]:
        if key in counts:
            print(f"{key:<12}: {counts[key]}")

    print(f"\nResults saved to: {output_file}")