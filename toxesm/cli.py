# ============================================================
# IMPORTS
# ============================================================
import argparse
import os
import sys
import warnings

from toxesm.predict import run_prediction

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        prog="toxesm",
        description="ToxESM: Peptide toxicity prediction using ESM-2 embeddings and XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Path to input FASTA file"
    )

    parser.add_argument(
        "--output",
        default="toxesm_results.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    print("\nToxESM - Peptide Toxicity Prediction (XGBoost)\n")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found -> {args.input}")
        sys.exit(1)

    print(f"Input file : {args.input}")
    print(f"Model      : XGBoost")
    print(f"Output file: {args.output}\n")

    try:
        print("Running prediction...\n")

        run_prediction(
            input_file=args.input,
            output_file=args.output
        )

        print("\nPrediction completed successfully.")
        print(f"Results saved to: {args.output}\n")

    except Exception as e:
        print("\nPrediction failed.")
        print(f"Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()