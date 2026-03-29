from setuptools import setup, find_packages

setup(
    name="toxesm",
    version="0.1.0",
    description="Transformer-based peptide toxicity prediction using ESM-2 and XGBoost",
    author="Abhishek Saha",

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        "torch",
        "fair-esm",
        "pandas",
        "xgboost",
        "tqdm",
        "biopython",
        "joblib",
        "scikit-learn"   # keep this (XGBoost sklearn wrapper dependency)
    ],

    entry_points={
        "console_scripts": [
            "toxesm=toxesm.cli:main"
        ]
    },

    python_requires=">=3.8",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

    keywords="peptide toxicity prediction ESM protein machine learning bioinformatics",
)