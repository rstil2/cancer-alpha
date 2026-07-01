"""Study 1 pipeline configuration (small-n, n≈158, 110 features, KIRC panel)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "production_tcga"
CLINICAL_CACHE = PROJECT_ROOT / "data_integration" / "tcga_large_cache" / "clinical"
OUTPUT_DIR = PROJECT_ROOT / "data" / "study1_results"
MAPPING_CSV = OUTPUT_DIR / "file_patient_mapping_study1.csv"

# Study 1 cancer panel (KIRC, not LUSC)
CANCER_TYPES = [
    "TCGA-BRCA",
    "TCGA-LUAD",
    "TCGA-COAD",
    "TCGA-PRAD",
    "TCGA-STAD",
    "TCGA-HNSC",
    "TCGA-KIRC",
    "TCGA-LIHC",
]

# Manuscript imbalanced cohort targets (sum = 158)
TARGET_CLASS_COUNTS = {
    "TCGA-BRCA": 32,
    "TCGA-LUAD": 28,
    "TCGA-COAD": 22,
    "TCGA-PRAD": 19,
    "TCGA-STAD": 17,
    "TCGA-KIRC": 15,
    "TCGA-HNSC": 14,
    "TCGA-LIHC": 11,
}

FEATURE_COUNTS = {
    "methylation": 20,
    "mutation": 25,
    "copy_number": 20,
    "fragmentomics": 15,
    "clinical": 10,
    "icgc_expression": 20,
}

TOTAL_FEATURES = sum(FEATURE_COUNTS.values())  # 110

# Top CpGs / genes selected on first full pass and frozen in output_dir
TOP_METHYLATION_PROBES = 20
TOP_MUTATION_GENES = 22  # + 3 summary stats = 25

MUTATION_GENES = [
    "TP53", "KRAS", "PIK3CA", "APC", "PTEN", "BRAF", "EGFR", "CDKN2A",
    "RB1", "BRCA1", "BRCA2", "MLH1", "MSH2", "VHL", "ARID1A", "CTNNB1",
    "ERBB2", "FBXW7", "NRAS", "SMAD4", "STK11", "GATA3",
]

ICGC_EXPRESSION_GENES = [
    "ESR1", "ERBB2", "EGFR", "KRAS", "BRAF", "PIK3CA", "TP53", "BRCA1",
    "BRCA2", "MET", "ALK", "RET", "FGFR2", "CDK4", "CCND1", "MDM2",
    "PTEN", "STK11", "NRAS", "MYC",
]

RANDOM_STATE = 42
N_CV_FOLDS = 5
N_INDEPENDENT_RUNS = 10
SMOTE_K_NEIGHBORS = 4
