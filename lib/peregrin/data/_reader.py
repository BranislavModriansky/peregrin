from pathlib import Path
import sys

# Works both as package module and as direct script
try:
    from ..src.io.load import load_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from peregrin.src.io.load import load_data

DATA_DIR = Path(__file__).resolve().parent



columns = {'id': "TRACK_ID", 't': "POSITION_T", 'x': "POSITION_X", 'y': "POSITION_Y"}
 

naive_ctr = load_data(
    files=[
        str(DATA_DIR / "naive_ctr_BC39.csv"),
        str(DATA_DIR / "naive_ctr_BC42.csv"),
        str(DATA_DIR / "naive_ctr_BC43.csv"),
    ],
    columns=columns,
    cond_label="ctr",
)

naive_cxcl12 = load_data(
    files=[
        str(DATA_DIR / "naive_cxcl12_BC39.csv"),
        str(DATA_DIR / "naive_cxcl12_BC42.csv"),
        str(DATA_DIR / "naive_cxcl12_BC43.csv"),
    ],
    columns=columns,
    cond_label="cxcl12",
)

naive_mu = load_data(
    files=[
        str(DATA_DIR / "naive_mu_BC39.csv"),
        str(DATA_DIR / "naive_mu_BC42.csv"),
        str(DATA_DIR / "naive_mu_BC43.csv"),
    ],
    columns=columns,
    cond_label="mu",
)
