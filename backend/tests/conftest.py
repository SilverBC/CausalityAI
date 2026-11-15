import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("CAUSAL_LIMITS", "off")   # tests that exercise the demo limits turn them on themselves
