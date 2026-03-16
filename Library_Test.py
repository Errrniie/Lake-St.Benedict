#!/usr/bin/env python3
"""Test script to verify all required libraries import successfully."""

import pandas
import numpy
import lightgbm
import sklearn
import matplotlib
import seaborn
import joblib

print("✅ All libraries imported successfully!")
print(f"  - pandas {pandas.__version__}")
print(f"  - numpy {numpy.__version__}")
print(f"  - lightgbm {lightgbm.__version__}")
print(f"  - scikit-learn {sklearn.__version__}")
print(f"  - matplotlib {matplotlib.__version__}")
print(f"  - seaborn {seaborn.__version__}")
print(f"  - joblib {joblib.__version__}")
