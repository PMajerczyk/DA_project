"""Execute one or more notebooks in place, with notebooks/ as the working dir."""
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

for path in sys.argv[1:]:
    print(f"=== executing {path} ===", flush=True)
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=2400, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path) or "."}})
    nbformat.write(nb, path)
    print(f"=== OK {path} ===", flush=True)
