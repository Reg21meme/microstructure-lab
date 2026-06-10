import sys
import pathlib

# Add cpp/build to sys.path so mslab_bindings.so can be imported
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "cpp" / "build"))