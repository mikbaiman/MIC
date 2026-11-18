"""Run an unchanged task entry point in its expected working directory."""
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
root = Path(__file__).resolve().parents[1]
if len(sys.argv) < 3 or sys.argv[1] not in ("cls", "seg", "det"):
    raise SystemExit("Usage: bash uv/run.sh {cls|seg|det} script.py [arguments...]")
profile, filename = sys.argv[1:3]
working_dir = root / profile / "examples" if profile == "cls" else root / profile
script = (working_dir / filename).resolve()
if not script.is_file():
    raise SystemExit("Entry point not found: {}".format(script))
sys.path[:0] = [str(working_dir), str(root / profile), str(script.parent)]
if profile == "det":
    import maskrcnn_benchmark

    maskrcnn_benchmark.__path__.insert(0, str(root / "uv/det/build/lib/maskrcnn_benchmark"))
    # Upstream INSTALL.md asks to copy this helper into cityscapesScripts.
    # A package search path provides the same original helper without copying.
    import cityscapesscripts.evaluation

    cityscapesscripts.evaluation.__path__.insert(0, str(root / "det/tools/cityscapes"))
os.chdir(working_dir)
sys.argv = [str(script)] + sys.argv[3:]
runpy.run_path(str(script), run_name="__main__")
