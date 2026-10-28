"""Build ``open_job.ipynb``: the one-click Colab launcher (run after editing the cells here).

    python colab/build_launcher.py

GPU Relay links to this notebook on GitHub (``COLAB_LAUNCHER_URL``), so it opens in Colab
with one click. It holds no job and no secret: when the user presses Run all, it reads their
Hugging Face token from their Colab secret, finds the job that GPU Relay asked to run on
Colab in their storage repo, and runs that job's resume notebook here. GPU Relay never
controls Colab; the user opens this notebook and presses Run all.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

INTRO = """# GPU Relay: run your job on Colab

GPU Relay asked for a Colab session for one of your jobs. This notebook finds that job in your
private storage on Hugging Face and continues it here, from its latest checkpoint.

**Once per Colab account:** click the key icon (*Secrets*) on the left, add a secret named
`GPURELAY_HF_TOKEN` whose value is your Hugging Face token (write access), and turn on
*Notebook access*. If your storage repo is not named `gpurelay-store`, also add a secret
`GPURELAY_REPO` with its full name (`your-name/repo-name`).

**Each time:** *Runtime → Change runtime type →* choose a GPU, then *Runtime → Run all*.
Keep this tab open while the job runs. Near Colab's time limit the job saves a checkpoint and
GPU Relay continues it elsewhere."""

FIND = """# Find the job that GPU Relay asked to run on Colab.
import json
import subprocess
import sys

try:
    from google.colab import userdata
except ImportError:
    raise SystemExit("Open this notebook in Google Colab.") from None


def secret(name):
    try:
        return userdata.get(name) or ""
    except Exception:
        return ""


TOKEN = secret("GPURELAY_HF_TOKEN")
if not TOKEN.startswith("hf_"):
    raise SystemExit("Add the Colab secret GPURELAY_HF_TOKEN (key icon on the left) with your "
                     "Hugging Face token, turn on notebook access, then Run all again.")
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.24"])
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402 - installed just above

api = HfApi(token=TOKEN)
user = api.whoami()["name"]
candidates = [secret("GPURELAY_REPO")] if secret("GPURELAY_REPO") else []
candidates.append(f"{user}/gpurelay-store")
candidates += [d.id for d in api.list_datasets(author=user) if d.id not in candidates]

waiting = []  # (requested at, repo, job id)
for repo in candidates:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception:
        continue
    for name in files:
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "manual_session.json":
            if f"jobs/{parts[1]}/colab/resume.ipynb" not in files:
                continue
            path = hf_hub_download(repo, name, repo_type="dataset", token=TOKEN,
                                   force_download=True, local_dir="/tmp/gpurelay/find")
            with open(path, encoding="utf-8") as fh:
                waiting.append((json.load(fh).get("requested_at", 0), repo, parts[1]))
    if waiting:
        break
if not waiting:
    raise SystemExit("No job is waiting for Colab right now. GPU Relay asks (on its website, "
                     "and by email if you turned that on) when one of your jobs needs Colab.")
_, REPO, JOB = max(waiting)
print(f"Job {JOB} in {REPO} is waiting for Colab: starting it.")"""

RUN = """# Run the job's resume notebook here, cell by cell (it checks again that it may start).
notebook = hf_hub_download(REPO, f"jobs/{JOB}/colab/resume.ipynb", repo_type="dataset",
                           token=TOKEN, force_download=True, local_dir="/tmp/gpurelay/job")
with open(notebook, encoding="utf-8") as fh:
    cells = [c for c in json.load(fh)["cells"] if c["cell_type"] == "code"]
shell = get_ipython()  # Colab's IPython
for number, cell in enumerate(cells, start=1):
    result = shell.run_cell("".join(cell["source"]))
    if not result.success:
        print(f"GPU Relay stopped at step {number} of {len(cells)}: see the message above.")
        break
else:
    print("Done. The job's results are on your GPU Relay page.")"""


def build() -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO), nbformat.v4.new_code_cell(FIND),
                nbformat.v4.new_code_cell(RUN)]  # fmt: skip
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "GPU Relay: run your job", "provenance": []},
    })  # fmt: skip
    nbformat.validate(nb)
    out = Path(__file__).with_name("open_job.ipynb")
    nbformat.write(nb, str(out))
    return out


if __name__ == "__main__":
    print("wrote", build())
