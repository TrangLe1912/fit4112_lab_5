"""Execute all required Lab 5 notebooks. No Papermill."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=["01_prepare_labels.ipynb","02_missing_labels_and_baseline.ipynb","03_self_training_evaluation.ipynb"]
if __name__=="__main__":
    import nbformat
    from nbclient import NotebookClient
    for name in NOTEBOOKS:
        path=ROOT/"notebooks"/name
        print("Running",name,flush=True)
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=240,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
        print("PASS",name,flush=True)
    print("PASS: mandatory Lab 5 notebooks")
