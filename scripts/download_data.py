from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
root=Path(__file__).resolve().parents[1]; target=root/"data"/"winequality-red.csv"
def main():
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): print("Dataset already exists:",target);return
    url="https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    print("Downloading official UCI Wine Quality archive...")
    with urlopen(url,timeout=30) as r: z=ZipFile(BytesIO(r.read()))
    member=next((n for n in z.namelist() if n.lower().endswith("winequality-red.csv")),None)
    if not member: raise RuntimeError("winequality-red.csv not found in archive")
    target.write_bytes(z.read(member));print("Saved",target)
if __name__=="__main__": main()
