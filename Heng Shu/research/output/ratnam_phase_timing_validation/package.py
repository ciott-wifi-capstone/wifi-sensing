"""Package local evidence without raw CSI; verify every archived member by SHA256."""
import csv,hashlib,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert json.loads((R/'results/completion_audit.json').read_text(encoding='utf-8'))['all_pass']
    for p in (R/'figures').glob('*.svg'):ET.parse(p)
    required=['REPORT.md','README.md','METHOD.md','VALIDATION.md']
    assert all((R/f).is_file() for f in required)
    # Verify all local Markdown links, excluding TeX expressions and web URLs.
    import re
    for name in required:
        for link in re.findall(r'\]\(([^)]+)\)',(R/name).read_text(encoding='utf-8')):
            if not link.startswith(('http:','https:','#')):assert (R/link).is_file(),(name,link)
    files=sorted(p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.csv')
    rows=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    with (R/'artifact_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
    dest=R.parent/'ratnam_phase_timing_validation_review.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[R/'artifact_manifest.csv']:z.write(p,p.relative_to(R).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for row in rows:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    checksum=sha(dest);dest.with_suffix('.zip.sha256').write_text(checksum+'\n',encoding='ascii')
    print(json.dumps(dict(zip=str(dest),bytes=dest.stat().st_size,members=len(files)+1,sha256=checksum,all_members_verified=True,local_document_links_checked=True,svg_xml_checked=True),indent=2))
if __name__=='__main__':main()
