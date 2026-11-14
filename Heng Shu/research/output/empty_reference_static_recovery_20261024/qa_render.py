"""Read-only PDF rendering; all outputs are PNG QA derivatives."""
from pathlib import Path
import subprocess,shutil,json
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parent
qa=R/'qa';qa.mkdir(exist_ok=True)
exe=shutil.which('pdftoppm');assert exe
rows=[]
for p in sorted((R/'figures').rglob('*.pdf')):
    doc=PdfReader(p);dest=qa/p.stem;dest.mkdir(exist_ok=True)
    subprocess.run([exe,'-r','75','-png',str(p),str(dest/'page')],check=True,capture_output=True)
    pages=sorted(dest.glob('page-*.png'),key=lambda q:int(q.stem.split('-')[-1]))
    assert len(pages)==len(doc.pages)
    for n,page in enumerate(doc.pages):
        txt=page.extract_text() or '';assert len(txt)>50
        if 'human_atlas' in str(p):assert 'Uncalibrated per-state' in txt
    if len(pages)>1:
        w,h=350,480;cols=3;nr=(len(pages)+cols-1)//cols
        canvas=Image.new('RGB',(w*cols,h*nr),'#dddddd');draw=ImageDraw.Draw(canvas)
        for n,f in enumerate(pages):
            im=Image.open(f).convert('RGB');im.thumbnail((w-8,h-25));x=(n%cols)*w;y=(n//cols)*h;canvas.paste(im,(x+(w-im.width)//2,y+20));draw.text((x+8,y+3),f'{p.stem} page {n+1}',fill='black')
        canvas.save(qa/(p.stem+'_contact.png'))
    rows.append(dict(path=str(p),pages=len(doc.pages),all_pages_rendered=True,text_checked=True,sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest()))
(qa/'render_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
assert sum(x['pages'] for x in rows if 'human_atlas' in x['path'])==39
print('Rendered',sum(x['pages'] for x in rows),'pages from',len(rows),'PDFs')
