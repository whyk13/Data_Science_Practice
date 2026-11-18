"""Windows fallback after packaged renderer reports missing bundled LibreOffice.
Uses an isolated hidden Word instance, opens only supplied DOCX read-only, exports
to review/, and closes only its own instance. Does not modify source documents.
"""
from pathlib import Path
import sys
import json
import re
import fitz
import pythoncom
import win32com.client

root=Path(__file__).resolve().parents[1]
names=sys.argv[1:] or ['pbl-data-types-student-worksheet','pbl-data-types-reference']
pythoncom.CoInitialize()
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False
word.DisplayAlerts=0
word.AutomationSecurity=3
try:
    for name in names:
        src=root/(name+'.docx')
        out=root/'review'/('worksheet-render' if name.endswith('worksheet') else 'reference-render')
        out.mkdir(exist_ok=True,parents=True)
        pdf=out/(name+'.pdf')
        doc=word.Documents.Open(str(src),ReadOnly=True,AddToRecentFiles=False,Visible=False)
        try:
            doc.Repaginate()
            doc.ExportAsFixedFormat(str(pdf),17,OpenAfterExport=False)
        finally:
            doc.Close(SaveChanges=0)
        meta=[]
        with fitz.open(pdf) as f:
            for i,page in enumerate(f):
                page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'page-{i+1}.png')
                text=page.get_text()
                meta.append({'page':i+1,'characters':len(text),'text':text})
        (out/'pages.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        for stale in out.glob('page-*.png'):
            match=re.fullmatch(r'page-(\d+)\.png',stale.name)
            if match and int(match.group(1))>len(meta):
                assert stale.resolve().parent==out.resolve()
                stale.unlink()
        print(name,len(meta),'pages',flush=True)
        for x in meta:print(' ',x['page'],x['characters'],x['text'][:55].replace('\n',' / '))
finally:
    word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()
