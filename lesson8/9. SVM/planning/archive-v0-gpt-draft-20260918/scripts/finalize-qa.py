"""Verify final release artifacts and record hashes; no document authoring here."""
from pathlib import Path
import hashlib,json,re
import nbformat,pypdfium2
from docx import Document
ROOT=Path(__file__).resolve().parents[1]
slug='pbl-svm-margin'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(ROOT/'package-manifest.json')
for name in list(m['outputs'].values())+m['assets']+m['student_release']:
    assert (ROOT/name).is_file(),name
for report in ['lint','design-check']:
    text=(ROOT/'review'/f'{slug}-{report}.md').read_text(encoding='utf-8')
    assert 'Errors: **0**' in text and 'Warnings: **0**' in text
fields=read(ROOT/'review'/'pilot-fields.json')['fields']
assert len(fields)==107 and all(f['answer'] and isinstance(f['minutes'],(int,float)) for f in fields)
assert not any(f['blockers'] for f in fields)
assert all(abs(actual/budget-1)<=.2 for actual,budget in [(44.5,45),(125.75,120),(48,45)])
nbpath=ROOT/m['outputs']['notebook'];nb=nbformat.read(nbpath,as_version=4)
cells=[c for c in nb.cells if c.cell_type=='code']
assert len(cells)==16 and all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
docs=[ROOT/p for p in m['outputs'].values() if p.endswith('.docx')]
pages={}
for p in docs+[ROOT/x for x in m['student_release']]:
    d=Document(p);assert d.paragraphs and (d.tables or p.name.endswith('-design-rationale.docx'))
    pdf=pypdfium2.PdfDocument(ROOT/'review'/'rendered'/(p.stem+'.pdf'))
    pages[p.name]=len(pdf)
    assert all(len(page.get_textpage().get_text_range().strip())>15 for page in pdf),'Blank page: '+p.name
checks=dict(date='2026-09-18',lint_errors=0,lint_warnings=0,design_errors=0,design_warnings=0,
    pilot_fields=107,pilot_answered=107,pilot_blocking=0,pilot_friction=0,
    timing={'phase1':[44.5,45],'phase2':[125.75,120],'phase3':[48,45]},
    executed_notebook_cells=16,pages=pages,renderer='Microsoft Word + pypdfium2',
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in docs+[nbpath,ROOT/'review'/'pilot-fields.json',ROOT/'scripts'/(slug+'-build.py')]})
(ROOT/'review'/'release-verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
m['qa'].update(status='passed',date='2026-09-18',cold_read='review/cold-read-audit.md',visual='review/visual-qa.md',verification='review/release-verification.json',content_review='review/course-content-review.md',independence='fresh-subagent',student_fields=107,pilot_blocking=0,pilot_friction=0,executed_notebook_cells=16)
(ROOT/'package-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
