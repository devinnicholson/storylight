#!/usr/bin/env python3
"""Build an offline blind gallery from a verified eleven-image render batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import install_fidelity_display as installer  # noqa: E402

render = installer.render
HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Story image review</title>
<style>
body{font:16px system-ui;margin:2rem auto;max-width:1400px;padding:0 1rem;
background:#141922;color:#eef2f7}
h1,h2,h3{line-height:1.2}article{border-top:1px solid #536073;margin:3rem 0;padding-top:1rem}
.options{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:1.5rem}
img{width:100%;height:auto;background:#000}label{display:block;margin:.7rem 0}
select,button{font:inherit;padding:.4rem;margin-left:.5rem}button{cursor:pointer}
button:disabled{opacity:.55}li{margin:.4rem 0}.state{margin:1rem 0}details{margin:1rem 0}
</style>
<h1>Story image review</h1>
<p>Judge each option against the facts below. Pages with two states show ordered still images;
they do not demonstrate generated motion. Use the state buttons to inspect both images.</p>
<p>Ratings stay in this browser until you export them.
Leaving or reloading this page clears them.</p>
<main id="pages"></main><button id="export" disabled>Export review JSON</button>
<p id="status" role="status">Loading review…</p>
<script>
const root=document.querySelector('#pages'), reviews=[];
function el(tag,text,parent){const n=document.createElement(tag);
  if(text)n.textContent=text;if(parent)parent.append(n);return n;}
function choice(parent,label,values){const box=el('label',label,parent), s=el('select','',box);
  for(const [value,text] of [['','Unrated'],...values]){
    const o=el('option',text,s);o.value=value;}return s;}
function facts(parent,title,items){el('h3',title,parent);const list=el('ul','',parent);
  for(const t of items)el('li',t,list);}
fetch('review-data.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(data=>{
  for(const page of data.pages){
    const section=el('article','',root);el('h2',`Page ${page.page}`,section);
    facts(section,'Required facts',page.required);facts(section,'Mistakes to avoid',page.forbidden);
    if(page.options.length===1)el('p','One option is available for this page.',section);
    const controls=el('div','',section);controls.className='state';
    const grid=el('div','',section);grid.className='options';const images=[],options=[];
    for(const option of page.options){
      const card=el('section','',grid);el('h3',`Option ${option.label}`,card);
      const img=el('img','',card);images.push([img,option]);
      const correct=choice(card,'Correctness',[
        ['correct','Correct'],['incorrect','Incorrect'],['unsure','Unsure']]);
      const legible=choice(card,'Legibility',[['clear','Clear'],['unclear','Unclear']]);
      const continuity=choice(card,'Character and object continuity',[
        ['consistent','Consistent'],['inconsistent','Inconsistent'],['unsure','Unsure']]);
      const rating=choice(card,'Overall rating',[
        ['1','1 — poor'],['2','2'],['3','3'],['4','4'],['5','5 — excellent']]);
      const detail=el('details','',card);el('summary','Record individual fact checks',detail);
      const required=page.required.map(t=>choice(detail,t,[
        ['visible','Visible'],['missing','Missing or wrong'],['unsure','Unsure']]));
      const forbidden=page.forbidden.map(t=>choice(detail,t,[
        ['absent','Absent'],['present','Present'],['unsure','Unsure']]));
      options.push({label:option.label,correct,legible,continuity,rating,required,forbidden});
    }
    const buttons=[];
    function state(index){for(const [img,option] of images){img.src=option.frames[index];
      img.alt=`Page ${page.page}, option ${option.label}, state ${index+1}`;}
      buttons.forEach((button,i)=>{button.disabled=i===index;});}
    for(let i=0;i<page.states;i++){const button=el('button',`State ${i+1}`,controls);
      button.onclick=()=>state(i);buttons.push(button);}
    state(0);reviews.push({page:page.page,options});
  }
  document.querySelector('#status').textContent=
    'Ready. Unrated fields remain explicitly unrated in the export.';
  const button=document.querySelector('#export');button.disabled=false;button.onclick=()=>{
    const result={schema_version:1,review_id:data.review_id,pages:reviews.map(page=>({
      page:page.page,options:page.options.map(o=>({
      label:o.label,correct:o.correct.value,legible:o.legible.value,
      continuity:o.continuity.value,rating:o.rating.value,
      required:o.required.map(s=>s.value),forbidden:o.forbidden.map(s=>s.value)
    }))}))};
    const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),link=el('a');link.href=url;
    link.download=`review-${data.review_id}.json`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
}).catch(()=>{document.querySelector('#status').textContent=
  'Review data could not be loaded. Serve this folder with the local review server.';});
</script></html>
"""


def build(args) -> None:
    output = args.output.resolve()
    key = args.key_file.resolve()
    if args.output.exists() or args.output.is_symlink() or args.key_file.exists():
        raise ValueError("review outputs must be fresh")
    if key == output or output in key.parents:
        raise ValueError("mapping key must be outside the gallery")
    installer.assembly.smoke.private_path(args.key_file)
    batch = render.read_batch(args.batch, args.proof_batch_sha256)
    _, journal_sha256 = installer.completed_bundles(batch, args.proof_batch_sha256, args.render_dir)
    journal_data = installer.read_bounded(args.render_dir / "journal.jsonl", 262144)
    if hashlib.sha256(journal_data).hexdigest() != journal_sha256:
        raise ValueError("verified journal changed")
    journal = [json.loads(line) for line in journal_data.splitlines()]
    assets = {}
    for ordinal, row in enumerate(render.ordered_requests(batch)):
        content = installer.read_bounded(
            args.render_dir / f"image-{ordinal:02}/master.jpg", 16_000_000
        )
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != journal[ordinal * 2 + 2]["master_sha256"]:
            raise ValueError("verified image changed")
        assets[row.id] = (f"frame-{checksum}.jpg", content)
    manifest, _ = installer.assembly.smoke.load_manifest(installer.assembly.probe.STORY)
    public = {"schema_version": 1, "review_id": secrets.token_hex(16), "pages": []}
    mapping = {
        "review_id": public["review_id"],
        "batch_sha256": args.proof_batch_sha256,
        "journal_sha256": journal_sha256,
        "pages": [],
    }
    for page in batch.pages:
        index = page.source_page_index
        groups = [
            [
                row
                for row in batch.requests
                if row.source_page_index == index and row.variant == variant
            ]
            for variant in ("candidate", "accepted")
        ]
        groups = [group for group in groups if group]
        secrets.SystemRandom().shuffle(groups)
        states = len(page.display_step_ids)
        options, private_options = [], []
        for label, group in zip("AB"[: len(groups)], groups, strict=True):
            frames = [assets[row.id][0] for row in group]
            options.append(
                {"label": label, "frames": frames if len(frames) == states else frames * states}
            )
            private_options.append(
                {"label": label, "variant": group[0].variant, "requests": [row.id for row in group]}
            )
        source = manifest["pages"][index - 1]
        public["pages"].append(
            {
                "page": index,
                "states": states,
                "options": options,
                "required": source["required_facts"],
                "forbidden": source["forbidden_mistakes"],
            }
        )
        mapping["pages"].append({"page": index, "options": private_options})
    mapping["review_data_sha256"] = installer.assembly.digest(public)
    descriptor = os.open(args.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in assets.values():
        (args.output / name).write_bytes(content)
    (args.output / "review-data.json").write_text(json.dumps(public, indent=2) + "\n")
    (args.output / "index.html").write_text(HTML)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "render-dir", "output", "key-file"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--proof-batch-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        build(args)
        return 0
    except Exception:
        print("review gallery refused; inspect local verified inputs and fresh output paths")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
