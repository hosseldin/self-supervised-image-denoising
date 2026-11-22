"""Prepare the fixed SIDD subset used by the C17 experiments.

Scene membership and crop locations are inherited from M1 so later models
can be compared on the same images. Clean targets are saved for evaluation
splits only; noisy training crops remain available without paired targets.
"""
import csv
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# Keep every capture and crop from a physical scene in one split. Randomly
# mixing crops would let the same scene appear in both training and evaluation.
SCENES = {'train': ['001','003','005','008','009','010'],
          'val': ['002','004'], 'test': ['006','007']}
# This identifies the exact source archive, not just a similarly named download.
EXPECTED_SHA = 'f7dff04d6a0e135c0c64a12cf5a23d7a8efaf67d013f0ce49006b2e9fddbc3cf'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def prepare(archive, output):
    """Extract four fixed 256-pixel crops per capture and write their manifest."""
    archive, output = Path(archive), Path(output)
    if (output/'manifest.csv').exists(): raise FileExistsError('Manifest already exists')
    sha = digest(archive)
    if sha != EXPECTED_SHA: raise ValueError('SIDD Small archive SHA256 mismatch')
    rows = []
    with tarfile.open(archive, 'r:') as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        names = sorted(n for n in members if 'NOISY_SRGB' in n.upper() and n.lower().endswith('.png'))
        if len(names) != 160: raise ValueError('Expected 160 captures')
        for i, name in enumerate(names):
            capture = PurePosixPath(name).parent.name
            scene = capture.split('_')[1]
            split = next(k for k, v in SCENES.items() if scene in v)
            with tar.extractfile(members[name]) as f, Image.open(f) as im:
                im.load(); w, h = im.size
                # Quarter positions give a reproducible sample of each image.
                # Do not pick visually easy crops or change these for a new model.
                boxes = [(x-128,y-128,x+128,y+128)
                         for y in (h//4,3*h//4) for x in (w//4,3*w//4)]
                for k, box in enumerate(boxes):
                    ident = f'{capture}_{k}'
                    path = output/'noisy'/f'{ident}.png'; path.parent.mkdir(parents=True,exist_ok=True)
                    im.crop(box).convert('RGB').save(path)
                    rows.append(dict(id=ident,scene=scene,split=split,noisy=f'noisy/{ident}.png',
                                     clean='' if split=='train' else f'clean/{ident}.png',
                                     x=box[0],y=box[1]))
            # Ground truth is needed to score val/test. It is deliberately absent
            # from the training manifest and is not extracted for that split.
            if split != 'train':
                ref = name.replace('NOISY','GT')
                with tar.extractfile(members[ref]) as f, Image.open(f) as im:
                    for k, box in enumerate(boxes):
                        path=output/'clean'/f'{capture}_{k}.png';path.parent.mkdir(parents=True,exist_ok=True)
                        im.crop(box).convert('RGB').save(path)
            print(f'Prepared {i+1}/160',flush=True)
    counts = {s:sum(r['split']==s for r in rows) for s in SCENES}
    # Unequal crop counts are expected: scenes contain different capture counts.
    assert counts == {'train':364,'val':120,'test':156}, counts
    with (output/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write_json(output/'provenance.json',dict(archive_sha256=sha,scene_split=SCENES,counts=counts,
          crop_rule='256px centered at quarter positions, row-major; submitted M1',seed=681))

def records(root, split):
    """Read manifest rows without loading image pixels."""
    with (Path(root)/'manifest.csv').open(encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['split']==split]

# Use the same RGB scale for every method before comparing PSNR and SSIM.
def read_image(path):
    """Return RGB float32 pixels in [0, 1], matching the evaluation pipeline."""
    with Image.open(path) as im: return np.array(im.convert('RGB'),dtype=np.float32)/255

def load_noisy(root, split):
    """Stack a complete split in memory; this helper is not a streaming loader."""
    return np.stack([read_image(Path(root)/r['noisy']) for r in records(root,split)])
