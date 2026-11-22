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


# Split by scene, not by capture. Captures of the same scene look alike,
# so mixing them would leak evaluation content into training.
SCENES = {
    'train': ['001', '003', '005', '008', '009', '010'],
    'val': ['002', '004'],
    'test': ['006', '007'],
}

# The filename alone proves nothing, so check the exact archive contents.
EXPECTED_SHA = 'f7dff04d6a0e135c0c64a12cf5a23d7a8efaf67d013f0ce49006b2e9fddbc3cf'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        # 8 MB at a time, so the whole archive never sits in memory.
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # allow_nan=False: fail instead of writing NaN/Infinity, which is not valid JSON.
    path.write_text(
        json.dumps(obj, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def prepare(archive, output):
    """Extract four fixed 256-pixel crops per capture and write their manifest."""
    archive, output = Path(archive), Path(output)

    # Stop if a manifest already exists in the output folder.
    if (output / 'manifest.csv').exists():
        raise FileExistsError('Manifest already exists')

    sha = digest(archive)
    if sha != EXPECTED_SHA:
        raise ValueError('SIDD Small archive SHA256 mismatch')

    rows = []
    with tarfile.open(archive, 'r:') as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}

        # Sorted, so the capture order is the same on every run.
        names = sorted(
            n for n in members
            if 'NOISY_SRGB' in n.upper() and n.lower().endswith('.png')
        )
        if len(names) != 160:
            raise ValueError('Expected 160 captures')

        for i, name in enumerate(names):
            capture = PurePosixPath(name).parent.name

            # The second part of the SIDD folder name is the scene number.
            scene = capture.split('_')[1]
            split = next(k for k, v in SCENES.items() if scene in v)

            with tar.extractfile(members[name]) as f, Image.open(f) as im:
                im.load()
                w, h = im.size

                # Same four crops as M1: 256x256, centred on the quarter positions.
                # Fixed locations mean nobody picks the easy-looking regions.
                boxes = [
                    (x - 128, y - 128, x + 128, y + 128)
                    for y in (h // 4, 3 * h // 4)
                    for x in (w // 4, 3 * w // 4)
                ]

                for k, box in enumerate(boxes):
                    ident = f'{capture}_{k}'
                    path = output / 'noisy' / f'{ident}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    im.crop(box).convert('RGB').save(path)

                    rows.append(dict(
                        id=ident,
                        scene=scene,
                        split=split,
                        noisy=f'noisy/{ident}.png',
                        clean='' if split == 'train' else f'clean/{ident}.png',
                        x=box[0],
                        y=box[1],
                    ))

            # Clean references are only needed for val/test metrics.
            # For training they are never extracted and the path stays empty.
            if split != 'train':
                ref = name.replace('NOISY', 'GT')
                with tar.extractfile(members[ref]) as f, Image.open(f) as im:
                    for k, box in enumerate(boxes):
                        # Same boxes as the noisy image, so each pair stays aligned.
                        path = output / 'clean' / f'{capture}_{k}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        im.crop(box).convert('RGB').save(path)

            print(f'Prepared {i + 1}/160', flush=True)

    counts = {
        s: sum(r['split'] == s for r in rows)
        for s in SCENES
    }

    # 4 crops per capture. The sizes are uneven because the scenes
    # do not all have the same number of captures.
    assert counts == {'train': 364, 'val': 120, 'test': 156}, counts

    with (output / 'manifest.csv').open(
        'w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    # Keep the source hash and the crop rule next to the images for later checks.
    write_json(
        output / 'provenance.json',
        dict(
            archive_sha256=sha,
            scene_split=SCENES,
            counts=counts,
            crop_rule='256px centered at quarter positions, row-major; submitted M1',
            seed=681,
        ),
    )


def records(root, split):
    """Read manifest rows without loading image pixels."""
    with (Path(root) / 'manifest.csv').open(encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['split'] == split]


def read_image(path):
    """Return RGB float32 pixels in [0, 1], matching the evaluation pipeline."""
    with Image.open(path) as im:
        return np.array(im.convert('RGB'), dtype=np.float32) / 255


def load_noisy(root, split):
    """Stack a complete split in memory; this helper is not a streaming loader."""
    # All selected crops are loaded into RAM: about 290 MB for the 364 train crops
    # as float32, and roughly double that while np.stack builds the combined array.
    return np.stack([
        read_image(Path(root) / r['noisy'])
        for r in records(root, split)
    ])
