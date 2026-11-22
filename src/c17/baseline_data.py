"""Prepare the recorded M2 validation split without needing training code."""
import csv
import json
from pathlib import Path
import zipfile

from PIL import Image
from .data import digest, write_json

VALIDATION_ZIP_SHA256 = 'f60687266ac3657bb7b030267816a762396697d70ec575b02ff10bbf29acbafe'


def manifest_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def fixed_manifest(repo):
    return Path(repo) / 'results/verification/manifest.csv'


def prepare_validation_zip(archive_path, output, repo):
    """Import only the images and manifest from the supplied validation ZIP."""
    archive_path, output = Path(archive_path), Path(output)
    if digest(archive_path) != VALIDATION_ZIP_SHA256:
        raise ValueError('This is not the recorded Mennatalla_M2_Run_Data.zip')
    manifest = fixed_manifest(repo)
    rows = manifest_rows(manifest)
    names = ['manifest.csv', 'provenance.json']
    names += [row[key] for row in rows if row['split'] == 'val'
              for key in ('noisy', 'clean')]
    with zipfile.ZipFile(archive_path) as archive:
        prefix = 'M2_Run/data/m1/'
        if archive.read(prefix + 'manifest.csv') != manifest.read_bytes():
            raise ValueError('Archive manifest differs from the recorded split')
        # Extract an explicit allow-list; ignore all code stored in the old ZIP.
        for name in names:
            archive.getinfo(prefix + name)
        output.mkdir(parents=True, exist_ok=False)
        for name in names:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(prefix + name))
    print(f'Prepared 120 validation pairs in {output}', flush=True)


def prepare_directory(source, output, repo):
    """Recreate the fixed crops from an extracted SIDD-Small sRGB directory."""
    source, output = Path(source), Path(output)
    manifest = fixed_manifest(repo)
    rows = manifest_rows(manifest)
    captures = {}
    for row in rows:
        captures.setdefault(row['id'].rsplit('_', 1)[0], []).append(row)
    files = {}
    for path in source.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.png' and 'NOISY_SRGB' in path.name.upper():
            if path.parent.name in files:
                raise ValueError(f'Multiple noisy images for {path.parent.name}')
            files[path.parent.name] = path
    if set(files) != set(captures):
        raise ValueError('Expected the exact 160 SIDD-Small capture folders; choose the Data directory')
    pairs = {}
    for capture, noisy in files.items():
        clean = [p for p in noisy.parent.iterdir()
                 if p.suffix.lower() == '.png' and 'GT_SRGB' in p.name.upper()]
        if len(clean) != 1:
            raise ValueError(f'Expected one ground-truth image for {capture}')
        pairs[capture] = (noisy, clean[0])
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for index, (capture, crop_rows) in enumerate(captures.items(), 1):
        for key, path in zip(('noisy', 'clean'), pairs[capture]):
            if key == 'clean' and crop_rows[0]['split'] == 'train':
                continue
            source_hashes[f'{capture}/{path.name}'] = digest(path)
            with Image.open(path) as image:
                w, h = image.size
                expected_xy = [(x - 128, y - 128) for y in (h // 4, 3 * h // 4)
                               for x in (w // 4, 3 * w // 4)]
                if [(int(r['x']), int(r['y'])) for r in crop_rows] != expected_xy:
                    raise ValueError(f'Image dimensions do not match the fixed crop coordinates: {path}')
                for row in crop_rows:
                    x, y = int(row['x']), int(row['y'])
                    target = output / row[key]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((x, y, x + 256, y + 256)).convert('RGB').save(target)
        print(f'Prepared {index}/160 captures', flush=True)
    # Keep the recorded order and exact manifest bytes, including empty training targets.
    (output / 'manifest.csv').write_bytes(manifest.read_bytes())
    write_json(output / 'provenance.json', {
        'source': 'User-supplied extracted SIDD-Small sRGB directory',
        'mirror': 'https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset',
        'manifest_sha256': digest(manifest), 'source_file_sha256': source_hashes,
        'crop_rule': '256px centered at quarter positions, row-major; submitted M1',
        'counts': {'train': 364, 'val': 120, 'test': 156}, 'seed': 681,
        'note': 'Hashes identify this local input copy; no official checksum verification is claimed.'
    })
    print(f'Prepared the fixed 364/120/156 split in {output}', flush=True)
