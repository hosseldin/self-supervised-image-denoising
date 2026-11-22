"""Check saved M2 baseline evidence without running filters or writing results."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def review(root=ROOT):
    folder = Path(root) / 'results/m2_baselines'
    evidence = json.loads((folder / 'evidence.json').read_text(encoding='utf-8'))
    # Check the stored evidence before using it to explain the comparison.
    for relative, expected in evidence['package_sha256'].items():
        actual = hashlib.sha256((folder / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Saved evidence changed: {relative}')
    with (Path(root) / 'results/verification/manifest.csv').open(encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    scenes = {split: {r['scene'] for r in manifest if r['split'] == split}
              for split in ('train', 'val', 'test')}
    assert not (scenes['train'] & scenes['val'] or scenes['train'] & scenes['test']
                or scenes['val'] & scenes['test']), 'Scene overlap'
    assert {s: sum(r['split'] == s for r in manifest) for s in scenes} == {
        'train': 364, 'val': 120, 'test': 156}
    assert all(not r['clean'] for r in manifest if r['split'] == 'train')
    # Matching counts alone is not enough: every method must use the same crops.
    val_ids = [r['id'] for r in manifest if r['split'] == 'val']
    summaries = []
    for method in ('noisy', 'bm3d', 'nlm'):
        with (folder / method / 'per_crop.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        saved = json.loads((folder / method / 'summary.json').read_text(encoding='utf-8'))
        assert [r['id'] for r in rows] == val_ids, f'{method}: different crops or order'
        assert saved['split'] == 'val' and saved['count'] == len(rows) == 120
        assert saved['method'] == method
        # Recalculate the averages from all rows, not from the rounded table labels.
        for metric in ('psnr_db', 'ssim'):
            actual = mean(float(r[metric]) for r in rows)
            assert abs(actual - saved[metric]) < 1e-10, (method, metric)
        assert abs(median(float(r['seconds']) for r in rows) - saved['median_seconds']) < 1e-10
        summaries.append(saved)
    print('PASS: fixed 364/120/156 split, disjoint scenes, no clean training paths.')
    print('PASS: all three saved baseline averages match the same 120 validation crops.')
    print('Method       PSNR(dB)    SSIM     median s/crop')
    for r in summaries:
        print(f"{r['method']:<12} {r['psnr_db']:8.3f}    {r['ssim']:.4f}    {r['median_seconds']:.3f}")
    print('This checks recorded evidence; it does not rerun denoising or certify student review.')
    return summaries


if __name__ == '__main__':
    review()
