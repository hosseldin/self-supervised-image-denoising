"""Check the saved M2 validation results without running the filters again."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def review(root=ROOT):
    folder = Path(root) / 'results/mennatalla_m2_2026_11_22'
    evidence = json.loads((folder / 'evidence.json').read_text(encoding='utf-8'))

    # Check that each result file still matches its recorded hash.
    for relative, expected in evidence['package_sha256'].items():
        actual = hashlib.sha256((folder / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Saved evidence changed: {relative}')

    with (Path(root) / 'results/verification/manifest.csv').open(encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))

    # Crops from the same scene share content even if they come from different
    # captures, so a scene can only be in one split. Otherwise it's leakage.
    scenes = {split: {r['scene'] for r in manifest if r['split'] == split}
              for split in ('train', 'val', 'test')}
    assert not (scenes['train'] & scenes['val'] or scenes['train'] & scenes['test']
                or scenes['val'] & scenes['test']), 'Scene overlap'

    # Fixed split sizes.
    assert {s: sum(r['split'] == s for r in manifest) for s in scenes} == {
        'train': 364, 'val': 120, 'test': 156}

    # No clean reference paths in the training rows.
    # Note: this only checks the manifest, not what a training script actually loads.
    assert all(not r['clean'] for r in manifest if r['split'] == 'train')

    val_ids = [r['id'] for r in manifest if r['split'] == 'val']
    summaries = []
    for method in ('noisy', 'bm3d', 'nlm'):
        with (folder / method / 'per_crop.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        saved = json.loads((folder / method / 'summary.json').read_text(encoding='utf-8'))

        # Check that all methods use the same validation crops.
        # Also require the rows to follow the manifest order.
        assert [r['id'] for r in rows] == val_ids, f'{method}: different crops or order'
        assert saved['split'] == 'val' and saved['count'] == len(rows) == 120
        assert saved['method'] == method

        # Recompute the means from the CSV values, before rounding for display.
        # Allow a small tolerance for floating-point differences.
        for metric in ('psnr_db', 'ssim'):
            actual = mean(float(r[metric]) for r in rows)
            assert abs(actual - saved[metric]) < 1e-10, (method, metric)

        # Time is reported as a median, so compare with the median here too.
        assert abs(median(float(r['seconds']) for r in rows) - saved['median_seconds']) < 1e-10
        summaries.append(saved)

    print('PASS: fixed 364/120/156 split, disjoint scenes, no clean training paths.')
    print('PASS: all three methods use the same 120 validation crops, and their averages match the saved summaries.')
    print('Method       PSNR(dB)    SSIM     median s/crop')
    for r in summaries:
        print(f"{r['method']:<12} {r['psnr_db']:8.3f}    {r['ssim']:.4f}    {r['median_seconds']:.3f}")
    print('This only checks the saved results; it does not rerun denoising.')
    return summaries


if __name__ == '__main__':
    review()
