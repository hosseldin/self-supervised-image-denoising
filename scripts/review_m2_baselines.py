"""Check the earlier saved M2 baseline results without running the filters again."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def review(root=ROOT):
    # Keep this earlier run separate from the later Colab results.
    folder = Path(root) / 'results/m2_baselines'
    evidence = json.loads((folder / 'evidence.json').read_text(encoding='utf-8'))

    # Check that each result file still matches its recorded hash.
    for relative, expected in evidence['package_sha256'].items():
        actual = hashlib.sha256((folder / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Saved evidence changed: {relative}')

    with (Path(root) / 'results/verification/manifest.csv').open(encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))

    # Different captures can still show the same scene.
    # Check that scene content does not cross between the splits.
    scenes = {
        split: {r['scene'] for r in manifest if r['split'] == split}
        for split in ('train', 'val', 'test')
    }
    assert not (
        scenes['train'] & scenes['val']
        or scenes['train'] & scenes['test']
        or scenes['val'] & scenes['test']
    ), 'Scene overlap'

    # Expected crop counts from the fixed scene split.
    assert {s: sum(r['split'] == s for r in manifest) for s in scenes} == {
        'train': 364,
        'val': 120,
        'test': 156,
    }

    # Training rows should not point to clean reference images.
    # This checks the manifest, not what a separate training script loads.
    assert all(not r['clean'] for r in manifest if r['split'] == 'train')

    val_ids = [r['id'] for r in manifest if r['split'] == 'val']
    summaries = []

    for method in ('noisy', 'bm3d', 'nlm'):
        with (folder / method / 'per_crop.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        saved = json.loads(
            (folder / method / 'summary.json').read_text(encoding='utf-8')
        )

        # Check the crop IDs, not just the number of rows.
        # The rows must also follow the manifest order.
        assert [r['id'] for r in rows] == val_ids, (
            f'{method}: different crops or order'
        )
        assert saved['split'] == 'val' and saved['count'] == len(rows) == 120
        assert saved['method'] == method

        # Recompute the means from the CSV values, not from the rounded table.
        # Allow a small tolerance for floating-point differences.
        for metric in ('psnr_db', 'ssim'):
            actual = mean(float(r[metric]) for r in rows)
            assert abs(actual - saved[metric]) < 1e-10, (method, metric)

        # Use the median here because that is what the timing summary reports.
        assert abs(
            median(float(r['seconds']) for r in rows) - saved['median_seconds']
        ) < 1e-10

        summaries.append(saved)

    print('PASS: fixed 364/120/156 split, disjoint scenes, no clean training paths.')
    print('PASS: all three baselines use the same 120 validation crops, and their averages match the saved summaries.')
    print('Method       PSNR(dB)    SSIM     median s/crop')

    for r in summaries:
        print(
            f"{r['method']:<12} {r['psnr_db']:8.3f}    "
            f"{r['ssim']:.4f}    {r['median_seconds']:.3f}"
        )

    print('This only checks the saved results; it does not rerun denoising.')
    return summaries


if __name__ == '__main__':
    review()
