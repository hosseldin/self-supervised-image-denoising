"""Run the fixed classical filters and save a separate, traceable M2 experiment."""
import csv
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import time

import bm3d
import numpy as np
from skimage.restoration import denoise_nl_means

from .baseline_data import fixed_manifest, manifest_rows
from .data import digest, read_image, write_json
from .metrics import score

METHODS = ('noisy', 'bm3d', 'nlm')


def denoise(image, method):
    # Preserve the calls, float32 conversion and clipping used in the saved run.
    if method == 'bm3d':
        result = np.asarray(bm3d.bm3d_rgb(image, sigma_psd=0.20), dtype=np.float32)
    elif method == 'nlm':
        result = np.asarray(denoise_nl_means(
            image, h=0.08, patch_size=7, patch_distance=11,
            fast_mode=True, channel_axis=2), dtype=np.float32)
    elif method == 'noisy':
        result = image
    else:
        raise ValueError(f'Unknown baseline: {method}')
    return np.clip(result, 0, 1)


def save_csv(path, rows, fields):
    # Replace the snapshot only after it is fully written.
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    # Windows can briefly lock the destination while a viewer reads it.
    # Keep the old snapshot intact and retry the atomic replacement.
    for attempt in range(12):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 11:
                raise PermissionError(f'Close applications reading {path}; the pending snapshot is {temporary}')
            time.sleep(0.25)


def run_validation(data, output, repo, threads=2, limit=None):
    data, output, repo = Path(data), Path(output), Path(repo)
    all_rows = manifest_rows(data / 'manifest.csv')
    if all_rows != manifest_rows(fixed_manifest(repo)):
        raise ValueError('Input manifest differs from the fixed 364/120/156 scene split')
    validation = [row for row in all_rows if row['split'] == 'val']
    if len(validation) != 120:
        raise ValueError('Expected 120 validation crops')
    if threads < 1 or (limit is not None and not 1 <= limit <= 120):
        raise ValueError('Invalid thread count or crop limit')
    rows = validation if limit is None else validation[:limit]
    input_hashes = {}
    for row in rows:
        for key in ('noisy', 'clean'):
            input_hashes[row[key]] = digest(data / row[key])
    # Never overwrite a published table or an earlier run.
    output.mkdir(parents=True, exist_ok=False)
    full_run = len(rows) == 120
    config = {
        'status': 'running', 'scope': 'full_validation' if full_run else 'smoke_test',
        'count': len(rows), 'evaluation_only': True, 'training_performed': False,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'scikit-image', 'pillow', 'matplotlib', 'bm3d', 'bm4d')},
        'threads_requested': threads,
        'thread_policy': 'OMP/MKL/OPENBLAS limits set by CLI before imports; BM3D default np profile retained',
        'bm3d_sigma_psd': 0.20,
        'nlm': {'h': 0.08, 'patch_size': 7, 'patch_distance': 11, 'fast_mode': True},
        'seed_of_original_split': 681,
        'manifest_sha256': digest(data / 'manifest.csv'),
        'crop_ids': [row['id'] for row in rows],
        'source_sha256': {str(path.relative_to(repo)).replace('\\', '/'): digest(path)
                          for path in [repo / 'scripts/run_m2_baselines.py',
                                       *(repo / 'src/c17' / name for name in
                                         ('baselines.py', 'baseline_data.py', 'data.py', 'metrics.py'))]},
        'scoring': 'RGB float64; border 4; Gaussian SSIM window 11 sigma 1.5; population covariance',
        'timing': 'Filter call, float32 conversion and clipping only; excludes loading, metrics and output I/O',
        'note': 'New measurements; do not replace the archived Colab results with this run.'
    }
    write_json(output / 'environment_and_protocol.json', config)
    write_json(output / 'input_files_sha256.json', input_hashes)
    summaries, examples = [], {}
    for method in METHODS:
        folder = output / method
        folder.mkdir()
        values = []
        for index, row in enumerate(rows):
            noisy, clean = (read_image(data / row[key]) for key in ('noisy', 'clean'))
            if noisy.shape != (256, 256, 3) or clean.shape != noisy.shape:
                raise ValueError(f'Expected aligned 256x256 RGB crops: {row["id"]}')
            start = time.perf_counter()
            prediction = denoise(noisy, method)
            elapsed = time.perf_counter() - start
            psnr, ssim = score(prediction, clean)
            if not np.isfinite([psnr, ssim, elapsed]).all():
                raise ValueError(f'Non-finite measurement for {method}: {row["id"]}')
            values.append(dict(id=row['id'], psnr_db=psnr, ssim=ssim, seconds=elapsed))
            save_csv(folder / 'per_crop.csv', values, ('id', 'psnr_db', 'ssim', 'seconds'))
            if index == 0:
                examples[method] = prediction.copy()
                examples['clean'] = clean
            if index % 20 == 0 or index + 1 == len(rows):
                print(f'{method}: {index + 1}/{len(rows)} validation crops', flush=True)
        summary = dict(method=method, split='val', count=len(rows), scope=config['scope'],
                       manifest_sha256=config['manifest_sha256'],
                       psnr_db=float(np.mean([v['psnr_db'] for v in values])),
                       ssim=float(np.mean([v['ssim'] for v in values])),
                       psnr_std=float(np.std([v['psnr_db'] for v in values])),
                       mean_seconds=float(np.mean([v['seconds'] for v in values])),
                       median_seconds=float(np.median([v['seconds'] for v in values])))
        write_json(folder / 'summary.json', summary)
        summaries.append(summary)
    save_csv(output / 'comparison.csv', summaries,
             ('method', 'count', 'psnr_db', 'ssim', 'median_seconds'))
    plot_results(output, summaries, examples, rows[0]['id'], config['scope'])
    config.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'environment_and_protocol.json', config)
    print(f'COMPLETE: {config["scope"]}; results saved to {output}', flush=True)
    return summaries


def plot_results(output, summaries, images, first_id, scope):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    for axis, name in zip(axes, (*METHODS, 'clean')):
        title = 'Clean reference' if name == 'clean' else f'{name.upper()}\n{score(images[name], images["clean"])[0]:.3f} dB'
        axis.imshow(images[name])
        axis.set_title(title)
        axis.axis('off')
        Image.fromarray((images[name] * 255).round().astype('uint8')).save(output / f'{name}.png')
    fig.suptitle(f'First validation crop: {first_id}\nIndividual-crop scores, not table averages')
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(output / 'before_after.png', dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(['Noisy', 'BM3D', 'NLM'], [s['psnr_db'] for s in summaries],
                    color=['#94a3b8', '#0072b2', '#d55e00'])
    axis.bar_label(bars, fmt='%.3f', padding=4)
    axis.set(ylabel='Mean validation PSNR (dB)', ylim=(0, max(s['psnr_db'] for s in summaries) + 5),
             title=f'{scope.replace("_", " ")}: {summaries[0]["count"]} crops')
    fig.tight_layout()
    fig.savefig(output / 'validation_comparison.png', dpi=150)
    plt.close(fig)
