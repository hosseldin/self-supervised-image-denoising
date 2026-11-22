"""Small correctness and input-safety checks; no full SIDD download required."""
from pathlib import Path
from contextlib import redirect_stdout
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from c17.baselines import denoise, run_validation, save_csv
from c17.baseline_data import manifest_rows, prepare_directory, prepare_validation_zip
from c17.metrics import score


class BaselineChecks(unittest.TestCase):
    def test_snapshot_recovers_from_a_temporary_windows_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'per_crop.csv'
            target.write_text('previous snapshot')
            real_replace = Path.replace
            attempts = []
            def locked_once(source, destination):
                attempts.append(1)
                if len(attempts) == 1:
                    raise PermissionError('Simulated reader lock')
                return real_replace(source, destination)
            with patch.object(Path, 'replace', locked_once):
                save_csv(target, [{'id': 'crop-1', 'psnr_db': 30.0}], ('id', 'psnr_db'))
            with target.open(newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{'id': 'crop-1', 'psnr_db': '30.0'}])
            self.assertFalse(target.with_suffix('.tmp').exists())

    def test_psnr_known_constant_error(self):
        reference = np.full((32, 32, 3), 0.5)
        prediction = np.full_like(reference, 0.4)
        psnr, _ = score(prediction, reference)
        self.assertAlmostEqual(psnr, 20.0, places=10)

    def test_discarded_border_does_not_affect_metrics(self):
        reference = np.full((32, 32, 3), 0.5)
        prediction = reference.copy()
        prediction[:4] = 0
        prediction[-4:] = 1
        prediction[:, :4] = 0
        prediction[:, -4:] = 1
        psnr, ssim = score(prediction, reference)
        self.assertTrue(np.isinf(psnr))
        self.assertAlmostEqual(ssim, 1.0)

    def test_unknown_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            denoise(np.zeros((32, 32, 3)), 'cnn')

    def test_different_split_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            data = path / 'data'
            data.mkdir()
            (data / 'manifest.csv').write_text('id,split\nwrong,val\n')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                run_validation(data, path / 'output', ROOT, limit=1)
            self.assertFalse((path / 'output').exists())

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            data, output = path / 'data', path / 'output'
            data.mkdir()
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('keep this run')
            manifest = ROOT / 'results/verification/manifest.csv'
            (data / 'manifest.csv').write_bytes(manifest.read_bytes())
            row = next(r for r in manifest_rows(manifest) if r['split'] == 'val')
            for key in ('noisy', 'clean'):
                image_path = data / row[key]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (256, 256)).save(image_path)
            with self.assertRaises(FileExistsError):
                run_validation(data, output, ROOT, limit=1)
            self.assertEqual(marker.read_text(), 'keep this run')
            self.assertEqual(list(output.iterdir()), [marker])

    def test_wrong_validation_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            archive = path / 'input.zip'
            archive.write_bytes(b'not the recorded data')
            with self.assertRaisesRegex(ValueError, 'recorded'):
                prepare_validation_zip(archive, path / 'output', ROOT)
            self.assertFalse((path / 'output').exists())

    def test_incomplete_sidd_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, '160'):
                prepare_directory(path, path / 'output', ROOT)
            self.assertFalse((path / 'output').exists())

    def test_directory_preparation_preserves_split_and_crop_order(self):
        # Use small synthetic captures to test layout, alignment and training
        # target handling without downloading the full-resolution SIDD archive.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            repo, source, output = path / 'repo', path / 'source', path / 'output'
            manifest = repo / 'results/verification/manifest.csv'
            manifest.parent.mkdir(parents=True)
            rows = manifest_rows(ROOT / 'results/verification/manifest.csv')
            colors = [(20, 40, 60), (80, 100, 120), (140, 160, 180), (200, 220, 240)]
            image = Image.new('RGB', (512, 512))
            for k, color in enumerate(colors):
                x, y = (k % 2) * 256, (k // 2) * 256
                image.paste(color, (x, y, x + 256, y + 256))
            for row in rows:
                capture, index = row['id'].rsplit('_', 1)
                row['x'], row['y'] = str((int(index) % 2) * 256), str((int(index) // 2) * 256)
                folder = source / capture
                if not folder.exists():
                    folder.mkdir(parents=True)
                    image.save(folder / 'NOISY_SRGB_010.PNG')
                    image.save(folder / 'GT_SRGB_010.PNG')
            with manifest.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with redirect_stdout(io.StringIO()):
                prepare_directory(source, output, repo)
            self.assertEqual((output / 'manifest.csv').read_bytes(), manifest.read_bytes())
            self.assertEqual(len(list((output / 'noisy').glob('*.png'))), 640)
            self.assertEqual(len(list((output / 'clean').glob('*.png'))), 276)
            for row in rows:
                if row['split'] == 'train':
                    self.assertEqual(row['clean'], '')
                    self.assertFalse((output / 'clean' / (row['id'] + '.png')).exists())
            for row in [r for r in rows if r['split'] == 'val'][:4]:
                color = colors[int(row['id'].rsplit('_', 1)[1])]
                for key in ('noisy', 'clean'):
                    with Image.open(output / row[key]) as crop:
                        self.assertEqual(crop.size, (256, 256))
                        self.assertEqual(crop.getpixel((0, 0)), color)
                        self.assertEqual(crop.getpixel((255, 255)), color)

    def test_zero_crop_limit_is_rejected_by_cli(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/run_m2_baselines.py'), 'run',
             '--data', 'unused', '--output', 'unused', '--limit', '0'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--limit must be between 1 and 120', result.stderr)


if __name__ == '__main__':
    unittest.main()
