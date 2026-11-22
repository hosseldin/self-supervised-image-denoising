"""Prepare SIDD crops or run the M2 classical baselines on a CPU."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Create the fixed crops in a new folder')
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument('--sidd-dir', type=Path, help='Extracted SIDD-Small Data folder')
    source.add_argument('--validation-zip', type=Path, help='Mennatalla_M2_Run_Data.zip')
    prepare.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='Evaluate noisy input, BM3D and NLM')
    run.add_argument('--data', type=Path, required=True, help='Folder containing manifest.csv')
    run.add_argument('--output', type=Path, required=True, help='A new result folder')
    run.add_argument('--threads', type=int, default=2)
    run.add_argument('--limit', type=int, help='First N crops for a smoke test; omit for all 120')
    args = parser.parse_args()
    if args.command == 'prepare':
        from c17.baseline_data import prepare_directory, prepare_validation_zip
        if args.sidd_dir:
            prepare_directory(args.sidd_dir, args.output, ROOT)
        else:
            prepare_validation_zip(args.validation_zip, args.output, ROOT)
    else:
        if args.threads < 1:
            parser.error('--threads must be positive')
        if args.limit is not None and not 1 <= args.limit <= 120:
            parser.error('--limit must be between 1 and 120')
        # Set these before importing NumPy or the native filter libraries.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[name] = str(args.threads)
        from c17.baselines import run_validation
        run_validation(args.data, args.output, ROOT, args.threads, args.limit)


if __name__ == '__main__':
    main()
