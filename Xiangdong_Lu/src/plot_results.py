"""Render measured spectra after verifying frozen prediction hashes."""
import argparse
from pathlib import Path
from evaluate_results import evaluate
import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.tables)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    compare.plot(compare.load(args.tables/'spectra.csv'),
                 compare.load(args.tables/'negative_control.csv'),
                 compare.geometry(),args.out)
    print('Saved measured spectra and independent-empty control plots. Archival E = B_new.')


if __name__=='__main__':main()
