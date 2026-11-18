"""Verify frozen predictions before joining provisional geometry for evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluate(tables):
    tables = Path(tables)
    freeze_path = tables/'fit_provenance.json'
    if not freeze_path.is_file():
        raise ValueError('Missing fit_provenance.json: predictions must be frozen before geometry')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    hashes = freeze.get('frozen_before_geometry', {})
    if 'blind_predictions.csv' not in hashes:
        raise ValueError('Freeze does not cover blind_predictions.csv')
    for name,digest in hashes.items():
        if Path(name).name != name or '\\' in name:
            raise ValueError('Invalid freeze path')
        if hashlib.sha256((tables/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen prediction hash mismatch: '+name)
    config = json.loads((ROOT/'config/geometry.json').read_text(encoding='utf-8'))
    points = config['points_m']
    labels = config['archival_to_report_label']
    rx_x,rx_y = config['rx_m']
    with (tables/'blind_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        predictions = list(csv.DictReader(f))
    rows = []
    seen = set()
    for row in predictions:
        if row['run']!='run2' or row['mapping']!='cov':
            continue
        label = labels[row['point']]
        key = (label,row['branch'],row['method'])
        if key in seen:
            raise ValueError('Duplicate selected prediction')
        seen.add(key)
        x,y = points[label]
        gt = math.degrees(math.atan2(y-rx_y,rx_x-x))
        estimate = float(row['angle_deg'])
        if not math.isfinite(estimate):
            raise ValueError('Non-finite prediction')
        rows.append(dict(point=label,archival_point=row['point'],branch=row['branch'],
                         method=row['method'],reference_deg=gt,estimate_deg=estimate,
                         absolute_error_deg=abs(estimate-gt),provisional_convention=True))
    expected = {(p,b,m) for p in points for b in ('raw','background','kong_phase_hypothesis')
                for m in ('G3','J6_product')}
    if seen != expected:
        raise ValueError('Missing/unexpected four-point prediction combinations')
    summary = {}
    for branch in ('raw','background','kong_phase_hypothesis'):
        summary[branch] = {method:sum(row['absolute_error_deg'] for row in rows
                        if row['branch']==branch and row['method']==method)/len(points)
                        for method in ('G3','J6_product')}
    bg = summary['background']
    result = dict(conditional_mae_deg=summary,
                  background_conditional_mae_reduction_percent=100*(1-bg['J6_product']/bg['G3']),
                  points=4,static_person_runs_per_point=1,physically_verified_accuracy=False,
                  native_crosscheck_this_fit=freeze.get('native_crosscheck','historical recorded audit'),
                  limitations='Selected four-point subset; provisional AntSEL/sign; human-specific path not verified')
    return rows,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows,result=evaluate(args.tables)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'conditional_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
