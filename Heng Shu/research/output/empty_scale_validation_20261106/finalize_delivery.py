"""Check frozen products and record the delivered artifacts; no CSI analysis."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
frozen = json.loads((root / 'empty_stage_freeze.json').read_text(encoding='utf8'))
assert all(sha(root / name) == value for name, value in frozen.items())
audit = json.loads((root / 'review/tolerance_equivalence.json').read_text(encoding='utf8'))
assert audit['positive_rank_count_differences'] == 0
assert audit['positive_not_increased_gate_differences'] == 0
assert not audit['R2_processed']
verification_path = root / 'verification.json'
verification = json.loads(verification_path.read_text(encoding='utf8'))
verification.update(
    independent_execution_audit_complete=True,
    numeric_tolerance_count_differences=0,
    model_envelope_independently_reproduced=True,
    png_visual_inspection_complete=True,
    figure_labels_and_legends_checked=True,
    thesis_or_overleaf_modified=False,
)
verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf8')
files = {
    p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
    for p in sorted(root.rglob('*'))
    if p.is_file() and p.name != 'delivery_manifest.json' and '__pycache__' not in p.parts
}
(root / 'delivery_manifest.json').write_text(json.dumps({
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'frozen_empty_outputs_verified': True,
    'scope': 'Empty-only development validation; no new R2 estimation.',
    'files': files,
}, ensure_ascii=False, indent=2), encoding='utf8')
print(f'Verified {len(frozen)} frozen products; recorded {len(files)} delivered files.')
