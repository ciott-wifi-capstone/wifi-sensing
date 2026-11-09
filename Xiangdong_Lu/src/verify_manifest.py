"""Verify all distributed file checksums; local ignored outputs may be present."""
import hashlib
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    checked=0
    for line in (root/'MANIFEST_SHA256.txt').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path leaves the project')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Checksum mismatch: '+name)
        checked+=1
    print(f'PASS: {checked} distributed files verified (manifest excludes itself)')


if __name__=='__main__':
    main()
