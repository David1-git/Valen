"""Validate the installed candidate image without modifying its environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path('/opt/valen')
    env = dict(os.environ, VALEN_TEST_MODEL=args.model_path,
               VALEN_TEST_VLLM='1', VLLM_PLUGINS='valen',
               VLLM_USE_V2_MODEL_RUNNER='0')
    env.pop('PYTHONPATH', None)

    def run(command, filename):
        with (output / filename).open('w') as log:
            subprocess.run(command, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    (output / 'source-revision.txt').write_bytes((root / 'source-revision.txt').read_bytes())
    (output / 'plugin-wheel.sha256').write_bytes((root / 'plugin-wheel.sha256').read_bytes())
    run([sys.executable, '-m', 'pip', 'check'], 'pip-check.txt')
    run([sys.executable, '-m', 'pip', 'freeze'], 'runtime-packages.txt')
    run(['nvidia-smi'], 'hardware.txt')
    fingerprint = [sys.executable, '-m', 'valen_vllm.compatibility', '--source-tree']
    run(fingerprint, 'engine-before.json')
    run([sys.executable, '-m', 'unittest', 'discover', '-s',
         str(root / 'plugin/tests'), '-v'], 'tests.log')
    run([sys.executable, '-u', str(root / 'plugin/validate.py'),
         '--model-path', args.model_path, '--output-dir', str(output / 'parity')],
        'parity.log')
    run(fingerprint, 'engine-after.json')
    before = (output / 'engine-before.json').read_bytes()
    if before != (output / 'engine-after.json').read_bytes():
        raise RuntimeError('Engine sources changed during validation')
    receipt = {'status': 'passed',
               'source_revision': (root / 'source-revision.txt').read_text().strip(),
               'engine_fingerprint_sha256': hashlib.sha256(before).hexdigest()}
    (output / 'image-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
