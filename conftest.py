"""Root conftest: put the repo root on sys.path so tests can `import common`,
`import scratch_gpt` etc. without installing the project."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
