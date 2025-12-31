from __future__ import annotations

import sys
from pathlib import Path

# 让 `pytest` 总能导入本仓库下的 `rl_algo/` 包（无需先 `pip install -e .`）。
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
