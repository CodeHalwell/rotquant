"""Run cached chat parity before complete-turn ternary QAT and inference upgrades."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ternary_research_v2_plan import DEFAULT_CONFIG, plan, validate_config


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--historical-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=Path('/content/rotquant-ternary-v2-work'))
    parser.add_argument('--budget-minutes', type=int, default=480)
    parser.add_argument('--stage', choices=('parity', 'full'), default='parity')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    if not 1 <= args.budget_minutes <= 480:
        parser.error('budget must be 1..480 minutes')
    if args.dry_run:
        print(json.dumps({**plan(config), 'stage': args.stage, 'budget_minutes': args.budget_minutes}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('the full v2 study requires CUDA; use --dry-run locally')
    from scripts.ternary_research_v2_study import V2Study
    V2Study(args, config).run()


if __name__ == '__main__':
    main()
