"""
run_randommode_bat.py
=====================
V_pulling パラメータスタディの自動化スクリプト（3D randommode 用）。

config_3d.yaml を一時的に書き換えて run_randommode_3d.py を順番に実行する。
結果は config の outdir + "/{n_solid} grains/{V*G:.1f}K/" に自動保存される。

使い方
------
    python run_randommode_bat.py
    python run_randommode_bat.py --config config_3d.yaml --dry-run  # 実行確認のみ
"""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ─── パラメータグリッド定義 ───────────────────────────────────────────────────

V_VALUES = [1.0e-1, 3.0e-1, 5.0e-1, 7.0e-1, 9.0e-1]   # 引き上げ速度 [m/s]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="V_pulling sweep for randommode_3d.")
    parser.add_argument("--config", default="config_3d.yaml", help="ベースとなる config_3d.yaml のパス")
    parser.add_argument("--script", default="run_randommode_3d.py", help="実行する run スクリプト")
    parser.add_argument("--dry-run", action="store_true", help="実行せず条件一覧だけ表示する")
    return parser.parse_args()


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config_path = Path(args.config)
    script_path = Path(args.script)

    if not config_path.exists():
        print(f"[ERROR] config が見つかりません: {config_path}")
        sys.exit(1)
    if not script_path.exists():
        print(f"[ERROR] スクリプトが見つかりません: {script_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    G = float(base_cfg["physical"]["G"])
    n_solid = int(base_cfg.get("randommode", {}).get("n_solid", 10))
    base_outdir = base_cfg["output"]["outdir"]

    total = len(V_VALUES)
    print(f"パラメータスタディ: {total} 条件")
    print(f"ベース config    : {config_path}")
    print(f"実行スクリプト   : {script_path}")
    print(f"G                : {G:.1e} [K/m]")
    print(f"n_solid          : {n_solid}")
    print()

    results = []
    for n, V in enumerate(V_VALUES, start=1):
        cooling = G * V
        nsteps = max(10000, int(50000 / (V * 100)))
        out_dir = f"{base_outdir}/{n_solid} grains/{cooling:.1f}K"
        print(f"[{n:2d}/{total}] V={V:.2e} m/s,  G*V={cooling:.1f} K/s,  nsteps={nsteps}  →  {out_dir}")

        if args.dry_run:
            continue

        # config をディープコピーして V_pulling と nsteps を書き換え
        cfg = copy.deepcopy(base_cfg)
        cfg["physical"]["V_pulling"] = float(V)
        cfg["grid"]["nsteps"] = nsteps

        # 一時 config ファイルに書き出し
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg, tmp, allow_unicode=True)
            tmp_path = tmp.name

        # run_randommode_3d.py を呼び出し
        cmd = [sys.executable, str(script_path), "--config", tmp_path]
        print(f"    実行: {' '.join(cmd)}")

        ret = subprocess.run(cmd)
        Path(tmp_path).unlink(missing_ok=True)

        status = "OK" if ret.returncode == 0 else f"FAILED (code={ret.returncode})"
        results.append((V, cooling, nsteps, status))
        print(f"    → {status}\n")

    if not args.dry_run:
        print("=" * 60)
        print("スイープ完了サマリー")
        print("=" * 60)
        for V, cooling, nsteps, status in results:
            print(f"  V={V:.2e} m/s,  G*V={cooling:.1f} K/s,  nsteps={nsteps}  {status}")


if __name__ == "__main__":
    main()
