"""
run_sweep_twomode.py
====================
G x V_pulling パラメータスタディの自動化スクリプト。

config.yaml を一時的に書き換えて run_twomode.py を順番に実行する。
結果は result/twomode_sweep/G{G}/V{V_pulling}/ に保存される。

使い方
------
    python run_sweep_twomode.py
    python run_sweep_twomode.py --config config.yaml --dry-run  # 実行確認のみ
"""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ─── パラメータグリッド定義 ───────────────────────────────────────────────────

G_VALUES = [1.0e+2]       # 温度勾配 [K/m]
V_VALUES = [2.0e-2, 3.0e-2, 4.0e-2, 5.0e-2, 6.0e-2, 7.0e-2, 8.0e-2, 9.0e-2, 1.0e-1]       # 引き上げ速度 [m/s]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="G x V_pulling sweep for twomode.")
    parser.add_argument("--config", default="config.yaml", help="ベースとなる config.yaml のパス")
    parser.add_argument("--script", default="run_twomode.py", help="実行する run スクリプト")
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

    total = len(G_VALUES) * len(V_VALUES)
    print(f"パラメータスタディ: {len(G_VALUES)} G値 × {len(V_VALUES)} V値 = {total} 条件")
    print(f"ベース config : {config_path}")
    print(f"実行スクリプト: {script_path}")
    print()

    results = []
    n = 0
    for G in G_VALUES:
        for V in V_VALUES:
            n += 1
            cooling = G * V
            out_dir = f"result/modify_solid_liquid_interface_1/twomode_2/{V*G}"
            print(f"[{n:2d}/{total}] G={G:.1e}, V={V:.1e}, G*V={cooling:.3e}  →  {out_dir}")

            if args.dry_run:
                continue

            # config をディープコピーして書き換え
            cfg = copy.deepcopy(base_cfg)
            cfg["physical"]["G"] = float(G)
            cfg["physical"]["V_pulling"] = float(V)

            # out_dir を上書き（run_twomode.py 内の f-string に合わせて書き換え）
            # run_twomode.py 側の out_dir 定義を使わせるため、
            # twomode セクションに out_dir キーを追加して読み込ませる方式にする
            cfg.setdefault("twomode", {})["out_dir_override"] = out_dir

            # 一時 config ファイルに書き出し
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tmp:
                yaml.dump(cfg, tmp, allow_unicode=True)
                tmp_path = tmp.name

            # run_twomode.py を呼び出し（一時 config を渡す）
            cmd = [sys.executable, str(script_path), "--config", tmp_path]
            print(f"    実行: {' '.join(cmd)}")

            ret = subprocess.run(cmd)
            Path(tmp_path).unlink(missing_ok=True)

            status = "OK" if ret.returncode == 0 else f"FAILED (code={ret.returncode})"
            results.append((G, V, status))
            print(f"    → {status}\n")

    if not args.dry_run:
        print("=" * 60)
        print("スイープ完了サマリー")
        print("=" * 60)
        for G, V, status in results:
            print(f"  G={G:.1e}, V={V:.1e}  {status}")


if __name__ == "__main__":
    main()
