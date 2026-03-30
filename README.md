# GPU-Accelerated Multi-Phase-Field Solidification

2D / 3D の multi-phase-field solidification を Numba CUDA で計算するためのコードです。

現在のリポジトリには次の実装が入っています。

- 既存の 2D 実装
- 3D random-mode 実装
- 3D 物理検証基盤

3D 側の現在の前提は次のとおりです。

- 境界条件: `x, y periodic` / `z Neumann (mirror)`
- 液相インデックス: `LIQ = 0`
- 界面異方性: 8 個の `{111}` 法線に対する `max |cos|` ベース
- APT: `mf`, `nf` による active parameter tracking
- torque term: 3D 版を実装済み
- 温度更新: 現在は一様冷却

## 実行環境

- Python 3.10 以降
- NVIDIA GPU
- CUDA が利用可能な Numba 環境

セットアップ例:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirement.txt
```

CUDA 認識確認:

```powershell
python -c "from numba import cuda; print(cuda.is_available())"
```

## 主なファイル

```text
src/
  gpu_kernels.py          2D CUDA kernels
  gpu_kernels_3d.py       3D CUDA kernels
  seed_modes.py           2D initial-condition helpers
  seed_modes_3d.py        3D initial-condition helpers
  plot_utils.py           2D plotting helpers
  plot_utils_3d.py        3D plotting helpers
  orientation_utils.py    quaternion / rotated {111} utilities

run_singlemode.py         2D single-grain run
run_twomode.py            2D bicrystal run
run_randommode.py         2D random Voronoi run
run_imagemode.py          2D image-seed run
run_randommode_3d.py      3D random Voronoi run

config.yaml               2D config
config_3d.yaml            3D config

tests/run_verification_3d.py
tests/verification_3d.py  3D verification suite
verification_report.md    最新の 3D 検証レポート
```

## 2D 実行

```powershell
python run_singlemode.py
python run_twomode.py
python run_randommode.py
python run_imagemode.py
```

出力先は主に `result/` 配下です。

## 3D 実行

3D は現在 `run_randommode_3d.py` を用います。

```powershell
python run_randommode_3d.py
```

小さめの smoke test 例:

```powershell
python run_randommode_3d.py --nx 32 --ny 32 --nz 32 --seed-height 8 --nsteps 10 --save-every 5
```

主なオプション:

- `--config`: 設定ファイルパス
- `--out-dir`: 出力先の上書き
- `--nx --ny --nz`: 格子サイズ上書き
- `--nsteps`: ステップ数上書き
- `--save-every`: 保存間隔上書き
- `--seed-height`: 初期 seed 高さ上書き
- `--n-solid`: 固相粒数上書き

既定の設定は `config_3d.yaml` にあります。

## 3D 検証

3D 検証は `tests/run_verification_3d.py` から実行します。

フル実行:

```powershell
python tests\run_verification_3d.py
```

ケース一覧:

```powershell
python tests\run_verification_3d.py --list-cases
```

一部だけ実行:

```powershell
python tests\run_verification_3d.py --cases bicrystal_competition_low_high_driving bicrystal_kinetic_dominated_regime_test
```

anisotropy / torque の切り替え:

```powershell
python tests\run_verification_3d.py --anisotropy off --torque off
```

主な出力:

- `verification_report.md`: 全体要約
- `tests/output/latest/verification_results.csv`: 数値結果一覧
- `tests/output/latest/<case_name>/`: 各ケースの PNG / CSV / JSON / case report

## 検証ケースの構成

quick sanity:

- `basic_constraints`
- `kmax_overflow_detection`
- `static_flat_interface`
- `two_d_limit_consistency`
- `boundary_conditions`
- `isotropic_orientation_independence`
- `anisotropic_preferred_growth`
- `torque_term_contribution`
- `grain_competition`
- `grain_boundary_groove_trijunction`
- `convergence_grid_dt_delta`

main physics:

- `single_grain_preferred_growth_benchmark`
- `directional_preference_map`
- `anisotropy_threshold_test`
- `bicrystal_competition_low_high_driving`
- `bicrystal_interfacial_energy_dominated_regime_test`
- `bicrystal_kinetic_dominated_regime_test`
- `multigrain_competition_extension`
- `groove_depth_only`
- `robust_groove_angle_estimation`

## 現在の確認結果

現時点の最新フル検証結果は次のとおりです。

- `20/20` cases passed
- `main physics: 9/9`
- `quick sanity: 11/11`

詳細は `verification_report.md` を参照してください。

## 重要な注意

### 1. `KMAX` は compile-time 定数です

`src/gpu_kernels_3d.py` の `KMAX` は CUDA JIT 時に固定されます。

そのため、`config_3d.yaml` の `gpu.KMAX` は、カーネル側と一致させてください。

### 2. `MAX_GRAINS` と `n_solid`

3D random-mode では `number_of_grain = n_solid + 1` です。

```text
number_of_grain <= MAX_GRAINS
```

を満たさないと実行時に停止します。

### 3. `threads_per_block`

現環境では 3D カーネルの既定値を `[4, 4, 4]` にしています。
`[8, 8, 8]` は GPU によっては resource limit に当たることがあります。

### 4. `tests/output/` は作業成果物です

検証画像や中間 CSV は `tests/output/` に出ます。
このディレクトリは生成物として扱っています。

## 参考

対象論文:

```text
Chuanqi Zhu, Yusuke Seguchi, Masayuki Okugawa, Chunwen Guo, Yuichiro Koizumi,
"Influences of growth front surfaces on the grain boundary development of
multi-crystalline silicon during directional solidification: 2D/3D multi-phase-field study",
Materialia 27 (2023) 101702.
https://doi.org/10.1016/j.mtla.2023.101702
```
