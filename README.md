# GPU-Accelerated Multi-Phase Field Solidification Simulation (2D)

以下論文のモデルを GPU で再現実装したシミュレーションコードです。

> **再現元論文**:
> Chuanqi Zhu, Yusuke Seguchi, Masayuki Okugawa, Chunwen Guo, Yuichiro Koizumi,
> "Influences of growth front surfaces on the grain boundary development of multi-crystalline silicon during directional solidification: 2D/3D multi-phase-field study",
> *Materialia* 27 (2023) 101702.
> DOI: [10.1016/j.mtla.2023.101702](https://doi.org/10.1016/j.mtla.2023.101702)

論文の 2D シミュレーション（多結晶シリコンの方向性凝固・粒界発達）を GPU（numba CUDA）で高速化した再現実装です。
論文の `ver6-4` に対して、生成組織の断面を成長させる設定（`ver6-5`）に拡張しています。

---

## 目次

1. [概要](#概要)
2. [物理モデル](#物理モデル)
3. [数値手法](#数値手法)
4. [GPU 高速化：Active Parameter Tracking](#gpu-高速化active-parameter-tracking)
5. [コード構成](#コード構成)
6. [インストール・環境構築](#インストール環境構築)
7. [設定ファイル（config.yaml）](#設定ファイルconfigyaml)
8. [実行方法](#実行方法)
9. [シミュレーションモード](#シミュレーションモード)
10. [出力](#出力)
11. [主要セルの詳細説明](#主要セルの詳細説明)
12. [パラメータ変更ガイド](#パラメータ変更ガイド)
13. [よくあるエラーと対処法](#よくあるエラーと対処法)

---

## 概要

このコードは、冷却速度 $\dot{T} = G \cdot V_\text{pulling}$ で温度が一様に下がる系において、
液相から複数の固相結晶粒が成長していく過程を 2D でシミュレートします。

特徴：

- **多相フェーズフィールド法**（Steinbach 型 MPFM）に基づく多結晶凝固の再現
- **固液界面の異方性**：{111} ファセットを基準にした界面エネルギー異方性（論文 Appendix A2–A4）
- **粒界エネルギー**：等方性・非コヒーレント（論文の現モデルの仮定）
- **運動学的異方性**：界面法線と {111} 方向のなす角に依存するモビリティ補正（論文 Eq. 7, 9）
- **Active Parameter Tracking（APT）**：各セルで「接触している相のみ」を計算することで計算量を大幅削減
- **numba CUDA** による GPU カーネルで高速実行

### 論文との対応

| 論文の設定 | このコード |
|---|---|
| 2D simulation viewed from ⟨110⟩ orientation | `run_randommode.py` （`random_110` は notebook 固有） |
| 断面画像からシード方位を読み込む | `run_imagemode.py` |
| Periodic boundary on lateral sides | x 方向周期境界 |
| Symmetric boundary condition | y 方向 Neumann 境界 |
| Active Parameter Tracking | `kernel_update_nfmf` + `kernel_update_phasefield_active` |

---

## 物理モデル

### フェーズフィールド変数

各グレイン（相）に対して位相場 $\phi_i(\mathbf{x}, t) \in [0, 1]$ を定義します。
$\phi_i = 1$ は「その位置がグレイン $i$ に属する」ことを意味し、
$\sum_i \phi_i = 1$ の拘束条件を満たします。

- $i = 0$：液相（LIQ）
- $i = 1, 2, \ldots, N-1$：各固相結晶粒

### 進化方程式（論文 Eq. 1）

$$\frac{\partial \phi_i}{\partial t} = -\frac{2}{n} \sum_{j=1}^{n} m_{ij} \left\lbrace \sum_{k=1}^{n} \left[ \frac{1}{2}\left(\varepsilon_{ik}^2 - \varepsilon_{jk}^2\right)\nabla^2\phi_k + \left(w_{ik} - w_{jk}\right)\phi_k \right] - \frac{8}{\pi}\sqrt{\phi_i \phi_j} \,\Delta g_{ij} \right\rbrace$$

ここで $n$ は当該セルでのアクティブ相数、$m_{ij}$ はフェーズフィールドモビリティです。

### 化学的駆動力（論文 Eq. 4）

$$\Delta g = \Delta T \cdot S_f = (T_\text{melt} - T) \cdot S_f$$

- $S_f = 2.12 \times 10^4\ \text{J/(K}\cdot\text{m}^3\text{)}$：融解エントロピー
- 固液ペア $(i=\text{solid},\ j=\text{liquid})$ にのみ適用。粒界（固固）は $\Delta g = 0$

### 界面パラメータ変換（論文 Eq. 2, 3, 5）

$$\varepsilon = \sqrt{\frac{8\delta\gamma}{\pi^2}}, \quad w = \frac{4\gamma}{\delta}, \quad m = \frac{\pi^2 \beta}{8\delta}$$

- $\delta$：拡散界面幅（`delta_factor × dx`）
- $\gamma$：界面エネルギー $[\text{J/m}^2]$
- $\beta$：attachment kinetic coefficient $[\text{m}^4/(\text{J}\cdot\text{s})]$

### 固液界面の異方性（論文 Eq. 6, 8, Appendix A2–A4）

$$\varepsilon(\theta) = \varepsilon_0 \cdot a(\theta)$$

**異方性関数** $a(\theta)$（ラウンディング付き）：

$$a(\theta) = \mu \left[1 + \zeta\left(C + \tan\alpha_0 \cdot S\right)\right]$$

$$C = \sqrt{\cos^2\theta + \rho^2}, \quad S = \sqrt{\sin^2\theta + \rho^2} = \sqrt{1 - \cos^2\theta + \rho^2}$$

| 記号 | コード変数 | 論文 Table 1 値 | 意味 |
|---|---|---|---|
| $\alpha_0$ | `a0_deg` | 54.7° | カスプ形状を規定する角度（{100}方向で表面エネルギー最大） |
| $\zeta$ | `delta_a` | 0.36 | カスプの深さを規定する係数 |
| $\mu$ | `mu_a` | 0.6156 | 振幅係数（最大値の正規化） |
| $\rho$ | `p_round` | 0.05 | カスプの尖りをなめらかにするパラメータ |

> $\theta$ は界面法線と最近接 {111} 方向のなす角。2D では各グレインの8つの {111} 方向 $(\pm1,\pm1,\pm1)/\sqrt{3}$ との内積の絶対値の最大値を $\cos\theta$ として使用。

### 固液異方性の数値実装：トルク項（論文 Appendix A1, A10–A15）

等方性の場合 $\varepsilon^2 \nabla^2\phi$ だったものが、異方性では：

$$\varepsilon^2\nabla^2\phi \;\longrightarrow\; \nabla\!\left(\varepsilon(\theta)^2 \nabla\phi\right) + \sum_{p=x,y} \frac{\partial}{\partial p}\!\left[\varepsilon(\theta)\frac{\partial\varepsilon(\theta)}{\partial \phi_p}|\nabla\phi|^2\right]$$

右辺第2項が「トルク項」で、3つのサブ項 (A11) に展開されます：

$$= \varepsilon_0^2 \sum_{p=x,y} \left\lbrace \underbrace{\frac{\partial a}{\partial p}\frac{\partial a}{\partial \phi_p}|\nabla\phi|^2}_{\text{(I)}} + \underbrace{a\frac{\partial}{\partial p}\!\left(\frac{\partial a}{\partial \phi_p}\right)|\nabla\phi|^2}_{\text{(II)}} + \underbrace{a\frac{\partial a}{\partial \phi_p}\frac{\partial}{\partial p}\!\left(|\nabla\phi|^2\right)}_{\text{(III)}} \right\rbrace$$

各偏微分は (A12)–(A15) に従い $\partial\cos\theta/\partial p$、$\partial\cos\theta/\partial\phi_p$ に帰着させて差分で実装。

### 運動学的異方性（論文 Eq. 7, 9）

$$m(\theta) = m_0 \cdot b(\theta)$$

$$b(\theta) = \xi + (1 - \xi) \cdot \tan(\omega\theta) \cdot \tanh\!\left(\frac{1}{\tan(\omega\theta)}\right)$$

| 記号 | コード変数 | 論文 Table 1 値 | 意味 |
|---|---|---|---|
| $\xi$ | `ksi` | 0.30 | カスプの深さ（{111}方向のモビリティ比の下限） |
| $\omega$ | `omg_deg` | 10° | カスプの幅（エンドが 10° に設定） |

### 論文 Table 1 の標準パラメータ値

| 量 | 記号 | 値 | 単位 |
|---|---|---|---|
| 格子間隔 | $\Delta x$ | $2.0 \times 10^{-5}$ | m |
| 時間刻み | $\Delta t$ | $2.0 \times 10^{-3}$ | s |
| 融点 | $T_m$ | 1687.0 | K |
| {100} 方向の界面エネルギー | $\gamma_{100}$ | 0.44 | J/m² |
| {111} 方向の界面エネルギー | $\gamma_{111}$ | 0.32 | J/m² |
| 粒界エネルギー | $\gamma_\text{GB}$ | 0.60 | J/m² |
| 融解エントロピー | $S_f$ | $2.12 \times 10^4$ | J/(K·m³) |
| {100} 方向の attachment kinetic coefficient | $\beta_{100}$ | $4.62 \times 10^{-4}$ | m⁴/(J·s) |
| 粒界の attachment kinetic coefficient | $\beta_\text{GB}$ | $0.05\,\beta_{100}$ | — |

---

## 数値手法

### 空間離散化

- **一様格子**：$N_x \times N_y$、格子間隔 $\Delta x = \Delta y = dx$
- **座標系**：$l$（x 方向）、$m$（y 方向）
- **境界条件**：
  - $x$ 方向：**周期境界**（結晶が端を超えて周期的につながる）
  - $y$ 方向：**Neumann 境界**（$\partial\phi/\partial y = 0$、鏡像条件）
- **勾配**：2次中心差分
- **Laplacian**：2次中心差分（5点ステンシル）
- **交差微分** $\partial^2\phi/\partial x \partial y$：4点差分

### 時間積分

前進 Euler 法（1次精度）：

$$\phi_i^{n+1} = \phi_i^n + \Delta t \cdot \dot{\phi}_i^n$$

各タイムステップ後に **正規化**（$\sum_i \phi_i = 1$ の回復）と **クリッピング**（$\phi_i \in [0,1]$）を実施します。

### 温度場

全セルで一様に温度が下がります：

$$T^{n+1}(x,y) = T^n(x,y) - G \cdot V_\text{pulling} \cdot \Delta t$$

初期温度分布は線形勾配：$T_0(m) = T_\text{melt} + G \cdot (m - m_\text{seed}) \cdot \Delta y$

---

## GPU 高速化：Active Parameter Tracking

### 問題の背景

多相フェーズフィールドでは全 $N$ 相を全セルで更新すると $O(N^2 \cdot N_x \cdot N_y)$ の計算量になります。
$N=20$ なら通常の方法より400倍のコストになります。

### APT の仕組み

各セルには「実際に存在している相（$\phi_i > 0$ またはその隣接セルで $\phi_i > 0$）」しか関与しません。
それ以外の相は $\phi_i = 0$ のまま変化しません。

APT では 2 つの補助配列を使います：

| 配列 | 形状 | 意味 |
|---|---|---|
| `nf[l, m]` | `(nx, ny)` | セル $(l,m)$ でのアクティブ相数 |
| `mf[t, l, m]` | `(MAX_GRAINS, nx, ny)` | セル $(l,m)$ の $t$ 番目アクティブ相の ID |

`kernel_update_nfmf` カーネルで毎ステップ `nf`, `mf` を更新し、
`kernel_update_phasefield_active` カーネルでは `for t in range(nf[l,m])` のみループします。

### KMAX の意味

`KMAX` はカーネル内のローカル配列サイズです：

```python
lap_sl = cuda.local.array(KMAX, float32)
b_sl   = cuda.local.array(KMAX, float32)
```

GPU のローカルメモリ（レジスタ or ローカルメモリ）は静的サイズが必要なため、
1セル内のアクティブ相数の上限として `KMAX` を設定します。
通常の凝固では界面付近でも4〜8相程度なので `KMAX=18` で十分です。

---

## コード構成

```
GPU-multi-phase-field-model-solification_2d.ipynb  # 元ノートブック（参照用）
config.yaml                                         # 全モード共通パラメータ
requirement.txt                                     # Python依存ライブラリ
result/                                             # 出力（自動生成）
  singlemode/                                       # run_singlemode.py の出力
  twomode/                                          # run_twomode.py の出力
  randommode/                                       # run_randommode.py の出力
  imagemode/                                        # run_imagemode.py の出力
  new/{M_GB_ratio}/{cooling_rate}/                  # notebook 出力（旧）

# ---- 実行スクリプト（モード別） ----
run_singlemode.py    # 検証用：単結晶固液界面成長（number_of_grain=2）
run_twomode.py       # 検証用：2粒競合成長・粒界形成（number_of_grain=3）
run_randommode.py    # 本番用：Voronoi ランダム多結晶（number_of_grain=n_solid+1）
run_imagemode.py     # 本番用：画像由来粒構造（number_of_grain=色数+1）

# ---- 共通ライブラリ ----
gpu_kernels.py       # CUDA デバイス関数・カーネル（ノートブックから抽出）
seed_modes.py        # 初期条件生成（phi, temp, grain_map）
orientation_utils.py # 四元数・{111}法線の計算
plot_utils.py        # PNG 保存ユーティリティ（Agg バックエンド）

# ---- 検証スクリプト ----
validate_modes.py    # CPU のみで初期条件・数値健全性を検証
```

### モジュールの役割

| ファイル | 提供する関数 |
|---|---|
| `gpu_kernels.py` | `kernel_update_nfmf`, `kernel_update_phasefield_active`, `kernel_update_temp` および全デバイス関数 |
| `seed_modes.py` | `init_singlemode_phi`, `init_twomode_phi`, `generate_random_grain_map`, `load_grain_map_from_image`, `init_phi_from_grain_map`, `init_temperature_field`, `build_interaction_matrices` |
| `orientation_utils.py` | `build_quaternion_from_config`, `rgb_to_unit_quaternion`, `load_quaternions_from_csv`, `assign_quaternions_to_grains`, `compute_rotated_n111` |
| `plot_utils.py` | `save_phase_map`, `save_temperature_map` |

### ノートブックのセル構成（参照用）

| セル | ID | 内容 |
|---|---|---|
| 0 | `d5b73807` | **設定読み込み**（import、`config.yaml` 読み込み、出力先作成） |
| 1 | `7d0ba150` | **結晶方位の設定**（`random_110` / `image_line`、四元数・{111}法線の事前計算） |
| 2 | `980b2eae` | **APT 配列と相互作用行列の初期化**（`mf`, `nf`, `wij`, `aij`, `mij`） |
| 3 | `c7cc73f9` | **界面パラメータ変換**（`eps_from_gamma`, `w_from_gamma`, `mij_from_M`） |
| 4 | `86afb621` | **CUDAデバイス関数**：異方性関数 `calc_a_from_cos`, `calc_b_from_cos` |
| 5 | `bd13bdb7` | **CUDAデバイス関数**：最近接 {111} との cos 計算 |
| 6 | `1562cec3` | **CUDAデバイス関数**：境界条件インデックス、勾配 |
| 7 | `69701ca9` | **CUDAデバイス関数**：セルごとの $\varepsilon^2$ 計算 |
| 8 | `0625991c` | **CUDAデバイス関数**：異方性拡散項（第1項） |
| 9 | `a5aa21e6` | **CUDAデバイス関数**：2階微分、トルク項 (A11–A15) |
| 10 | `7dddaa06` | **CUDAカーネル**：`kernel_update_nfmf`（APT配列更新） |
| 11 | `840e0a86` | **CUDAカーネル**：`kernel_update_phasefield_active`（メイン更新） |
| 12 | `df60f47a` | **CUDAカーネル**：`kernel_update_temp`（温度更新） |
| 13 | `c9a33b23` | **初期化**（$\phi$, $T$ の初期値設定、初期図の保存） |
| 14 | `13ef5be0` | **GPU転送**（CPU→GPU、グリッド設定） |
| 15 | `37362037` | **メインループ**（時間発展・可視化・保存） |

---

## インストール・環境構築

### 必要条件

- Python 3.10 以上
- NVIDIA GPU（CUDA 対応）
- CUDA Toolkit（システムにインストール済みであること）
  → `nvcc --version` で確認

### venv + pip での環境構築

```bash
# 仮想環境を作成
python -m venv .venv

# 有効化（Windows）
.venv\Scripts\activate

# 有効化（Linux/Mac）
source .venv/bin/activate

# パッケージのインストール
pip install -r requirement.txt
```

### 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `numpy` | 配列計算全般 |
| `numba` | CUDA JIT コンパイル（GPU カーネル定義） |
| `scipy` | 四元数 → 回転行列変換（`Rotation`） |
| `matplotlib` | 可視化・PNG 保存（Agg バックエンド） |
| `pillow` | 粒構造画像の読み込み（`imagemode`） |
| `pyyaml` | `config.yaml` の読み込み |
| `ipykernel`, `jupyter` | Jupyter Notebook 実行環境（ノートブック使用時のみ） |

### CUDA の確認

```python
from numba import cuda
print(cuda.gpus)         # 認識されている GPU 一覧
cuda.detect()            # 詳細情報
```

---

## 設定ファイル（config.yaml）

すべての物理・数値パラメータをここで管理します。実行スクリプトを直接編集せずに条件を変えられます。

```yaml
grid:
  nx: 256
  ny: 256
  dx: 1.0e-4
  dy: 1.0e-4
  dt: 1.0e-4
  nsteps: 20000

physical:
  T_melt: 1687
  G: 1.0e+2             # 温度勾配 [K/m]
  V_pulling: 5.0e-2     # 引き抜き速度 [m/s]  → 冷却速度 = G × V_pulling
  Sf: 2.12e+4

interface:
  delta_factor: 6.0
  gamma_100: 0.44
  gamma_GB: 0.60

anisotropy:
  a0_deg: 54.7
  delta_a: 0.36
  mu_a: 0.6156
  p_round: 0.05
  ksi: 0.30
  omg_deg: 10.0

mobility:
  M_SL: 5.0e-5
  M_GB_ratio: 0.1

gpu:
  MAX_GRAINS: 20
  KMAX: 18
  threads_per_block: [16, 16]

seed:                         # notebook (image_line モード) 専用
  mode: "image_line"
  number_of_grain_fallback: 17
  random_seed: 42
  image_path: "path/to/seed.bmp"
  line_axis: "horizontal"
  line_index: null
  color_tolerance: 0
  height: 32

output:
  dir_template: "result/new/{M_GB_ratio:.2f}/{cooling_rate:.1e}"
  save_every: 200

# --- 検証モード ---
singlemode:
  seed_height: 32
  orientation_type: "euler"
  euler_deg: [0.0, 0.0, 0.0]

twomode:
  seed_height: 32
  split_ratio: 0.5
  grain1_seed_offset: 0
  grain2_seed_offset: 0
  grain1:
    orientation_type: "euler"
    euler_deg: [0.0, 45.0, 0.0]
  grain2:
    orientation_type: "euler"
    euler_deg: [0.0, 0.0, 0.0]

# --- 本番モード ---
randommode:
  seed_height: 32
  n_solid: 10
  random_seed: 42
  orientation_mode: "random"   # "random" または "file"
  orientation_seed: 42
  orientation_csv: ""

imagemode:
  seed_height: 32
  image_path: "path/to/grain_map.bmp"
  orientation_mode: "rgb"      # "random", "file", または "rgb"
  orientation_seed: 42
  orientation_csv: ""
```

### 各パラメータの詳細

#### `grid`

| パラメータ | 説明 | 注意 |
|---|---|---|
| `nx`, `ny` | 計算領域のサイズ | 2の累乗（128, 256, 512）が GPU 効率に良い |
| `dx`, `dy` | 格子間隔 [m] | 現在は `dx == dy` を前提としてコードが書かれている |
| `dt` | 時間刻み [s] | 安定性条件：`dt < dx² / (2 * M * eps²)` 程度 |
| `nsteps` | 総ステップ数 | 凝固完了まで十分な値を設定 |

#### `physical`

| パラメータ | 説明 |
|---|---|
| `T_melt` | 融点。初期温度は $T_\text{melt} + G \cdot (y - y_\text{seed})$ で設定 |
| `G` | 温度勾配（初期温度分布の傾き）[K/m] |
| `V_pulling` | 凝固フロントの移動速度 [m/s]。冷却速度 $\dot{T} = G \cdot V$ |
| `Sf` | 融解エントロピー（液相←→固相の駆動力スケール） |

#### `interface`

| パラメータ | 説明 |
|---|---|
| `delta_factor` | 界面の厚さを格子間隔の何倍にするか。6〜8 が標準的 |
| `gamma_100` | (100) 面方向を基準とした固液界面エネルギー |
| `gamma_GB` | 粒界エネルギー。`gamma_100` より大きくすると粒界が移動しやすい |

#### `anisotropy`

| パラメータ | 説明 |
|---|---|
| `a0_deg` | 54.7° は {100} 方向（{111} に対して）。論文の最適値 |
| `delta_a` | 大きいほど界面エネルギーの方位依存性が強くなる |
| `mu_a` | $a(c)$ の基準値。$\mu_a \approx 1/(1 + \delta_a \cdot C(c_0) + \delta_a \tan\alpha_0 \cdot S(c_0))$ に設定すると $\gamma_{100}$ が一致 |
| `p_round` | 0 に近いほどファセット（平坦面）が明確になるが数値的に不安定になる |
| `ksi` | 0.3 程度。$b(\theta)$ の下限（最も遅い方向のモビリティ比） |
| `omg_deg` | 角度スケール。大きいほど $b(\theta)$ の変化がなだらか |

#### `gpu`

| パラメータ | 説明 |
|---|---|
| `MAX_GRAINS` | `nf`/`mf` 配列の最大次元。`number_of_grain` 以上が必要 |
| `KMAX` | カーネル内ローカル配列サイズ。`MAX_GRAINS` 以上であれば安全 |
| `threads_per_block` | CUDA スレッドブロック。通常 `[16, 16]` または `[32, 8]` |

> **注意**：`MAX_GRAINS` と `KMAX` を変えた場合は、カーネルを **再コンパイル**（プロセス再起動または `gpu_kernels.py` を再インポート）する必要があります。

---

## 実行方法

### Python スクリプト（推奨）

仮想環境を有効化した状態で、モードに応じたスクリプトを実行します：

```bash
# 検証用：単結晶固液界面
python run_singlemode.py

# 検証用：2粒競合成長
python run_twomode.py

# 本番用：Voronoi ランダム多結晶
python run_randommode.py

# 本番用：画像由来粒構造（config.yaml の imagemode.image_path を設定してから）
python run_imagemode.py
```

### Jupyter Notebook（元実装）

```bash
jupyter notebook
```

`GPU-multi-phase-field-model-solification_2d.ipynb` を開き、Cell 0 から順に実行します。
notebook は `seed.mode: "random_110"` または `"image_line"` で動作します。

> **重要**：`MAX_GRAINS` / `KMAX` を変更した後は必ず Cell 0 から全セルを再実行してください。

### 初期条件のみ検証（GPU 不要）

```bash
python validate_modes.py
```

CPU のみで初期条件の構造・数値健全性（phi 範囲、sum=1、四元数ノルム等）を確認できます。

---

## シミュレーションモード

### モード比較

| モード | スクリプト | 粒数 | 初期粒配置 | 方位指定 |
|---|---|---|---|---|
| `singlemode` | `run_singlemode.py` | 1固相 + 液相 | 平坦界面（y一様） | Euler角 or 四元数 |
| `twomode` | `run_twomode.py` | 2固相 + 液相 | 左右分割 | 粒ごとに個別設定 |
| `randommode` | `run_randommode.py` | n_solid + 液相 | Voronoi テッセレーション | random / file |
| `imagemode` | `run_imagemode.py` | 色数 + 液相 | 画像由来 | random / file / rgb |

---

### singlemode（検証用）

単結晶が液相へ成長するシナリオ。界面が平坦に進むことでカーネルの基本動作を確認できます。

```yaml
singlemode:
  seed_height: 32
  orientation_type: "euler"
  euler_deg: [0.0, 0.0, 0.0]
  # orientation_type: "quaternion"
  # quaternion: [0.0, 0.0, 0.0, 1.0]
```

出力：`result/singlemode/`

---

### twomode（検証用）

2粒が競合成長し粒界を形成するシナリオ。方位差による成長速度の違いを確認できます。

```yaml
twomode:
  seed_height: 32
  split_ratio: 0.5          # grain1/grain2 の境界位置（0〜1）
  grain1_seed_offset: 0     # grain1 の初期固相高さ追加オフセット [grid pts]
  grain2_seed_offset: 0
  grain1:
    orientation_type: "euler"
    euler_deg: [0.0, 45.0, 0.0]
  grain2:
    orientation_type: "euler"
    euler_deg: [0.0, 0.0, 0.0]
```

出力：`result/twomode/`

---

### randommode（本番用）

Voronoi テッセレーションで $n\_solid$ 個の粒を生成。多粒競合成長・多重粒界の挙動を確認できます。

```yaml
randommode:
  seed_height: 32
  n_solid: 10               # 固相粒数（number_of_grain = n_solid + 1）
  random_seed: 42           # Voronoi シード点配置の乱数シード
  orientation_mode: "random"  # "random" または "file"
  orientation_seed: 42
  orientation_csv: ""       # orientation_mode = "file" のとき使用
```

**`n_solid + 1 > MAX_GRAINS` のとき実行時エラーになります**（事前チェック）。

出力：`result/randommode/`

---

### imagemode（本番用）

外部画像から粒構造を読み込みます。各ユニーク RGB 色 = 1粒として自動的に粒 ID を割り当てます。

```yaml
imagemode:
  seed_height: 32
  image_path: "D:/path/to/grain_map.bmp"  # 必須
  orientation_mode: "rgb"   # "random", "file", または "rgb"
  orientation_seed: 42
  orientation_csv: ""
```

**`orientation_mode` の選択肢：**

| 値 | 意味 |
|---|---|
| `"random"` | 各粒にランダムな単位四元数（`orientation_seed` で再現性確保） |
| `"file"` | CSV から読み込み（1行 = 1粒、列 = x, y, z, w） |
| `"rgb"` | RGB 値から決定論的に四元数を生成（同色 → 同方位、ノートブック準拠） |

**RGB → 四元数の変換（`rgb_to_unit_quaternion`、ノートブック準拠）：**

$$v_k = \frac{R_k}{255} \cdot 2 - 1 \quad (k = R, G, B)$$

$$w = \sqrt{\max(1 - |v|^2,\; 0)}, \quad q = (v_x, v_y, v_z, w) / |q|$$

これにより RGB 値が単位四元数（SciPy の `(x, y, z, w)` 形式）として解釈されます。

**検出色数 + 1 > `MAX_GRAINS` のとき実行時エラーになります**（使用する画像の色数を事前に確認してください）。

出力：`result/imagemode/`

---

## 出力

### 保存場所

| モード | 出力ディレクトリ |
|---|---|
| `singlemode` | `result/singlemode/` |
| `twomode` | `result/twomode/` |
| `randommode` | `result/randommode/` |
| `imagemode` | `result/imagemode/` |
| notebook | `result/new/{M_GB_ratio:.2f}/{cooling_rate:.1e}/` |

### 保存される画像

全モード共通で以下の3ファイルを最初に保存します：

- `step_0.png`：初期状態のグレインマップ
- `initial_phase_map.png`：`step_0.png` と同内容（別名コピー）
- `initial_temperature.png`：初期温度分布

その後、`config.yaml` の `save_every` ステップごとに：

- `step_200.png`, `step_400.png`, ...：グレインマップ

### 画像の見方

- **カラーマップ**：`tab20`（各色 = 各グレイン ID）
- ID 0（濃い青）= 液相、ID 1 以上 = 各固相グレイン
- 横軸 = x 方向（0〜nx-1）、縦軸 = y 方向（origin="lower"、下が y=0）
- y=0 側が初期固相（下から成長）

---

## 主要セルの詳細説明

### Cell 0：設定読み込み

Cell 0 ではライブラリの import、`config.yaml` の読み込み、各パラメータの Python 変数化、出力ディレクトリの作成を行います。

```python
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
```

### Cell 1：結晶方位と {111} 法線の事前計算

```python
grain_quaternions[gid]  # shape: (N, 4), SciPy (x,y,z,w) 形式
grain_n111[gid]         # shape: (N, 8, 3), 各グレインの8つの{111}方向（グローバル座標）
```

`grain_n111` は毎ステップカーネル内で使われます。
事前計算することで GPU カーネルの演算量を削減しています。

8つの {111} 方向：$\frac{1}{\sqrt{3}}(\pm1, \pm1, \pm1)$ を回転行列で変換したもの。
2D シミュレーションでは $(n_x, n_y)$ 成分のみ使用します。

### Cell 3：界面パラメータ変換

相互作用行列の充填：

| 行列 | 粒界 (solid-solid) | 固液界面 (solid-liquid) |
|---|---|---|
| `wij[i,j]` | $W_\text{GB} = 4\gamma_\text{GB}/\delta$ | $W_0^\text{SL} = 4\gamma_{100}/\delta$（基準値） |
| `aij[i,j]` | $\varepsilon_\text{GB}$ | $\varepsilon_0^\text{SL}$（基準値）|
| `mij[i,j]` | $(\pi^2/8\delta) \cdot M_\text{GB}$ | $(\pi^2/8\delta) \cdot M_\text{SL}$ |

> 固液界面の $\varepsilon$ と $W$ は **異方性により実行時に上書き** されます（`aij`/`wij` に格納された値はデフォルト値）。

### Cell 4：デバイス関数 `calc_a_from_cos` / `calc_b_from_cos`

```python
@cuda.jit(device=True, inline=True)
def calc_a_from_cos(cost, a0, delta_a, mu_a, p_round):
    # 論文 Appendix A 式 (A2)-(A4)
    # cost: cos(θ) の絶対値（最近接{111}との角）
    # 戻り値: 異方性係数 a(cost)
```

```python
@cuda.jit(device=True, inline=True)
def calc_b_from_cos(best_cost, ksi, omg):
    # 論文 式(7), (9)
    # best_cost: cos(θ) の最大値
    # 戻り値: 運動学的異方性係数 b(θ)
```

### Cell 6：境界条件インデックス

```python
idx_xp(l, nx)  # l+1（端では 0 へラップ）：周期境界
idx_xm(l, nx)  # l-1（端では nx-1 へラップ）：周期境界
idx_yp(m, ny)  # m+1（端では ny-1：Neumann反射）
idx_ym(m, ny)  # m-1（端では 0：Neumann反射）
```

### Cell 8：`aniso_term1_solid`

固液界面の異方性 Laplacian 項（いわゆる「主要項」）：

$$\nabla \cdot (\varepsilon^2 \nabla \phi_0) \approx \frac{({\varepsilon^2_c + \varepsilon^2_{x+}})(\phi_{x+} - \phi_c) - ({\varepsilon^2_c + \varepsilon^2_{x-}})(\phi_c - \phi_{x-})}{2\Delta x^2} + (\text{y方向も同様})$$

セルごとに $\varepsilon^2 = \varepsilon_0^2 \cdot a^2(\cos\theta)$ を計算してから差分します。

### Cell 9：トルク項 `torque_A11`

異方性がある場合に生まれる付加項で、界面の向きを結晶の優先方向に引き付ける効果があります。
論文 Appendix A の式 (A11)〜(A15) に対応します。
計算には液相 $\phi_0$ の1・2階偏微分が必要です：

```python
phixx, phiyy, phixy = d2_phi_xy(phi, 0, l, m, nx, ny, dx)
```

### Cell 10：`kernel_update_nfmf`

毎ステップ呼び出し、各セルの「アクティブ相リスト」を更新します。

判定条件：
```
phi[i, l, m] > 0  OR  隣接セルのいずれかで phi[i, ...] > 0
```

後者の条件（隣接セルチェック）により、「次のステップで界面が広がる」相も事前にアクティブと判定します。

### Cell 11：`kernel_update_phasefield_active`（メインカーネル）

メインの時間発展カーネル。1 GPU スレッドが 1 セル $(l, m)$ を担当します。

実行フロー：

1. **アクティブ相ごとに異方性項をキャッシュ**（`lap_sl[t]`, `b_sl[t]`）
2. **支配固相 `i_s`** を特定（$\phi_i$ が最大の固相）→ `w_sl` を計算
3. **進化方程式を積分**：全アクティブ相ペア $(i, j)$ の組み合わせで `dpi` を蓄積
4. **クリッピング** $[0, 1]$ と **正規化** $\sum_i \phi_i = 1$

カーネル内での駆動力の符号：

```python
# 固相 i が液相 j から固相へ成長する場合：
driving_force = -Sf * (T - T_melt)  # T < T_melt で負 → 固相が成長
```

---

## パラメータ変更ガイド

### 冷却速度を変えたい

```yaml
physical:
  G: 1.0e+3          # 温度勾配を10倍に
  V_pulling: 5.0e-2  # 速度は同じ → 冷却速度10倍
```

または

```yaml
physical:
  G: 1.0e+2
  V_pulling: 5.0e-1  # 速度を10倍
```

どちらも $\dot{T} = G \cdot V$ が変わります。

### グレイン数を変えたい（randommode）

```yaml
randommode:
  n_solid: 15        # 15グレイン

gpu:
  MAX_GRAINS: 20     # n_solid + 1 = 16 以上なら OK（余裕あり）
  KMAX: 18
```

> `MAX_GRAINS` を変えたら **カーネルを再コンパイル**（プロセス再起動）。

### 計算領域を大きくしたい

```yaml
grid:
  nx: 512
  ny: 512
```

メモリ使用量は $O(N \cdot N_x \cdot N_y)$ で増加します。
`float32` で `MAX_GRAINS=20, nx=ny=512` の場合：`20 × 512 × 512 × 4 bytes ≈ 20 MB`

### 異方性を強くしたい

```yaml
anisotropy:
  delta_a: 0.5       # 0.36 → 0.5（より強い異方性）
  p_round: 0.01      # よりシャープなファセット（数値的に注意）
```

### 粒界が動きやすくしたい

```yaml
mobility:
  M_GB_ratio: 0.5    # デフォルト 0.10 → 5倍
```

---

## よくあるエラーと対処法

### `FileNotFoundError: image not found`

`imagemode` で `image_path` が存在しない場合。

```yaml
imagemode:
  image_path: "D:/正しいパス/grain_map.bmp"
```

### `ValueError: imagemode.image_path is not set`

`imagemode.image_path` が空文字のまま実行した場合。`config.yaml` で正しいパスを設定してください。

### `RuntimeError: number_of_grain=XX exceeds MAX_GRAINS=YY`

`randommode` や `imagemode` で検出された粒数が多すぎます。

対処法：
1. `gpu.MAX_GRAINS` を増やす（カーネル再コンパイル必要）
2. `randommode.n_solid` を減らす
3. `imagemode` では色数の少ない画像を使うか、前処理で色を削減する

### `CUDA out of memory`

`nx`, `ny`, `MAX_GRAINS` を小さくするか、他の GPU プロセスを終了してください。

### `numba.cuda.cudadrv.driver.CudaAPIError`

CUDA Toolkit が正しくインストールされているか確認：

```bash
nvcc --version
nvidia-smi
python -c "from numba import cuda; cuda.detect()"
```

### カーネルが古い定義を使っている

`MAX_GRAINS` / `KMAX` を変更した後は Python プロセスを再起動してください。
Jupyter の場合は **Kernel → Restart & Run All** で全セルを最初から実行してください。

---

## 理論的背景の補足

### なぜ {111} ファセットを使うのか

FCC 金属（シリコンなど）では {111} 面が最密充填面であり、固液界面エネルギーが最小になります。
2D シミュレーションでは 3D の {111} 法線を $(n_x, n_y)$ に射影して使用します。

### トルク項が必要な理由

$\varepsilon$ が方向依存の場合、通常の $\nabla \cdot (\varepsilon^2 \nabla\phi)$ だけでは不足し、
$\frac{\partial}{\partial x}\left(\varepsilon^2 \frac{\partial \phi}{\partial y}\right)$ 型のクロス項が残ります。
これが「トルク項」と呼ばれる補正で、ファセット形状の正確な再現に不可欠です。

### 正規化（制約条件の維持）について

前進 Euler では数値誤差により $\sum_i \phi_i \neq 1$ になります。
クリッピング + 正規化により毎ステップ制約を回復します：

```python
phi_new[i] = clip(phi_new[i], 0, 1)
phi_new[i] /= sum(phi_new)   # アクティブ相の合計が1になるよう正規化
```

非アクティブ相（$\phi_i = 0$）はそのまま 0 に保たれます。

---

## ライセンス・引用

このコードは以下の論文のモデルを再現実装したものです。研究に使用する場合は必ず原論文を引用してください：

```
Chuanqi Zhu, Yusuke Seguchi, Masayuki Okugawa, Chunwen Guo, Yuichiro Koizumi,
"Influences of growth front surfaces on the grain boundary development of
multi-crystalline silicon during directional solidification: 2D/3D multi-phase-field study",
Materialia 27 (2023) 101702.
https://doi.org/10.1016/j.mtla.2023.101702
```

また、多相フェーズフィールド法の基礎については以下も参照：

```
I. Steinbach, "Phase-field models in materials science",
Modell. Simul. Mater. Sci. Eng. 17(7) (2009) 073001.
```
