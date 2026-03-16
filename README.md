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
9. [出力](#出力)
10. [シード（初期結晶核）の設定](#シード初期結晶核の設定)
11. [各セルの詳細説明](#各セルの詳細説明)
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
| 2D simulation viewed from ⟨110⟩ orientation | `seed_mode: "random_110"` |
| 断面画像からシード方位を読み込む | `seed_mode: "image_line"` |
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

$$\frac{\partial \phi_i}{\partial t} = -\frac{2}{n} \sum_{j=1}^{n} m_{ij} \left\{ \sum_{k=1}^{n} \left[ \frac{1}{2}\left(\varepsilon_{ik}^2 - \varepsilon_{jk}^2\right)\nabla^2\phi_k + \left(w_{ik} - w_{jk}\right)\phi_k \right] - \frac{8}{\pi}\sqrt{\phi_i \phi_j} \,\Delta g_{ij} \right\}$$

ここで $n$ は当該セルでのアクティブ相数、$m_{ij}$ はフェーズフィールドモビリティです。

### 化学的駆動力（論文 Eq. 4）

$$\Delta g = \Delta T \cdot S_f = (T_\text{melt} - T) \cdot S_f$$

- $S_f = 2.12 \times 10^4\ \text{J/(K·m}^3\text{)}$：融解エントロピー
- 固液ペア $(i=\text{solid},\ j=\text{liquid})$ にのみ適用。粒界（固固）は $\Delta g = 0$

### 界面パラメータ変換（論文 Eq. 2, 3, 5）

$$\varepsilon = \sqrt{\frac{8\delta\gamma}{\pi^2}}, \quad w = \frac{4\gamma}{\delta}, \quad m = \frac{\pi^2 \beta}{8\delta}$$

- $\delta$：拡散界面幅（`delta_factor × dx`）
- $\gamma$：界面エネルギー $[\text{J/m}^2]$
- $\beta$：attachment kinetic coefficient $[\text{m}^4/(\text{J·s})]$

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

$$= \varepsilon_0^2 \sum_{p=x,y} \left\{ \underbrace{\frac{\partial a}{\partial p}\frac{\partial a}{\partial \phi_p}|\nabla\phi|^2}_{\text{(I)}} + \underbrace{a\frac{\partial}{\partial p}\!\left(\frac{\partial a}{\partial \phi_p}\right)|\nabla\phi|^2}_{\text{(II)}} + \underbrace{a\frac{\partial a}{\partial \phi_p}\frac{\partial}{\partial p}\!\left(|\nabla\phi|^2\right)}_{\text{(III)}} \right\}$$

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
GPU-multi-phase-field-model-solification_2d.ipynb  # メインノートブック
config.yaml                                         # シミュレーションパラメータ
requirement.txt                                     # Python依存ライブラリ
result/                                             # 出力（自動生成）
```

### ノートブックのセル構成

| セル | ID | 内容 |
|---|---|---|
| 0 | `4b7a9ff0` | タイトル・説明（Markdown） |
| 1 | `8dd6626f` | **設定読み込み**（config.yaml → Python変数） |
| 2 | `81ab8d11` | **結晶方位の設定**（四元数・{111}法線の事前計算） |
| 3 | `6f545928` | **APT 配列と相互作用行列の初期化** |
| 4 | `4a55110e` | **界面パラメータの変換関数と行列充填** |
| 5 | `4f450a44` | **CUDAデバイス関数**：異方性関数 $a(c)$、$b(\theta)$ |
| 6 | `ec9e934e` | **CUDAデバイス関数**：最近接 {111} との cos 計算 |
| 7 | `509e58d1` | **CUDAデバイス関数**：境界条件インデックス、勾配 |
| 8 | `dc650256` | **CUDAデバイス関数**：セルごとの $\varepsilon^2$ 計算 |
| 9 | `e6698b93` | **CUDAデバイス関数**：異方性拡散項（第1項） |
| 10 | `e492311c` | **CUDAデバイス関数**：2階微分、トルク項 (A11–A15) |
| 11 | `b1f44282` | **CUDAカーネル**：`kernel_update_nfmf`（APT配列更新） |
| 12 | `cab2615b` | **CUDAカーネル**：`kernel_update_phasefield_active`（メイン更新） |
| 13 | `95e767b6` | **CUDAカーネル**：`kernel_update_temp`（温度更新） |
| 14 | `134d0c0d` | **初期化**（$\phi$, $T$ の初期値設定、初期図の出力） |
| 15 | `c2b0947c` | **GPU転送**（CPU→GPU、グリッド計算） |
| 16 | `4903b694` | **メインループ**（時間発展・可視化・保存） |

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
| `matplotlib` | 可視化・PNG 保存 |
| `pillow` | シード画像の読み込み（`image_line` モード） |
| `pyyaml` | `config.yaml` の読み込み |
| `ipykernel`, `jupyter` | Jupyter Notebook 実行環境 |

### CUDA の確認

```python
from numba import cuda
print(cuda.gpus)         # 認識されている GPU 一覧
cuda.detect()            # 詳細情報
```

---

## 設定ファイル（config.yaml）

すべての物理・数値パラメータをここで管理します。
ノートブックを直接編集せずに条件を変えられます。

```yaml
# =====================================================
# Multi-Phase Field Solidification - Configuration
# =====================================================

grid:
  nx: 256           # x方向グリッド数
  ny: 256           # y方向グリッド数
  dx: 1.0e-4        # 格子間隔 x [m]
  dy: 1.0e-4        # 格子間隔 y [m]（現在 dx = dy を前提）
  dt: 1.0e-4        # 時間刻み [s]
  nsteps: 20000     # 総ステップ数

physical:
  T_melt: 1687      # 融点 [K]（シリコンの場合）
  G: 1.0e+2         # 温度勾配 [K/m]（初期温度分布に使用）
  V_pulling: 3.5e-2 # 引き抜き速度 [m/s]（冷却速度 = G × V_pulling）
  Sf: 2.12e+4       # 融解エントロピー [J/(m³·K)]

interface:
  delta_factor: 6.0   # 界面厚さ = delta_factor × dx
  gamma_100: 0.44     # 固液界面エネルギー（(100)基準）[J/m²]
  gamma_GB: 0.60      # 粒界エネルギー（等方性）[J/m²]

anisotropy:
  a0_deg: 54.7        # 基準角 α₀ [degrees]
  delta_a: 0.36       # 異方性強度 δₐ
  mu_a: 0.6156        # 異方性プリファクター μₐ
  p_round: 0.05       # 角の丸め係数（0に近いほどシャープなファセット）
  ksi: 0.30           # 運動学的異方性の最小値 ξ
  omg_deg: 10.0       # 運動学的異方性の角度スケール ω [degrees]

mobility:
  M_SL: 5.0e-5        # 固液間モビリティ [m/(Pa·s)]
  M_GB_ratio: 0.05    # M_GB = M_SL × M_GB_ratio（粒界のモビリティ比）

gpu:
  MAX_GRAINS: 20      # 相（グレイン+液相）の最大数（カーネルのバッファサイズ）
  KMAX: 18            # 1セル内のアクティブ相数の上限（ローカル配列サイズ）
  threads_per_block: [16, 16]  # CUDAスレッドブロックサイズ

seed:
  mode: "image_line"  # "random_110" または "image_line"
  number_of_grain_fallback: 17  # random_110 モード時の固相グレイン数
  random_seed: 42               # random_110 モード時の乱数シード
  image_path: "path/to/seed.bmp"  # image_line モード時の BMP 画像パス
  line_axis: "horizontal"         # 断面の向き："horizontal" または "vertical"
  line_index: null                # 断面位置（null = 中央）
  color_tolerance: 0              # 色の同一判定許容差（0〜255）
  height: 32                      # 初期固相の高さ [グリッド点数]

output:
  dir_template: "result/ver6-5/{M_GB_ratio:.2f}/{cooling_rate:.1e}"
  save_every: 200   # 何ステップごとに PNG を保存するか
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
| `MAX_GRAINS` | `nf`/`mf` 配列の最大次元。`image_line` で検出された粒数+1 以上が必要 |
| `KMAX` | カーネル内ローカル配列サイズ。`MAX_GRAINS` 以上であれば安全 |
| `threads_per_block` | CUDA スレッドブロック。通常 `[16, 16]` または `[32, 8]` |

> **注意**：`MAX_GRAINS` と `KMAX` を変えた場合は、カーネルを **再コンパイル**（ノートブックを最初から再実行）する必要があります。

#### `seed`

詳細は [シードの設定](#シード初期結晶核の設定) セクションを参照。

#### `output`

`dir_template` には以下の変数が使えます：

| プレースホルダー | 意味 |
|---|---|
| `{M_GB_ratio:.2f}` | 粒界モビリティ比（小数2桁） |
| `{cooling_rate:.1e}` | 冷却速度 $G \cdot V$ [K/s] （指数表記） |

---

## 実行方法

### 1. config.yaml を編集

```yaml
seed:
  mode: "random_110"    # 外部画像不要で動作確認できる
  number_of_grain_fallback: 10
```

### 2. Jupyter Notebook を起動

```bash
# 仮想環境を有効化した状態で
jupyter notebook
```

`GPU-multi-phase-field-model-solification_2d.ipynb` を開き、**上から順にセルを実行**します。

> **重要**：セルを飛ばして実行すると変数未定義エラーになります。
> 必ず Cell 1（設定読み込み）から順番に実行してください。

### 3. 実行順序の確認

```
Cell 1  → 設定読み込み
Cell 2  → 結晶方位の計算
Cell 3  → APT配列・相互作用行列の初期化
Cell 4  → 界面パラメータ変換
Cell 5-10 → CUDAデバイス関数の定義（JITコンパイルはこの時点では未実行）
Cell 11-13 → CUDAカーネルの定義
Cell 14 → 初期φ・T の設定と初期図の描画
Cell 15 → GPU へのデータ転送
Cell 16 → メインループ実行（最も時間がかかる）
```

### 4. カーネル再コンパイルが必要な場合

以下を変更したときは **Cell 1 から全セルを再実行**してください：
- `KMAX`, `MAX_GRAINS`（ローカル配列サイズが変わる）
- `LIQ`（定数として焼き込まれている箇所がある）
- デバイス関数の引数・ロジック変更

---

## 出力

### 保存場所

`config.yaml` の `output.dir_template` で指定したディレクトリに PNG が保存されます。

デフォルト例：`result/ver6-5/0.05/3.5e+00/`
- `0.05` → `M_GB_ratio = 0.05`
- `3.5e+00` → 冷却速度 $G \cdot V = 100 \times 0.035 = 3.5$ K/s

### 保存される画像

- `step_0.png`：初期状態のグレインマップ
- `step_200.png`, `step_400.png`, ...：`save_every` ステップごとのグレインマップ

### 画像の見方

- **カラーマップ**：`tab20`（各色 = 各グレイン ID）
- ID 0（濃い青）= 液相、ID 1 以上 = 各固相グレイン
- 横軸 = x 方向（0〜nx-1）、縦軸 = y 方向（origin="lower"、下が y=0）
- y=0 側が初期固相（下から成長）

---

## シード（初期結晶核）の設定

### モード 1：`random_110`

ランダムな方位を持つグレインを x 方向に均等に並べます。

```yaml
seed:
  mode: "random_110"
  number_of_grain_fallback: 17   # グレイン数（液相含まず）
  random_seed: 42                 # 乱数シード（再現性のため）
  height: 32                      # 初期固相の高さ [グリッド点]
```

**方位の決め方**（`random_110` モード）：

1. $\langle 110 \rangle$ 方向（$[1, 1, 0]$ 単位ベクトル）を面外軸として設定
2. グローバル z 軸から $\langle 110 \rangle$ への回転 $R_\text{align}$ を計算
3. 各グレインに対してランダムな角度 $\theta \in [0, 2\pi)$ で $\langle 110 \rangle$ 周りに回転 $R_\text{twist}$
4. 合成回転 $R = R_\text{twist} \cdot R_\text{align}$ を四元数で保存

この設定は「面内で任意に回転した結晶が、[110] 方向を面外軸として持つ」2D 設定を再現します。

### モード 2：`image_line`

BMP 画像の 1 ライン（断面）から各ピクセルの RGB 値 → 四元数 → 結晶方位を読み取ります。
論文の計算で生成した 3D 組織の断面を 2D シードとして使う場合に有効です。

```yaml
seed:
  mode: "image_line"
  image_path: "path/to/seed.bmp"   # RGB エンコードされた方位画像
  line_axis: "horizontal"           # "horizontal"：水平ライン、"vertical"：垂直ライン
  line_index: null                  # null = 中央のライン
  color_tolerance: 0                # 隣接ピクセルが「同一グレイン」とみなす色差許容値
  height: 32                        # 初期固相の高さ
```

**RGB → 四元数の変換**（`rgb_to_unit_quaternion`）：

$$v_k = \frac{R_k}{255} \cdot 2 - 1 \quad (k = R, G, B)$$

$$w = \sqrt{\max(1 - |v|^2, 0)}, \quad q = (v_x, v_y, v_z, w) / |q|$$

これにより RGB 値が単位四元数（SciPy の `(x, y, z, w)` 形式）として解釈されます。

**グレイン分割**（`segment_seed_line`）：

隣接ピクセルの RGB 差が `color_tolerance` を超えたら別グレインとして分割します。
`color_tolerance = 0` はアンチエイリアスなしの完全分割。
アンチエイリアス画像の場合は `color_tolerance = 5〜20` 程度に設定します。

> `image_line` モードで検出されたグレイン数+1（液相）が `MAX_GRAINS` を超えるとエラーになります。

---

## 各セルの詳細説明

### Cell 2：結晶方位と {111} 法線の事前計算

```python
grain_quaternions[gid]  # shape: (N, 4), SciPy (x,y,z,w) 形式
grain_n111[gid]         # shape: (N, 8, 3), 各グレインの8つの{111}方向（グローバル座標）
```

`grain_n111` は毎ステップカーネル内で使われます。
事前計算することで GPU カーネルの演算量を削減しています。

8つの {111} 方向：$\frac{1}{\sqrt{3}}(\pm1, \pm1, \pm1)$ を回転行列で変換したもの。
2D シミュレーションでは $(n_x, n_y)$ 成分のみ使用します。

### Cell 4：界面パラメータ変換

相互作用行列の充填：

| 行列 | 粒界 (solid-solid) | 固液界面 (solid-liquid) |
|---|---|---|
| `wij[i,j]` | $W_\text{GB} = 4\gamma_\text{GB}/\delta$ | $W_0^\text{SL} = 4\gamma_{100}/\delta$（基準値） |
| `aij[i,j]` | $\varepsilon_\text{GB}$ | $\varepsilon_0^\text{SL}$（基準値）|
| `mij[i,j]` | $(\pi^2/8\delta) \cdot M_\text{GB}$ | $(\pi^2/8\delta) \cdot M_\text{SL}$ |

> 固液界面の $\varepsilon$ と $W$ は **異方性により実行時に上書き** されます（`aij`/`wij` に格納された値はデフォルト値）。

### Cell 5：デバイス関数 `calc_a_from_cos` / `calc_b_from_cos`

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

### Cell 7：境界条件インデックス

```python
idx_xp(l, nx)  # l+1（端では 0 へラップ）：周期境界
idx_xm(l, nx)  # l-1（端では nx-1 へラップ）：周期境界
idx_yp(m, ny)  # m+1（端では ny-1：Neumann反射）
idx_ym(m, ny)  # m-1（端では 0：Neumann反射）
```

### Cell 9：`aniso_term1_solid`

固液界面の異方性 Laplacian 項（いわゆる「主要項」）：

$$\nabla \cdot (\varepsilon^2 \nabla \phi_0) \approx \frac{({\varepsilon^2_c + \varepsilon^2_{x+}})(\phi_{x+} - \phi_c) - ({\varepsilon^2_c + \varepsilon^2_{x-}})(\phi_c - \phi_{x-})}{2\Delta x^2} + (\text{y方向も同様})$$

セルごとに $\varepsilon^2 = \varepsilon_0^2 \cdot a^2(\cos\theta)$ を計算してから差分します。

### Cell 10：トルク項 `torque_A11`

異方性がある場合に生まれる付加項で、界面の向きを結晶の優先方向に引き付ける効果があります。
論文 Appendix A の式 (A11)〜(A15) に対応します。
計算には液相 $\phi_0$ の1・2階偏微分が必要です：

```python
phixx, phiyy, phixy = d2_phi_xy(phi, 0, l, m, nx, ny, dx)
```

### Cell 11：`kernel_update_nfmf`

毎ステップ呼び出し、各セルの「アクティブ相リスト」を更新します。

判定条件：
```
phi[i, l, m] > 0  OR  隣接セルのいずれかで phi[i, ...] > 0
```

後者の条件（隣接セルチェック）により、「次のステップで界面が広がる」相も事前にアクティブと判定します。

### Cell 12：`kernel_update_phasefield_active`（メインカーネル）

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
  V_pulling: 3.5e-2  # 速度は同じ → 冷却速度10倍
```

または

```yaml
physical:
  G: 1.0e+2
  V_pulling: 3.5e-1  # 速度を10倍
```

どちらも $\dot{T} = G \cdot V$ が変わります。

### グレイン数を変えたい（`random_110`）

```yaml
seed:
  mode: "random_110"
  number_of_grain_fallback: 30   # 30グレイン

gpu:
  MAX_GRAINS: 35    # グレイン数 + 液相(1) + 余裕
  KMAX: 30          # MAX_GRAINS に合わせる
```

> `MAX_GRAINS` を変えたら **カーネルを再コンパイル**（Cell 1 から再実行）。

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
  M_GB_ratio: 0.5    # デフォルト 0.05 → 10倍
```

---

## よくあるエラーと対処法

### `FileNotFoundError: seed image not found`

```yaml
seed:
  mode: "random_110"   # 画像不要なこちらに切り替え
```

または `image_path` を実際のファイルパスに修正してください。

### `ValueError: Detected phases=XX exceeds MAX_GRAINS`

`image_line` モードで検出されたグレイン数が多すぎます。

対処法：
1. `MAX_GRAINS` を増やす（カーネル再コンパイル必要）
2. `color_tolerance` を増やして隣接グレインを統合

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

Jupyter でカーネル定義セルだけを再実行しても numba の JIT キャッシュが残ることがあります。
**Kernel → Restart & Run All** で全セルを最初から実行してください。

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
