# 202609_quantum_algorithm

量子アルゴリズム、とくに Hamiltonian simulation に関する研究ノート、数値実験、可視化をまとめたリポジトリです。

## ディレクトリ構成

```text
.
├── note/                                  # LaTeX 研究ノート
├── practical_trotter_error_estimation/   # 実用的な Trotter 誤差推定
├── randomized_hamiltonian_simulation/    # ランダム化 Hamiltonian simulation
└── references/                            # ローカル参照資料（Git 管理外）
```

### `note/`

確率混合による Hamiltonian simulation の LaTeX ノートです。本文は `main.tex`、参考文献情報は `references.bib` にあります。

### `practical_trotter_error_estimation/`

arXiv:2606.30738 “Practical Estimation of Trotter Error for Hamiltonian Simulation” に対応する数値評価です。開放境界・一様 Heisenberg 鎖の even–odd 分割について、naive $L_1$ bound、nested-commutator bound、leading-BCH/diagonal estimate、および有限時間dephasing補正を比較します。入口は `practical_trotter_error_estimation/heisenberg_chain_bounds.ipynb` です。実行時に同ディレクトリ内の `results/` と `figures/` が自動作成されます。詳細は `practical_trotter_error_estimation/README.md` を参照してください。

### `randomized_hamiltonian_simulation/`

Childs–Ostrander–Su “Faster quantum simulation by randomization” と Campbell “Random Compiler for Fast Hamiltonian Simulation” に対応する、積公式と確率的 Hamiltonian simulation の数値モジュール・可視化です。

- `randomized_trotter_qdrift_comparison.ipynb`: randomized Lie–Trotter と qDRIFT の誤差スケーリング、および Bloch 球＋局所誤差拡大アニメーション
- `randomized_trotter.py`, `qdrift.py`: 各手法の状態発展・厳密平均チャネル・回路サンプリング
- `quantum_utils.py`: Pauli 行列、密度行列、Bloch 変換などの共通数値処理
- `higher_order_lie_trotter_animation.ipynb`: Strang 分割と高次 Lie–Trotter 分解の可視化

### `references/`

参照用の元論文をローカルで保管します。著作権上の理由から `references/papers/` は Git 管理対象外です。

## Python 環境

プロジェクトルートで仮想環境を作成し、依存パッケージをインストールします。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

数値実験を開く場合は、次のように `practical_trotter_error_estimation/` を起点に Jupyter Notebook を起動します。既定実行は $N\le40$ で、$N>10$ の基底状態には DMRG/MPO を利用します。

```bash
cd practical_trotter_error_estimation
jupyter notebook heisenberg_chain_bounds.ipynb
```

数値実装のテストはプロジェクトルートから実行できます。

```bash
python -m pytest practical_trotter_error_estimation/tests
```

可視化 Notebook は次のコマンドで開けます。

```bash
cd randomized_hamiltonian_simulation
jupyter notebook randomized_trotter_qdrift_comparison.ipynb
```

可視化用の数値モジュールは次のコマンドで検証できます。

```bash
python -m pytest randomized_hamiltonian_simulation/tests
```

## LaTeX ノートのビルド

TeX Live または MacTeX と `latexmk`、`uplatex`、`dvipdfmx`、`upbibtex` が必要です。

```bash
cd note
latexmk main.tex
```

生成されたPDFは `note/out/main.pdf` に出力されます。`note/out/` はGit管理対象外です。
