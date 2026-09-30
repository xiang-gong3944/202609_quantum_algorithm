# Practical Trotter error estimation

このディレクトリは、arXiv:2606.30738 “Practical Estimation of Trotter Error for Hamiltonian Simulation” に対応する数値評価をまとめています。対象は開放境界の一様 Heisenberg 鎖

\[
H=\sum_{j=1}^{N-1}(X_jX_{j+1}+Y_jY_{j+1}+Z_jZ_{j+1})=A+B
\]

に対する Trotter 誤差評価だけを扱います。`A` は奇数結合、`B` は偶数結合からなるため、各層内の結合は可換です。

## Files

- `heisenberg_chain_bounds.ipynb`: 2次・4次 Suzuki 公式について、誤差対分割数と、要求分割数対系サイズを比較する Notebook
- `heisenberg_bounds.py`: Hamiltonian、積公式、Pauli 交換子、BCH、ED/DMRG、リソース逆算の実装
- `tests/test_heisenberg_bounds.py`: 小規模行列および DMRG の検証
- `results/`: Notebook が実行時に自動作成し、CSV を出力するディレクトリ
- `figures/`: Notebook が実行時に自動作成し、図を出力するディレクトリ

## Compared quantities

Notebook は次の3系列を比較します。

1. `naive_l1`: Hamiltonian fragment のノルム和だけを使う finite-\(r\) bound
2. `commutator`: nested commutator を使う finite-\(r\) bound
3. `paper_gs`: leading BCH error operator の基底状態期待値を使う長時間漸近 estimate

`paper_gs` は finite-\(r\) の rigorous bound ではなく、他の2系列とは metric scope が異なります。\(N\le10\) では、Hamiltonian の縮退固有空間を射影演算子として扱った `paper_all_state` も診断用に計算します。

`epsilon` vs `r` の図は scope を混同しないよう2段に分けています。上段は ground-state vector error と `paper_gs`、下段は worst-case spectral error と `paper_all_state` を対応させます。naive/commutator は全状態に有効なので両段に表示しますが、ground state に特化した上界ではありません。paper線の薄青帯は inverse-Liouvillian 上界を使った finite-time dephasing 補正範囲、青点線は有限時間平均を直接評価した補正値です。

## Finite-time dephasing correction

論文の $D_H(E_p)$ 置換は長時間極限なので、Notebook は有限時間に残る

\[
Q_t(E_p)=\frac1t\int_0^t e^{isH}E_pe^{-isH}\,ds-D_H(E_p)
\]

も ED で評価します。`heisenberg_dephasing_vs_t.png` は $N=10$ での $1/t$ envelope、`heisenberg_dephasing_vs_n.png` は $t=N$ でのサイズ依存性とエネルギーgapを示します。`heisenberg_dephasing.csv` にGS・worst-caseの残差、inverse-Liouvillian bound、minimum-gap bound、小規模の非線形検証を保存します。`r` vs `N` の図にも同じ補正を連続値 $r_*$ の帯として表示しますが、EDで直接評価できる $N\le10$ で止め、大規模側へは外挿しません。

この補正は leading BCH operator 内の有限時間誤差です。高次BCH剰余は含まないため、追加後もpaper系列を完全なfinite-$r$保証とは扱いません。

## Run

プロジェクトルートで依存関係をインストールした後、次を実行します。

```bash
cd practical_trotter_error_estimation
jupyter notebook heisenberg_chain_bounds.ipynb
```

Notebook の既定値は \(N\le40\), \(t=N\), \(\epsilon=10^{-3}\) です。`RUN_LARGE_SYSTEMS=True` にすると \(N=50,100,200\) が追加されます。`RUN_NONLINEAR_DEPHASING=False` にすると、$N\le8$ の指数関数による非線形dephasing検証を省略できます。大規模点では quimb の DMRG/MPO を使うため、計算時間とメモリ使用量が増えます。

テストは次のコマンドで実行できます。

```bash
python -m pytest practical_trotter_error_estimation/tests
```

高速な検証だけを行う場合は `-m "not slow"` を付けてください。
