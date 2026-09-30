# Randomized Hamiltonian simulation

このディレクトリは、次の2本の論文で扱われるランダム化 Hamiltonian simulation を数値的に比較・可視化します。

- Childs, Ostrander, and Su, “Faster quantum simulation by randomization”
- Campbell, “Random Compiler for Fast Hamiltonian Simulation”

## Files

- `randomized_trotter_qdrift_comparison.ipynb`: randomized Lie–Trotter と qDRIFT の誤差スケーリングとアニメーション
- `randomized_trotter.py`: randomized Lie–Trotter の状態発展と平均チャネル
- `qdrift.py`: qDRIFT の状態発展と平均チャネル
- `quantum_utils.py`: Pauli 行列、密度行列、Bloch 変換などの共通処理
- `higher_order_lie_trotter_animation.ipynb`: Strang 分割と高次 Lie–Trotter 分解の補助的な可視化

## Run

プロジェクトルートで依存関係をインストールした後、次を実行します。

```bash
cd randomized_hamiltonian_simulation
jupyter notebook randomized_trotter_qdrift_comparison.ipynb
```

テストはプロジェクトルートから実行します。

```bash
python -m pytest randomized_hamiltonian_simulation/tests
```
