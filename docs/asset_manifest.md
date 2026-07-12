# 発表用アセット一覧

## 公開リポジトリに置く図表

| ファイル | 内容 | 想定スライド | 備考 |
|---|---|---:|---|
| `domino_surface_pressure_comparison.png` | case 43/63の正解・予測・絶対誤差 | 1, 15, 16 | Colab実出力、3次元の主図 |
| `domino_training_curve.png` | 120分の学習曲線 | 13 | 横軸時間を中心に読む |
| `domino_surface_metrics.csv` | 未学習2形状の評価値 | 14, 17 | 数値の一次資料 |
| `domino_metrics_summary.png` | RelL2と相関の要約 | 14 | スライド用に追加生成 |
| `domino_local_stencil.png` | Colabが出した全体ステンシル図 | 補足 | 4点が重なるため主図には使わない |
| `domino_local_stencil_zoom.png` | query点と近傍3点の拡大図 | 8, 9 | スライド用に追加生成 |

配置先：`docs/assets/run_20260712/`

## 結果の根拠

- 実行条件と解釈：`docs/results/run_20260712.md`
- 機械可読な条件：`docs/assets/run_20260712/run_metadata.json`
- Colabノートブック：`notebooks/domino_surface_only_colab.ipynb`
- 実行ソース：commit `d89f0a7`

## Gitに置かないもの

| 対象 | 理由 | ローカル配置 |
|---|---|---|
| `domino_surface_full_g24.pt` | 学習済み重み。教材公開に必須でなく、今後大容量化する | `artifacts/run_20260712/checkpoints/` |
| `domino_surface_predictions.npz` | 8,192点×2形状の予測点群。図の再生成用 | `artifacts/run_20260712/outputs/` |
| `domino_surface_run_20260712.zip` | 上記を含むColabアーカイブ | `artifacts/` |
| フルデータセット | ライセンス・容量・再取得性 | Hugging Face等の配布元 |

## 外部出典候補

- [NVIDIA PhysicsNeMo DoMINo公式手順](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html)
- [DoMINo論文](https://arxiv.org/abs/2501.13350)
- [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml)
- [Colab用の表面派生データ](https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x)

公式図を引用する場合は、スライド作成時にライセンス・クレジット表記・最新版との一致を再確認する。引用せず自作図にする場合も、概念の根拠として論文と公式資料を付記する。

## スライド作成前チェック

- [ ] 発表時間と1枚あたりの説明量を確定
- [ ] 研究所のテンプレート、ロゴ、機密区分を確認
- [ ] 圧力単位と正規化／逆正規化の説明を脚注化
- [ ] 3分実行と120分実行の比較条件に注意書きを付ける
- [ ] 画像内は英語・ASCII、本文は日本語になっていることを確認
- [ ] GitHubとColabのリンクを最終確認
