# PhysicsNeMo DoMINo Colab Intro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_surface_only_colab.ipynb)

PhysicsNeMoの公式 `DoMINO` クラスを、Google Colab上で小規模に学習・評価する入門教材です。DrivAerML由来の3次元車両表面点群から表面圧力を予測し、正解・予測・誤差を同じ車体上で可視化します。

## この教材で行うこと

- PhysicsNeMo 2.1.1の実 `DoMINO` クラスを使用
- 表面のみを対象とし、圧力1変数を予測する縮小学習
- 大域形状符号化、局所符号化、表面ステンシルを通した順伝播・逆伝播
- 実時間による学習制御、Google Driveへのチェックポイント保存、途中再開
- 未学習形状に対する相対L2誤差、面積重み付き相対L2誤差、MAE、RMSE
- 3次元の表面圧力比較、学習曲線、局所ステンシル図の出力

これはCFDソルバーでも、フルスケールDoMINoの再現でもありません。学習効果を優先した小規模なサロゲートモデル実験です。

## 使い方

1. 上のColabバッジからノートブックを開き、GPUランタイムを選択します。
2. 最初は `RUN_MODE="fallback"`、`TRAIN_MINUTES=3` で全セルを実行します（通信不要の短時間モード）。
3. 短い順伝播・逆伝播テストと3種類の図が出ることを確認します。
4. 本実験では `RUN_MODE="full"`、`TRAIN_MINUTES=120` に変更します。
5. Google Driveを有効にすると、前処理結果とチェックポイントを再利用できます。

`fallback` はrun 105、130、202から抽出した実際の表面データをノート内に保持しています。合成圧力は使用しません。`full` はHugging Faceから表面位置・法線・面積・圧力だけを取得し、巨大な体積場は取得しません。

## 主な出力

- `domino_surface_pressure_comparison.png`
- `domino_training_curve.png`
- `domino_local_stencil.png`
- `domino_surface_metrics.csv`
- `domino_surface_predictions.npz`
- 再開可能なチェックポイントと実行条件ファイル

Colab標準環境での文字化けを避けるため、PNG内のタイトル・軸ラベル・凡例は英語表記にしています。ノートブック本文と説明は日本語です。

## 実行済み教材結果

2026年7月12日に、Tesla T4上で120分間の学習を完了しました。発表準備に使う軽量な図表と結果の解釈は、次の資料に整理しています。

- [実行結果と解釈](docs/results/run_20260712.md)
- [発表スライド20枚の構成案](docs/slide_plan.md)
- [図版・成果物一覧](docs/asset_manifest.md)
- [発表用の公開図表](docs/assets/run_20260712/)

学習済みチェックポイントと予測点群のNPZは容量と再利用方針のためGitには置かず、ローカルの `artifacts/` または外部ストレージで管理します。公開リポジトリには再現手順、数値、軽量な図表だけを収録します。

## ローカル検査

```bash
python scripts/check_domino_surface_colab.py
python scripts/smoke_domino_surface_inputs.py
```

2つ目の検査はPhysicsNeMoを導入した環境で、埋め込み実データを使った縮小DoMINoの順伝播・逆伝播を実行します。CUDAがない環境ではWarpの警告が出ますが、CPU上の短時間検査として動作します。

## 参考資料

- [NVIDIA PhysicsNeMo DoMINo公式手順](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html)
- [DoMINo論文](https://arxiv.org/abs/2501.13350)
- [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml)
- [Colab用の表面派生データ](https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x)

## ライセンス

コードとドキュメントはApache License 2.0です。ノートブック内に埋め込まれたDrivAerML派生データはCC BY-SA 4.0です。詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
