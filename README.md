# PhysicsNeMo DoMINo Colab Intro

**数分の疎通確認（CPU可）**<br>
[![Open Quick Check In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_quickcheck_colab.ipynb)

**縮小学習・評価（GPU、標準120分）**<br>
[![Open Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_surface_only_colab.ipynb)

PhysicsNeMoの公式 `DoMINO` クラスを、Google Colab上で検証・学習・評価する入門教材です。数分のquick checkと、DrivAerML由来の3次元車両表面点群から表面圧力を学習する120分教材を分けています。

## 最短検証：このページのURLかCLAUDE.mdをLLMに読ませるだけ

Claude Code、Claude、またはWebページやファイルを読める同等のLLMに、次のどちらかを読ませて検証を依頼するだけです。事前の環境構築やパラメータ設定は不要です。

- このページのURL: https://github.com/naoyamd/physicsnemo-domino-colab-intro
- ルートの [`CLAUDE.md`](CLAUDE.md): URLを直接参照できないLLMには、ファイルを添付するか本文を貼り付けます。

依頼文は、次の一文だけで始められます。

> このリポジトリのDoMINO quick checkを実行し、CLAUDE.mdの形式で結果を報告してください。

ブラウザを操作できるLLMは、Colabを開いて全セルを実行し、合否まで報告します。ブラウザを操作できないLLMでも、必要なColabリンクと最小限の操作、共有すべき結果を案内します。その場合だけ、案内されたセルの実行と最終出力の共有が必要です。

quick checkは、Google Drive、Hugging Face、APIキー、手動パラメータ設定を使いません。GPUがあれば自動利用し、なければCPUで実行します。公式DoMINOに埋め込み実データを入力し、forward、backward、1 optimizer step、更新後推論までを機械判定します。

`CLAUDE.md` はClaude Code向けの規約名ですが、内容は通常のMarkdown手順書なので、他のLLMでも利用できます。詳しい手動手順は、[LLM + Colab quick start](docs/llm_colab_quickstart.md) を参照してください。

> quick checkは実装経路の疎通確認であり、予測精度、CFDの妥当性、フルスケールDoMINoの再現を保証しません。

## 120分教材で行うこと

- PhysicsNeMo 2.1.1の実 `DoMINO` クラスを使用
- 表面のみを対象とし、圧力1変数を予測する縮小学習
- 大域形状符号化、局所符号化、表面ステンシルを通した順伝播・逆伝播
- 実時間による学習制御、Google Driveへのチェックポイント保存、途中再開
- 未学習形状に対する相対L2誤差、面積重み付き相対L2誤差、MAE、RMSE
- 3次元の表面圧力比較、学習曲線、局所ステンシル図の出力

これはCFDソルバーでも、フルスケールDoMINoの再現でもありません。学習効果を優先した小規模なサロゲートモデル実験です。

## 120分教材の使い方

1. 上の「縮小学習・評価」バッジからノートブックを開き、GPUランタイムを選択します。
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
python scripts/check_domino_quickcheck_colab.py
python scripts/check_domino_surface_colab.py
python scripts/smoke_domino_surface_inputs.py
```

1つ目はPython標準ライブラリだけでquick-check Notebookの構文、埋め込みデータ、非対話性、合格判定を検査します。3つ目はPhysicsNeMoを導入した環境で、埋め込み実データを使った縮小DoMINoの順伝播・逆伝播を実行します。CUDAがない環境ではWarpの警告が出ますが、CPU上の短時間検査として動作します。

## 参考資料

- [NVIDIA PhysicsNeMo DoMINo公式手順](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html)
- [DoMINo論文](https://arxiv.org/abs/2501.13350)
- [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml)
- [Colab用の表面派生データ](https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x)

## ライセンス

コードとドキュメントはApache License 2.0です。ノートブック内に埋め込まれたDrivAerML派生データはCC BY-SA 4.0です。詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
