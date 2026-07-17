# LLMとGoogle ColabでDoMINOをすぐに検証する

## 最短フロー

1. [DoMINO Quick CheckをColabで開く](https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_quickcheck_colab.ipynb)。
2. `ランタイム` → `すべてのセルを実行` を選ぶ。設定変更や認証は行わない。
3. 最終セルの `DOMINO_QUICKCHECK_RESULT=...` と `DOMINO_QUICKCHECK: PASS` をLLMに渡す。

初回はPhysicsNeMoのインストールに数分かかることがあります。計算部分は、埋め込み実データ512点と小型モデルを使う1 optimizer stepだけです。GPUは任意で、CPUでも実行できます。

## Claude Codeへ渡すプロンプト

リポジトリをClaude Codeで開いた後、次をそのまま依頼できます。ルートの `CLAUDE.md` に、判定条件と報告形式が記載されています。

```text
このリポジトリのCLAUDE.mdに従って、DoMINO quick checkを検証してください。
まずローカルの静的検査を実行し、その後Colabのquick-check Notebookを編集せずに全セル実行してください。
ブラウザ操作ができない場合は、私が行う操作と、共有すべき最終セル出力を明示してください。
静的検査とColabランタイム検査を分けて報告し、予測精度を検証したとは表現しないでください。
```

LLMがブラウザを操作できない環境でも、必要な人手はColabを開いて全セル実行し、最終セルを貼り戻す操作だけです。

## 何を検証しているか

| 段階 | 合格の証拠 | 検証内容 |
|---|---|---|
| ローカル静的検査 | `check_domino_quickcheck_colab.py` の `OK:` | Notebook構文、実データpayload、非対話性、PASS契約 |
| Colabランタイム | JSONの `status: pass`、全 `checks: true`、最終行の `PASS` | 公式DoMINOの生成、forward、backward、有限勾配、パラメータ更新 |
| 120分教材 | 指標CSV、予測NPZ、図、完了manifest | 縮小学習と未学習形状評価 |

quick checkは実装経路の疎通確認です。1 step後の損失や図から予測性能を判断してはいけません。学習結果まで確認する場合は、quick check合格後に `notebooks/domino_surface_only_colab.ipynb` を使用します。

## PASS出力の読み方

最終セルは1行JSONと最終マーカーを出します。環境によって数値は変わりますが、判定に使うのは次の形です。

```text
DOMINO_QUICKCHECK_RESULT={"status":"pass",...,"checks":{"output_shape_ok":true,...}}
Result file: /content/domino_quickcheck/quickcheck_result.json
DOMINO_QUICKCHECK: PASS
```

`quickcheck_result.json` と確認用PNGは `/content/domino_quickcheck/` に保存されます。

## 失敗時

- `PASS` がない: 最初に表示されたtracebackを共有する。後続エラーは原因ではない場合があります。
- installセルで失敗: Colabのランタイムを再接続して全セルを再実行する。継続する場合は、Python・Torch・PhysicsNeMoの表示とinstallセルの全文を共有する。
- GPUがない: エラーではありません。quick checkは自動的にCPUへ切り替わります。
- Driveの認証を求められた: 120分教材を開いています。ファイル名が `domino_quickcheck_colab.ipynb` か確認します。
