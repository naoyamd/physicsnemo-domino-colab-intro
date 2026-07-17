# Claude Code向けDoMINO検証手順

このリポジトリで「DoMINOを検証して」と依頼された場合は、まず短時間の `quick check` を使う。120分教材は、明示的に学習結果の再現まで求められた場合だけ使う。

## 1. ローカル静的検査

追加パッケージを導入せず、次を実行する。

```bash
python3 scripts/check_domino_quickcheck_colab.py
```

これはNotebookの構文、埋め込み実データ、非対話性、合格判定コードを検査する。`OK:` が出ても、DoMINOのランタイム検証が済んだとは報告しない。

## 2. Colabランタイム検査

次を開き、Notebookを編集せずに `ランタイム` → `すべてのセルを実行` を行う。

https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_quickcheck_colab.ipynb

- ブラウザ操作が使える場合は、Colabで実行して最終セルを読む。
- ブラウザ操作が使えない場合は、上のURLをユーザーに開いてもらい、最終セルの `DOMINO_QUICKCHECK_RESULT=...` から後を共有してもらう。
- Google Drive、Hugging Face、APIキー、GPU指定は不要。CPUとGPUのどちらでもよい。

合格条件は、次のすべてを満たすこと。

1. `DOMINO_QUICKCHECK_RESULT` の `status` が `pass`
2. `checks` の全項目が `true`
3. 最終行が `DOMINO_QUICKCHECK: PASS`

途中のグラフや損失値の大小を精度評価として扱わない。`PASS` がない場合は、最初のtracebackが失敗根拠である。

## 3. 報告形式

必ず次を分けて報告する。

```text
Static check: PASS | FAIL
Colab runtime: PASS | FAIL | NOT RUN
Device: cpu | cuda | unknown
Evidence: 最終マーカー、または最初のエラー
Scope: forward/backward/1 optimizer stepの疎通確認。予測精度は未評価。
```

## Notebookを変更する場合

`notebooks/domino_quickcheck_colab.ipynb` は生成物。`scripts/build_domino_quickcheck_colab.py` を編集して再生成し、その後に静的検査を実行する。

```bash
python3 scripts/build_domino_quickcheck_colab.py
python3 scripts/check_domino_quickcheck_colab.py
```
