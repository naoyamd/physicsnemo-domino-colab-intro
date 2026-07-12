# 第三者データに関する表示

## ノートブック内のDrivAerML派生データ

`notebooks/domino_surface_only_colab.ipynb` には、`EmmiAI/DrivAerML_subsampled_10x` のrun 105、130、202から抽出した圧縮データが、通信不要のフォールバックとして含まれます。

- 派生データ提供元: https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x
- 原データ: https://huggingface.co/datasets/neashton/drivaerml
- 原論文: Ashton et al., *DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics*, arXiv:2408.11969
- ライセンス: CC BY-SA 4.0
- 加えた変更: run 105、130、202から、対応する表面位置・法線・面積・圧力を各4,096点、固定乱数シードで抽出し、圧縮後にBase64形式でノートブックへ埋め込みました。

このフォールバックは、公開データへの通信が利用できない場合でも教材用の短時間実行を再現できるようにするためのものです。リポジトリのコードに適用されるApache License 2.0の対象ではありません。
