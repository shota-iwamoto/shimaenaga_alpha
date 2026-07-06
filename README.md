# Shimaenaga

tier-3開発用

---

## 使用ライブラリ・著作権表示

Shimaenaga は以下のサードパーティライブラリを使用しています。

### ランタイム依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Copyright (c) 2005-2025, NumPy Developers |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause | Copyright (c) 2007-2026, The scikit-learn developers |

### ビルド時依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [pybind11](https://github.com/pybind/pybind11) | BSD 3-Clause | Copyright (c) 2016 Wenzel Jakob and others |

---

## アルゴリズム参照・著作権表示

Shimaenaga は以下のプロジェクトで発表されたアルゴリズムや手法を参考に実装しています。

### XGBoost

- **Project**: https://github.com/dmlc/xgboost
- **License**: Apache 2.0
- **Reference**: Tianqi Chen and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD 2016. https://arxiv.org/abs/1603.02754
- **参考にした手法**: L1/L2 正則化付き葉の重み計算式、ヘシアンベースの分割ゲイン、最小子ノード制約

### LightGBM

- **Project**: https://github.com/microsoft/LightGBM
- **License**: MIT
- **Reference**: Guolin Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- **参考にした手法**: ヒストグラムベース分割探索・差分トリック、Leaf-wise 木成長、GOSS（勾配ベースサンプリング）、EFB（排他的特徴量バンドリング）

### CatBoost

- **Project**: https://github.com/catboost/catboost
- **License**: Apache 2.0
- **Reference**: Liudmila Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features." NeurIPS 2018. https://arxiv.org/abs/1706.09516
- **参考にした手法**: Ordered Boosting（予測シフト低減）、Ordered Target Encoding（カテゴリカル特徴量）、対称木（Oblivious Tree）成長

### Gradient Boosting Machine

- **Reference**: Jerome H. Friedman. "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics, 2001. https://www.jstor.org/stable/2699986
- **参考にした手法**: 勾配ブースティング全体の理論的基盤(残差への逐次的な関数近似、加法モデル)

### Attention Is All You Need

- **Reference**: Ashish Vaswani et al. "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762
- **参考にした手法**: スケーリング付き内積注意(scaled dot-product attention, softmax(QK^T / √d))、マルチヘッド注意 — Tier-2 自己注意およびゲート機構の設計に対応

### LambdaMART

- **Reference**: Christopher J.C. Burges. "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report, 2010. https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- **参考にした手法**: ランキング目的関数における λ 勾配(NDCG 変化量に基づくペアワイズ勾配のスケーリング)

---

## ライセンス

MIT License. <br/>
詳細は [LICENSE](LICENSE) を参照してください。
サードパーティ(pybind11)の著作権表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
