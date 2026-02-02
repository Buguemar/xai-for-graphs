# Explainable AI for Graph-Based Learning: A Survey Beyond Graph Neural Networks
*[Margarita Bugueño, Russa Biswas, Gerard de Melo]* 

📄 This repository accompanies our **updated and extended survey paper**, currently submitted to ACM TIST, 2026.

The goal of this repository is to provide a **living, structured, and reproducible companion** to the paper, including:
- A curated list of explainability methods
- A unified taxonomy beyond GNN-centric approaches
- Summary tables for reference to the community

---
🔗 An **earlier preprint version (2024)** of this work is publicly available on HAL Science:
[“Graph-Based Explainable AI: A Comprehensive Survey”](https://hal.science/hal-04660442/file/Graph_Based_Explainable_AI____HAL_version.pdf)

#### ⚠️ **Note:** The HAL preprint corresponds to an earlier version and does **not** include the latest revisions, extensions, and corrections present in the ACM submission.

Please cite the ACM version once available. Until then, the HAL preprint may be used for reference.

---

## 🔍 Scope of the Survey

This survey systematically reviews **graph-based explainability methods** that:
- Go **beyond standard Graph Neural Networks (GNNs)**
- Cover scoring, extraction, and generation-based **explanation modalities**
- Along gradients, decomposition, path-reasoning, data integration, surrogates, perturbation, and graph creation-based **explainability approaches**
- Address node-level, graph-level, and generative tasks
- Are evaluated using quantitative and/or qualitative protocols

---
## 🧠 Categorization Overview

We categorize scoring, extraction, and generation-based methods along the following dimensions:
- **Explainability Approach** (Gradients, Decomposition, Perturbation, etc.)
- **GNN dependency** (if the method targets GNN models only or not)
- **Task** (graph classification, node classification, link prediction, generative tasks, and others)
- **Data Type** (Real or Synthetic)
- **Validation of the method** (Quantitative / Qualitative)


A visual overview of our proposed categorization is available in [`images/categorization.pdf`](images/categorization.pdf).

---

## Scoring-Based Explainers

#### 📊 Comparison of Methods in This Category

| Method | Year | Approach | GNN-based | Graph-Classif. | Node-Classif. | Other Tasks | Data Type | Validation | Code |
|--------|------|----------|-----------|------|-------|-------------|-----------|------------|------|
| Integrated Gradients | 2017 | Gradients | ✓ | ✓ | – | – | Real | Ql | ✓ |
| BayesGrad | 2018 | Gradients | ✓ | ✓ | – | Regression | Real / Synthetic | Qn / Ql | ✓ |
| SA | 2019 | Gradients | ✓ | ✓ | ✓ | Regression | Real | Ql | ✓ |
| C-Gradients | 2019 | Gradients | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | – |
| Guided BP | 2019 | Gradients | ✓ | ✓ | ✓ | – | Real | Ql | ✓ |
| CAM | 2019 | Gradients | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | – |
| GradCAM | 2019 | Gradients | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | – |
| Excitation BP | 2019 | Decomposition | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | – |
| c-Excitation BP | 2019 | Decomposition | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | – |
| LRP | 2019 | Decomposition | ✓ | ✓ | ✓ | Regression | Real | Ql | ✓ |
| GNN-LRP | 2020 | Decomposition | ✓ | ✓ | – | – | Real / Synthetic | Qn | ✓ |
| GLRP | 2021 | Decomposition | ✓ | ✓ | – | – | Real | Ql | ✓ |
| GNES | 2021 | Gradients | ✓ | ✓ | – | – | Real | Qn / Ql | ✓ |
| SE-SGFormer | 2025 | Path Reasoning | ✗ | – | – | Link Sign | Real | Qn | ✓ |
| GF-LRP | 2025 | Decomposition | ✗ | – | – | Generative | Real | Qn / Ql | – |

**Legend:** ✓ = Supported, – = Not supported, Qn = Quantitative, Ql = Qualitative

#### :books: List of Papers in This Category

1. (**Integrated Gradients**) Sundararajan, M., Taly, A., & Yan, Q. (2017). **Axiomatic attribution for deep networks**. In International conference on machine learning (pp. 3319-3328). PMLR. [[Paper]](https://proceedings.mlr.press/v70/sundararajan17a.html) [[Source Code]](https://github.com/ankurtaly/Integrated-Gradients)
2. (**BayesGrad**) Akita, H., Nakago, K., Komatsu, T., Sugawara, Y., Maeda, S. I., Baba, Y., & Kashima, H. (2018). **Bayesgrad: Explaining predictions of graph convolutional networks**. In Neural Information Processing: 25th International Conference, ICONIP 2018, Siem Reap, Cambodia, December 13–16, 2018, Proceedings, Part V 25 (pp. 81-92). Springer International Publishing. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-04221-9_8) [[Source Code]](https://github.com/pfnet-research/bayesgrad)
3. (**SA**) [_Original_] Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., & Müller, K. R. (2010). **How to explain individual classification decisions**. The Journal of Machine Learning Research, 11, 1803-1831. --- [_Applied_] Baldassarre, F., & Azizpour, H. (2019). **Explainability techniques for graph convolutional networks**. arXiv preprint arXiv:1905.13686. [[Paper]](https://arxiv.org/abs/1905.13686) [[Source Code]](https://github.com/baldassarreFe/graph-network-explainability)
4. (**C-Gradients**) [_Original_] Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). **Deep inside convolutional networks: Visualising image classification models and saliency maps**. arXiv preprint arXiv:1312.6034. --- [_Applied_] Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H. (2019). **Explainability methods for graph convolutional neural networks**. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10772-10781). [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html)
5. (**Guided BP**) [_Original_] Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). **Striving for simplicity: The all convolutional net**. arXiv preprint arXiv:1412.6806. --- [_Applied_] Baldassarre, F., & Azizpour, H. (2019). **Explainability techniques for graph convolutional networks**. arXiv preprint arXiv:1905.13686. [[Paper]](https://arxiv.org/abs/1905.13686) [[Source Code]](https://github.com/baldassarreFe/graph-network-explainability)
6. (**CAM**) [_Original_] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). **Learning deep features for discriminative localization**. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2921-2929). --- [_Applied_] Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H. (2019). **Explainability methods for graph convolutional neural networks**. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10772-10781). [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html)
7. (**Grad-CAM**) [_Original_] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). **Grad-cam: Visual explanations from deep networks via gradient-based localization**. In Proceedings of the IEEE international conference on computer vision (pp. 618-626). --- [_Applied_] Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H. (2019). **Explainability methods for graph convolutional neural networks**. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10772-10781). [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html)
8. (**Excitation BP & c-Excitation BP**) Pope, P. E., Kolouri, S., Rostami, M., Martin, C. E., & Hoffmann, H. (2019). **Explainability methods for graph convolutional neural networks**. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10772-10781). [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html)
9. (**LRP**) [_Original_] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015). **On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation**. PloS one, 10(7), e0130140. --- [_Applied_] Baldassarre, F., & Azizpour, H. (2019). **Explainability techniques for graph convolutional networks**. arXiv preprint arXiv:1905.13686. [[Paper]](https://arxiv.org/abs/1905.13686) [[Source Code]](https://github.com/baldassarreFe/graph-network-explainability)
12. (**GNN-LRP**) Schnake, T., Eberle, O., Lederer, J., Nakajima, S., Schütt, K. T., Müller, K. R., & Montavon, G. (2021). **Higher-order explanations of graph neural networks via relevant walks**. IEEE transactions on pattern analysis and machine intelligence, 44(11), 7581-7596. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9547794) [[Source Code]](https://git.tu-berlin.de/thomas_schnake/paper_gnn_lrp)
13. (**GLRP**) Chereda, H., Bleckmann, A., Menck, K., Perera-Bel, J., Stegmaier, P., Auer, F., ... & Beißbarth, T. (2021). **Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer**. Genome medicine, 13, 1-16. [[Paper]](https://link.springer.com/article/10.1186/s13073-021-00845-7) [[Source Code]](https://gitlab.gwdg.de/UKEBpublic/graph-lrp)
14. (**GNES**) Gao, Y., Sun, T., Bhatt, R., Yu, D., Hong, S., & Zhao, L. (2021). **Gnes: Learning to explain graph neural networks**. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 131-140). IEEE. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9679041?casa_token=ACw635YeMXQAAAAA:KcfwQJWcyJKlFjN7m4k2yrzet6R6Gw85qmKXXWvVLX7YUF3DuDCxZoNKMADp6WcgGP_Hw0KNxePRmA) [[Source Code]](https://github.com/YuyangGao/GNES)
15. (**SE-SGFormer**) Li, L., Liu, J., Ji, X., Wang, M., & Zhang, Z. (2025). **Self-Explainable Graph Transformer for Link Sign Prediction**. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 11, pp. 12084-12092). [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33316) [[Source Code]](https://github.com/liule66/SE-SGformer)
16. (**GF-LRP**) Rodrigo-Bonet, E., & Deligiannis, N. (2024). **Gf-lrp: a method for explaining predictions made by variational graph auto-encoders**. IEEE Transactions on Emerging Topics in Computational Intelligence, 9(1), 281-291. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10586750?casa_token=6UAKIkOW4jYAAAAA:jMJ61403waEuG9Tczyo9n6vJYkkgee2sZjokQ-JN1oJFRtm9Wgt4ZjsIZeUVDBmvzk4Zb42IP0Krkw) 

## Extraction-Based Explainers

### Sequential Paths

#### 📊 Comparison of Methods in This Category

| Method | Year | Approach | GNN-based | Task | Data Type | Validation | Code |
|--------|------|----------|-----------|------|-----------|------------|------|
| Dedalo | 2014 | Path Reasoning | ✗ | Clustering | Real | Qn | – |
| UniWalk | 2017 | Data Integration | ✗ | Recommendation | Real | Ql | ✓ |
| RippleNet | 2018 | Path Reasoning | ✗ | Recommendation | Real | Ql | ✓ |
| CFKG | 2018 | Path Reasoning | ✗ | Recommendation | Real | Ql | – |
| EIUM | 2019 | Path Reasoning | ✗ | Recommendation | Real | Ql | – |
| DFGN | 2019 | Path Reasoning | ✗ | QA | Real | Ql | ✓ |
| KPRN | 2019 | Path Reasoning | ✗ | Recommendation | Real | Ql | – |
| PGPR | 2019 | Path Reasoning | ✗ | Recommendation | Real | Ql | ✓ |
| CAFE | 2020 | Path Reasoning | ✗ | Recommendation | Real | Ql | ✓ |
| DKER | 2020 | Path Reasoning | ✗ | Recommendation | Real | Ql | – |
| ELPE | 2020 | Path Reasoning | ✗ | KG Completion | Real | Ql | ✓ |
| LOGER | 2021 | Path Reasoning | ✗ | Recommendation | Real | Qn | ✓ |
| EXACTA | 2021 | Path Reasoning | ✗ | TCA | Real | Qn | – |
| Ekar | 2022 | Path Path Reasoning | ✗ | Recommendation | Real | Ql | – |
| PathReasoner | 2022 | Path Reasoning | ✗ | QA | Real | Ql | – |
| KR-GCN | 2023 | Path Reasoning | ✓ | Recommendation | Real | Qn / Ql | – |
| MES | 2024 | Path Reasoning | ✗ | Recommendation | Real | Qn | – |

**Legend:** ✓ = Supported, – = Not supported, Qn = Quantitative, Ql = Qualitative

#### :books: List of Papers in This Category

1. (**Dedalo**) Tiddi, I., d’Aquin, M., & Motta, E. (2014). **Dedalo: Looking for clusters explanations in a labyrinth of linked data**. In The Semantic Web: Trends and Challenges: 11th International Conference, ESWC 2014, Anissaras, Crete, Greece, May 25-29, 2014. Proceedings 11 (pp. 333-348). Springer International Publishing. [[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-319-07443-6_23.pdf) 
2. (**UniWalk**) Park, H., Jeon, H., Kim, J., Ahn, B., & Kang, U. (2017). **Uniwalk: Explainable and accurate recommendation for rating and network data**. arXiv preprint arXiv:1710.07134. [[Paper]](https://arxiv.org/pdf/1710.07134) [[Source Code]](http://datalab.snu.ac.kr/uniwalk)
3. (**RippleNet**) Wang, H., Zhang, F., Wang, J., Zhao, M., Li, W., Xie, X., & Guo, M. (2018). **Ripplenet: Propagating user preferences on the knowledge graph for recommender systems**. In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 417-426). [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3269206.3271739?casa_token=6Q0NQEyLmM0AAAAA:OlbvvUyWyxvObWMpHInzDOSeZQORVrV93gFkVz6f3Bg8n-UEeTOO3KyiEHopylqlIjJJJIcxXTddeA) [[Source Code]](https://github.com/hwwang55/RippleNet)
4. (**CFKG**) Ai, Q., Azizi, V., Chen, X., & Zhang, Y. (2018). **Learning heterogeneous knowledge base embeddings for explainable recommendation**. Algorithms, 11(9), 137. [[Paper]](https://www.mdpi.com/1999-4893/11/9/137) 
5. (**EIUM**) Huang, X., Fang, Q., Qian, S., Sang, J., Li, Y., & Xu, C. (2019). **Explainable interaction-driven user modeling over knowledge graph for sequential recommendation**. In proceedings of the 27th ACM international conference on multimedia (pp. 548-556). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3343031.3350893?casa_token=LZg9JDiqcsgAAAAA:NBwbPeFXXFSwvkt-7qCe2xj2boOMTyIMDSFKV9wA9-NdrY10PlCJzcIgw4_SrmOaRfCwaiWB1lzLlQ) 
6. (**DFGN**) Qiu, L., Xiao, Y., Qu, Y., Zhou, H., Li, L., Zhang, W., & Yu, Y. (2019). **Dynamically fused graph network for multi-hop reasoning**. In Proceedings of the 57th annual meeting of the association for computational linguistics (pp. 6140-6150). [[Paper]](https://aclanthology.org/P19-1617/) [[Source Code]](https://github.com/woshiyyya/DFGN-pytorch)
7. (**KPRN**) Wang, X., Wang, D., Xu, C., He, X., Cao, Y., & Chua, T. S. (2019). **Explainable reasoning over knowledge graphs for recommendation**. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 5329-5336). [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4470) 
8. (**PGPR**) Xian, Y., Fu, Z., Muthukrishnan, S., De Melo, G., & Zhang, Y. (2019). **Reinforcement knowledge graph reasoning for explainable recommendation**. In Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval (pp. 285-294). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3331184.3331203?casa_token=WqR7ejkZFEAAAAAA:uB9PFCgxGatxX1rVQd3XkVkJz5DztKDIEXBK22A6vbkmbZDJ5-nM9m_4k59PBSzrb3cFg1C56MKlCQ) [[Source Code]](https://github.com/orcax/PGPR)
9. (**CAFE**) Xian, Y., Fu, Z., Zhao, H., Ge, Y., Chen, X., Huang, Q., ... & Zhang, Y. (2020). **CAFE: Coarse-to-fine neural symbolic reasoning for explainable recommendation**. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 1645-1654). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3340531.3412038) [[Source Code]](https://github.com/orcax/CAFE)
10. (**DKER**) Zhang, Y., Xu, X., Zhou, H., & Zhang, Y. (2020, January). **Distilling structured knowledge into embeddings for explainable and accurate recommendation**. In Proceedings of the 13th international conference on web search and data mining (pp. 735-743). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3336191.3371790?casa_token=dVRnKRo16ZoAAAAA:FtLg34Fi1_nEPCkCh62rM0dQ42u31YBxvYAeAABFo1yQFtIot4r_8Bx71UAGEG78O9cZ_8MkGmi8ig) 
11. (**ELPE**) Bhowmik, R., & de Melo, G. (2020). **Explainable link prediction for emerging entities in knowledge graphs**. In The Semantic Web–ISWC 2020: 19th International Semantic Web Conference, Athens, Greece, November 2–6, 2020, Proceedings, Part I 19 (pp. 39-55). Springer International Publishing. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-62419-4_3) [[Source Code]](https://github.com/kingsaint/InductiveExplainableLinkPrediction)
12. (**LOGER**) Zhu, Y., Xian, Y., Fu, Z., De Melo, G., & Zhang, Y. (2021). **Faithfully explainable recommendation via neural logic reasoning**. arXiv preprint arXiv:2104.07869. [[Paper]](https://aclanthology.org/2021.naacl-main.245/) [[Source Code]](https://github.com/orcax/LOGER)
13. (**EXACTA**) Xian, Y., Zhao, H., Lee, T. Y., Kim, S., Rossi, R., Fu, Z., ... & Muthukrishnan, S. (2021, August). **Exacta: Explainable column annotation**. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (pp. 3775-3785). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467211?casa_token=SDYJmRsFFt4AAAAA:q8naNEniOd4_U8tOEl33arl82ZvZ2hsE9arKFBAmWsdQvoDzu_aFTOMpLbEHmBu2dEMI7cXT2q7PGw) 
14. (**Ekar**) Song, W., Duan, Z., Yang, Z., Zhu, H., Zhang, M., & Tang, J. (2019). **Ekar: an explainable method for knowledge aware recommendation**. arXiv preprint arXiv:1906.09506. [[Paper]](https://arxiv.org/abs/1906.09506) 
15. (**PathReasoner**) Zhan, X., Huang, Y., Dong, X., Cao, Q., & Liang, X. (2022). **PathReasoner: Explainable reasoning paths for commonsense question answering**. Knowledge-Based Systems, 235, 107612. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705121008741?casa_token=lseHhOtbjecAAAAA:-1TKMjxuE3L5LRppXWd7XOYNFJQPN4nNM4INXohw7Y575XPoNPnK6uOnuNTP3C4GRVJ1FT1hAAU) 
16. (**KR-GCN**) Ma, T., Huang, L., Lu, Q., & Hu, S. (2023). **Kr-gcn: Knowledge-aware reasoning with graph convolution network for explainable recommendation**. ACM Transactions on Information Systems, 41(1), 1-27. [[Paper]](https://dl.acm.org/doi/full/10.1145/3511019)
17. (**MES**) Tiwary, N., Noah, S. A. M., Fauzi, F., & Yee, T. S. (2024). **Max Explainability Score–A quantitative metric for explainability evaluation in knowledge graph-based recommendations**. Computers and Electrical Engineering, 116, 109190. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0045790624001186?casa_token=0Ls2vlNbZWoAAAAA:Xrlqo8dRUeh81O9h1hdRPhbZYxj9nCPwBENpbZj3a9iBfncvtf9BLmLIzl5Vqo0R6hYRvJk-Rm4)

### Logic Rules

#### 📊 Comparison of Methods in This Category
| Method | Year | Approach | Task | Data Type | Validation | Code |
|--------|------|----------|------|-----------|------------|------|
| LORE | 2018 | Surrogate | Classification | Real | Qn / Ql | – |
| TEM | 2018 | Surrogate | Recommendation | Real | Ql | ✓ |
| RuleRec | 2019 | Data Integration | Recommendation | Real | Ql | ✓ |
| ExCut | 2020 | Data Integration | Clustering | Real | Qn / Ql | ✓ |
| ExCaR | 2021 | Data Integration | Causal Reasoning | Real | Qn / Ql | ✓ |

**Legend:** ✓ = Supported, – = Not supported, Qn = Quantitative, Ql = Qualitative

#### :books: List of Papers in This Category

1. (**LORE**) Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018). **Local rule-based explanations of black box decision systems**. arXiv preprint arXiv:1805.10820. [[Paper]](https://arxiv.org/pdf/1805.10820)
2. (**TEM**) Wang, X., He, X., Feng, F., Nie, L., & Chua, T. S. (2018). **Tem: Tree-enhanced embedding model for explainable recommendation**. In Proceedings of the 2018 world wide web conference (pp. 1543-1552). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3178876.3186066) [[Source Code]](https://github.com/xiangwang1223/TEM)
3. (**RuleRec**) Ma, W., Zhang, M., Cao, Y., Jin, W., Wang, C., Liu, Y., ... & Ren, X. (2019). **Jointly learning explainable rules for recommendation with knowledge graph**. In The world wide web conference (pp. 1210-1221). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3308558.3313607?casa_token=q-0LyLtv6csAAAAA:WvAKtRDRQ9M3FZ1_oHlBR7neyraS7vhosFq-HSczBQiKt1Y-JMZjy6uqwSiIfFvwG-X70iAvud7R3Q) [[Source Code]](https://github.com/THUIR/RuleRec)
4. (**ExCut**) Gad-Elrab, M. H., Stepanova, D., Tran, T. K., Adel, H., & Weikum, G. (2020). **Excut: Explainable embedding-based clustering over knowledge graphs**. In International Semantic Web Conference (pp. 218-237). Cham: Springer International Publishing. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-62419-4_13) [[Source Code]](https://github.com/mhmgad/ExCut)
5. (**ExCAR**) Du, L., Ding, X., Xiong, K., Liu, T., & Qin, B. (2021). **Excar: Event graph knowledge enhanced explainable causal reasoning**. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 2354-2363). [[Paper]](https://aclanthology.org/2021.acl-long.183.pdf) [[Source Code]](https://github.com/sjcfr/ExCAR)

### Subgraph

#### 📊 Comparison of Methods in This Category
| Method | Year | Approach | GNN-based | Graph-Classif. | Node-Classif. | Other Tasks | Data Type | Validation | Code |
|--------|------|----------|-----------|------|-------|-------------|-----------|------------|------|
| ExplaiNE | 2019 | Perturbation | ✗ | – | – | Link Prediction | Real | Qn / Ql | – |
| GNNExplainer | 2019 | Perturbation | ✓ | ✓ | ✓ | Link Prediction | Real / Synthetic | Qn / Ql | ✓ |
| GraphMask | 2020 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn | ✓ |
| CoGE | 2020 | Data Integration | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | ✓ |
| PGExplainer | 2020 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| PGM-Explainer | 2020 | Surrogate | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn | ✓ |
| Causal Screening | 2020 | Perturbation | ✓ | ✓ | ✓ | – | Real | Qn / Ql | – |
| ReFine | 2021 | Perturbation | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | ✓ |
| Gem | 2021 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| MEG | 2021 | Perturbation | ✓ | ✓ | – | Regression | Real | Qn / Ql | – |
| RG-Explainer | 2021 | Graph Creation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | – |
| SE-GNN | 2021 | Data Integration | ✓ | – | ✓ | – | Real / Synthetic | Qn | ✓ |
| CMF | 2021 | Perturbation | ✓ | – | – | Forecasting | Real | Qn / Ql | – |
| CMGE | 2021 | Perturbation | ✓ | ✓ | ✓ | – | Real | Qn | ✓ |
| RCExplainer | 2021 | Surrogate | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn | – |
| RelEx | 2021 | Surrogate | ✗ | – | ✓ | – | Real / Synthetic | Qn / Ql | – |
| SubGraphX | 2021 | Perturbation | ✓ | ✓ | ✓ | Link Prediction | Real / Synthetic | Qn / Ql | ✓ |
| GraphSVX | 2021 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| GStarX | 2022 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| CF2 | 2022 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | – |
| ZORRO | 2022 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| TraP2 | 2022 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| GraphLime | 2022 | Surrogate | ✓ | – | ✓ | – | Real | Qn | – |
| MotifExplainer | 2022 | Data Integration | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | – |
| TAGE | 2022 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| EGIB | 2023 | Perturbation | ✓ | ✓ | ✓ | – | Real | Qn / Ql | – |
| GOAt | 2024 | Decomposition | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| STFExplainer | 2024 | Decomposition | ✓ | ✓ | – | Regression | Real / Synthetic | Qn | – |
| EAGX | 2024 | Data Integration | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | ✓ |
| Geb | 2024 | Perturbation | ✓ | – | ✓ | – | Real | Qn | – |
| K-FactExplainer | 2024 | Perturbation | ✓ | ✓ | ✓ | – | Real | Qn | – |
| XGExplainer | 2024 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn | ✓ |
| IGExplainer | 2024 | Surrogate | ✓ | ✓ | ✓ | Binary Tasks | Real / Synthetic | Qn / Ql | ✓ |
| SES | 2024 | Perturbation | ✓ | – | ✓ | – | Real / Synthetic | Qn / Ql | – |
| GAFExplainer | 2025 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |
| SEHG | 2025 | Perturbation | ✓ | – | ✓ | – | Real / Synthetic | Qn / Ql | – |
| SEGCRN | 2025 | Perturbation | ✓ | – | – | Forecasting | Real | Qn / Ql | ✓ |

**Legend:** ✓ = Supported, – = Not supported, Qn = Quantitative, Ql = Qualitative

#### :books: List of Papers in This Category

1. (**ExplaiNE**) Kang, B., Lijffijt, J., & De Bie, T. (2019). **Explaine: An approach for explaining network embedding-based link predictions**. arXiv preprint arXiv:1904.12694. [[Paper]](https://arxiv.org/abs/1904.12694) 
2. (**GNNExplainer**) Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). **Gnnexplainer: Generating explanations for graph neural networks**. Advances in neural information processing systems, 32. [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html) [[Source Code]](https://github.com/RexYing/gnn-model-explainer)
3. (**GraphMask**) Schlichtkrull, M. S., De Cao, N., & Titov, I. (2020). **Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking**. In International Conference on Learning Representations. [[Paper]](https://openreview.net/forum?id=WznmQa42ZAx) [[Source Code]](https://github.com/michschli/graphmask)
4. (**CoGE**) Faber, L., Moghaddam, A. K., & Wattenhofer, R. (2020). **Contrastive graph neural network explanation**. arXiv preprint arXiv:2010.13663. [[Paper]](https://arxiv.org/abs/2010.13663) [[Source Code]](https://github.com/lukasjf/contrastive-gnn-explanation)
5. (**PGExplainer**) Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., & Zhang, X. (2020). **Parameterized explainer for graph neural network**. Advances in neural information processing systems, 33, 19620-19631. [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html) [[Source Code]](https://github.com/flyingdoog/PGExplainer)
6. (**PGM-Explainer**) Vu, M., & Thai, M. T. (2020). Pgm-explainer: **Probabilistic graphical model explanations for graph neural networks**. Advances in neural information processing systems, 33, 12225-12235. [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/8fb134f258b1f7865a6ab2d935a897c9-Abstract.html) [[Source Code]](https://github.com/vunhatminh/PGMExplainer)
7. (**Causal Screening**) Wang, X., Wu, Y., Zhang, A., He, X., & Chua, T. S. (2020). **Causal screening to interpret graph neural networks**. [[Paper]](https://openreview.net/forum?id=nzKv5vxZfge)
8. (**ReFine**) Wang, X., Wu, Y., Zhang, A., He, X., & Chua, T. S. (2021). **Towards multi-grained explainability for graph neural networks**. Advances in Neural Information Processing Systems, 34, 18446-18458. [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/99bcfcd754a98ce89cb86f73acc04645-Abstract.html) [[Source Code]](https://github.com/Wuyxin/ReFine)
9. (**Gem**) Lin, W., Lan, H., & Li, B. (2021). **Generative causal explanations for graph neural networks**. In International Conference on Machine Learning (pp. 6666-6679). PMLR. [[Paper]](https://proceedings.mlr.press/v139/lin21d.html) [[Source Code]](https://github.com/wanyu-lin/ICML2021-Gem)
10. (**MEG**) Numeroso, D., & Bacciu, D. (2021). **Meg: Generating molecular counterfactual explanations for deep graph networks**. In 2021 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9534266?casa_token=GwiJKTHcQeQAAAAA:qNZMt3JXoNEbFKWoqyKDFeXK5PwcxffIUlpPa_F0uOi2VS2naZxKHkO_k4TUQMDDbBYOO8upO7BgYw)
11. (**RG-Explainer**) Shan, C., Shen, Y., Zhang, Y., Li, X., & Li, D. (2021). **Reinforcement learning enhanced explainer for graph neural networks**. Advances in Neural Information Processing Systems, 34, 22523-22533. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/hash/be26abe76fb5c8a4921cf9d3e865b454-Abstract.html)
12. (**SE-GNN**) Dai, E., & Wang, S. (2021). **Towards self-explainable graph neural network**. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 302-311). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482306) [[Source Code]](https://github.com/EnyanDai/SEGNN)
13. (**CMF**) Deng, S., Rangwala, H., & Ning, Y. (2021). **Understanding event predictions via contextualized multilevel feature learning**. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 342-351). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482309)
14. (**CMGE**) Wu, H., Chen, W., Xu, S., & Xu, B. (2021). **Counterfactual supporting facts extraction for explainable medical record based diagnosis with graph network**. In Proceedings of the 2021 conference of the north American chapter of the association for computational linguistics: human language technologies (pp. 1942-1955). [[Paper]](https://aclanthology.org/2021.naacl-main.156/) [[Source Code]](https://github.com/ckre/cmge)
15. (**RCExplainer**) Bajaj, M., Chu, L., Xue, Z. Y., Pei, J., Wang, L., Lam, P. C. H., & Zhang, Y. (2021). **Robust counterfactual explanations on graph neural networks**. Advances in Neural Information Processing Systems, 34, 5644-5655. [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/2c8c3a57383c63caef6724343eb62257-Abstract.html)
16. (**RelEx**) Zhang, Y., Defazio, D., & Ramesh, A. (2021). **Relex: A model-agnostic relational model explainer**. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (pp. 1042-1049). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3461702.3462562?casa_token=p739InrlUz4AAAAA:nAHJ_W9l4LNiKk8MVSLUMqXNXsefk8NtdMUl32Xf35mk2G1bfqSjVLGFbF6UJB6K6BSaarLDza1DKg)
17. (**SubGraphX**) Yuan, H., Yu, H., Wang, J., Li, K., & Ji, S. (2021). **On explainability of graph neural networks via subgraph explorations**. In International conference on machine learning (pp. 12241-12252). PMLR. [[Paper]](https://proceedings.mlr.press/v139/yuan21c.html) [[Source Code]](https://github.com/divelab/DIG)
18. (**GraphSVX**) Duval, A., & Malliaros, F. D. (2021). **Graphsvx: Shapley value explanations for graph neural networks**. In Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part II 21 (pp. 302-318). Springer International Publishing. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-86520-7_19) [[Source Code]](https://github.com/AlexDuvalinho/GraphSVX)
19. (**GStarX**) Zhang, S., Liu, Y., Shah, N., & Sun, Y. (2022). **Gstarx: Explaining graph neural networks with structure-aware cooperative games**. Advances in Neural Information Processing Systems, 35, 19810-19823. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7d53575463291ea6b5a23cf6e571f59b-Abstract-Conference.html) [[Source Code]](https://github.com/ShichangZh/GStarX)
20. (**CF2**) Tan, J., Geng, S., Fu, Z., Ge, Y., Xu, S., Li, Y., & Zhang, Y. (2022). **Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning**. In Proceedings of the ACM Web Conference 2022 (pp. 1018-1027). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3485447.3511948)
21. (**ZORRO**) Funke, T., Khosla, M., Rathee, M., & Anand, A. (2022). **Zorro: Valid, sparse, and stable explanations in graph neural networks**. IEEE Transactions on Knowledge and Data Engineering. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9866587?casa_token=DWmUW_X9no0AAAAA:wEXF1dQ-JaG9A-NtgXBn5wwCUg43IA9yaBYsCdq-RSl5jmXWfbtyY8Or8he1RDBs-v6B9crH_xWccA) [[Source Code]](https://github.com/funket/zorro)
22. (**TraP2**) Ji, C., Wang, R., & Wu, H. (2022). **Perturb more, trap more: Understanding behaviors of graph neural networks**. Neurocomputing, 493, 59-75. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231222004404?casa_token=0g9iTdQugZUAAAAA:vI-fOrDEWbA5YLr0TvroxBKsWfV-Y10_QpOJfh_cddsODftdWfEANrBLeP4-oDY618yaZt8H940) [[Source Code]](https://github.com/aI-area/TraP2)
23. (**GraphLime**) Huang, Q., Yamada, M., Tian, Y., Singh, D., & Chang, Y. (2022). **Graphlime: Local interpretable model explanations for graph neural networks**. IEEE Transactions on Knowledge and Data Engineering. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9811416?casa_token=IHNBRryIO2gAAAAA:92GIJ0aZ36H_plAcp88RYkN12G-pI6zjqpDdwagUGSRjNrFRsd0IPJSdWdLWIt5jHdpz-2lGVk1Cog)
24. (**MotifExplainer**) Yu, Z., & Gao, H. (2022). **Motifexplainer: a motif-based graph neural network explainer**. arXiv preprint arXiv:2202.00519. [[Paper]](https://arxiv.org/abs/2202.00519)
25. (**TAGE**) Xie, Y., Katariya, S., Tang, X., Huang, E., Rao, N., Subbian, K., & Ji, S. (2022). **Task-agnostic graph explanations**. Advances in Neural Information Processing Systems, 35, 12027-12039. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4eb7f0abf16d08e50ed42beb1e22e782-Abstract-Conference.html) [[Source Code]](https://github.com/divelab/DIG/tree/main/dig/xgraph/TAGE/)
26. (**EGIB**) Wang, J., Luo, M., Li, J., Lin, Y., Dong, Y., Dong, J. S., & Zheng, Q. (2023). **Empower Post-hoc Graph Explanations with Information Bottleneck: A Pre-training and Fine-tuning Perspective**. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2349-2360). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3580305.3599330?casa_token=Pwr3uPc_tUUAAAAA:4ybCiNaTuwg4qfXOjgwkzZniNQcBWGoIbU4RjUPb9_TyHoHPfQCpQz548CEnQ_7iF3TCITJv3-JNeg)
27. (**GOAt**) Lu, S., Mills, K. G., He, J., Liu, B., & Niu, D. (2024). **GOAt: Explaining graph neural networks via graph output attribution**. arXiv preprint arXiv:2401.14578. [[Paper]](https://arxiv.org/pdf/2401.14578) [[Source Code]](https://github.com/sluxsr/GOAt)
28. (**STFExplainer**) Ji, Y., Shi, L., Liu, Z., & Wang, G. (2024). **Stratified GNN explanations through sufficient expansion**. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 11, pp. 12839-12847). [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29180)
29. (**EAGX**) Liu, X., Ma, Y., Chen, D., & Liu, L. (2024). **Towards embedding ambiguity-sensitive graph neural network explainability**. IEEE Transactions on Fuzzy Systems. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10696966?casa_token=hCCLdMqMYf4AAAAA:kAzlW17w254kYeWOcjPRb8fKIc8JbN8JpLlt7N40Cc30LnheNUk6uxRHGh4b5T2NAwb7U3KtatK-LA) [[Source Code]](https://github.com/ymwkdaat/EAGX)
30. (**Geb**) Wang, Z., Zeng, Q., Lin, W., Jiang, M., & Tan, K. C. (2024). **Generating diagnostic and actionable explanations for fair graph neural networks**. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 19, pp. 21690-21698). [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/30168)
31. (**K-FactExplainer**) Huang, R., Shirani, F., & Luo, D. (2024). **Factorized explainer for graph neural networks**. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 11, pp. 12626-12634). [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29157)
32. (**XGExplainer**) Kubo, R., & Difallah, D. (2024). **Xgexplainer: Robust evaluation-based explanation for graph neural networks**. In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM) (pp. 64-72). Society for Industrial and Applied Mathematics. [[Paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.8) [[Source Code]](https://github.com/colab-nyuad/XGExplainer)
33. (**IGExplainer**) Bånkestad, M., Andersson, J. R., Mair, S., & Sjölund, J. (2024). **Ising on the graph: Task-specific graph subsampling via the Ising model**. arXiv preprint arXiv:2402.10206. [[Paper]](https://arxiv.org/abs/2402.10206) [[Source Code]](https://github.com/mariabankestad/IsingOnGraphs)
34. (**SES**) Huang, Z., Li, K., Wang, S., Jia, Z., Zhu, W., & Mehrotra, S. (2024). **SES: Bridging the gap between explainability and prediction of graph neural networks**. In 2024 IEEE 40th International Conference on Data Engineering (ICDE) (pp. 2945-2958). IEEE. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10597945?casa_token=FJaD07_YrT0AAAAA:ZNYjfh_ZKYUKx38Lus3aU1nBTY-CAg4NppS6jVtJXlOFTFI7MYghZ7LWpexYksMQPeEdWbB34bhK3Q)
35. (**GAFExplainer**) Hu, W., Wu, J., & Qian, Q. (2025). GAFExplainer: Global View Explanation of Graph Neural Networks Through Attribute Augmentation and Fusion Embedding. IEEE Transactions on Knowledge and Data Engineering. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10878445?casa_token=W_S__y8EGf0AAAAA:P_LoZwVlZO59Nwxj2Ys-R0CJhMZ-_oAyAa0vXApjqU4_gkZY_IaHDX32QYt0sgzQXSnH499Kv0MLQA) [[Source Code]](https://github.com/wyhi/GAFExplainer)
36. (**SEHG**) Huang, Z., Zhou, W., Li, Y., Wu, X., Xu, C., Fang, J., ... & Xia, F. (2025). **SEHG: Bridging Interpretability and Prediction in Self-Explainable Heterogeneous Graph Neural Networks**. In Proceedings of the ACM on Web Conference 2025 (pp. 1292-1304). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3696410.3714661)
37. (**SEGCRN**) García-Sigüenza, J., Curado, M., Llorens-Largo, F., & Vicent, J. F. (2025). **Self explainable graph convolutional recurrent network for spatio-temporal forecasting**. Machine Learning, 114(1), 2. [[Paper]](https://link.springer.com/article/10.1007/s10994-024-06725-6) [[Source Code]](https://github.com/jgars/SEGCRN)

## Generation-based Explainers

#### 📊 Comparison of Methods in This Category
| Method | Year | Approach | GNN-based | Graph-Classif. | Node-Classif. | Other Tasks | Data Type | Validation | Code |
|--------|------|----------|-----------|------|-------|-------------|-----------|------------|------|
| CogQA | 2019 | Graph Creation | ✓ | – | – | QA | Real | Ql | ✓ |
| XGNN | 2020 | Graph Creation | ✓ | ✓ | – | – | Real / Synthetic | Qn / Ql | – |
| ExplaGraph | 2021 | Graph Creation | ✗ | – | – | Stance Detection | Real | Qn | ✓ |
| GLGExplainer | 2023 | Surrogate | ✓ | ✓ | – | – | Real / Synthetic | Qn | ✓ |
| GCFExplainer | 2023 | Data Integration | ✓ | ✓ | – | – | Real | Qn | ✓ |
| LLM-KG | 2024 | Surrogate | ✗ | – | – | Generative | Real | Ql | – |
| KnowGNN | 2025 | Perturbation | ✓ | ✓ | ✓ | – | Real / Synthetic | Qn / Ql | ✓ |

**Legend:** ✓ = Supported, – = Not supported, Qn = Quantitative, Ql = Qualitative

#### :books: List of Papers in This Category

1. (**CogQA**) Ding, M., Zhou, C., Chen, Q., Yang, H., & Tang, J. (2019). **Cognitive Graph for Multi-Hop Reading Comprehension at Scale**. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 2694-2703). [[Paper]](https://aclanthology.org/P19-1259/) [[Source Code]](https://github.com/THUDM/CogQA)
2. (**XGNN**) Yuan, H., Tang, J., Hu, X., & Ji, S. (2020). **Xgnn: Towards model-level explanations of graph neural networks**. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 430-438). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403085)
3. (**ExplaGraphs**) Saha, S., Yadav, P., Bauer, L., & Bansal, M. (2021). **ExplaGraphs: An Explanation Graph Generation Task for Structured Commonsense Reasoning**. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 7716-7740). [[Paper]](https://aclanthology.org/2021.emnlp-main.609/) [[Source Code]](https://explagraphs.github.io/)
4. (**GLGExplainer**) Azzolin, S., Longa, A., Barbiero, P., Liò, P., & Passerini, A. (2023). **Global explainability of gnns via logic combination of learned concepts**. arXiv preprint arXiv:2210.07147. [[Paper]](https://arxiv.org/abs/2210.07147) [[Source Code]](https://github.com/steveazzolin/gnn_logic_global_expl)
5. (**GCFExplainer**) Huang, Z., Kosan, M., Medya, S., Ranu, S., & Singh, A. (2023). **Global counterfactual explainer for graph neural networks**. In Proceedings of the sixteenth ACM international conference on web search and data mining (pp. 141-149). [[Paper]](https://dl.acm.org/doi/abs/10.1145/3539597.3570376) [[Source Code]](https://github.com/mertkosan/GCFExplainer)
6. (**LLM-KG**) Abu-Rasheed, H., Abdulsalam, M. H., Weber, C., & Fathi, M. (2024). **Supporting student decisions on learning recommendations: An llm-based chatbot with knowledge graph contextualization for conversational explainability and mentoring**. arXiv preprint arXiv:2401.08517. [[Paper]](https://arxiv.org/abs/2401.08517) --- Hu, X., Liu, A., & Dai, Y. (2025). **Combining ChatGPT and knowledge graph for explainable machine learning-driven design: a case study**. Journal of Engineering Design, 36(7-9), 1479-1501. [[Paper]](https://www.tandfonline.com/doi/full/10.1080/09544828.2024.2355758) 
7. (**KnowGNN**) Ma, Y., Liu, X., Guo, C., Jin, B., & Liu, H. (2025). **KnowGNN: a knowledge-aware and structure-sensitive model-level explainer for graph neural networks**. Applied Intelligence, 55(2), 126. [[Paper]](https://link.springer.com/article/10.1007/s10489-024-06034-4) [[Source Code]](https://github.com/lxf770824530/KnowGNN)





