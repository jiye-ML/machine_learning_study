# PCA 主成分分析

* 基础知识请参见[10.PCA主成分分析.docx](10.PCA主成分分析.docx)

## 写在前面

* [为什么PCA不被推荐用来避免过拟合？](https://www.zhihu.com/question/47121788)
    * pca只关注了输入之间的correlation。(supervised) learning寻求的是输入和输出之间的联系。想象一个极端情况：
    输入的snr很低且噪声强相关，但是被学习的系统有超强能力完全忽略噪声。如果使用pca，估计很可能只保留了噪声。
    如果想要强调输入输出之correlation，不妨试试partial least square regression.
    * 
