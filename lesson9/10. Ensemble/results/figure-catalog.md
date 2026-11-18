# 图表目录与核查口径

所有图由 scripts/complete_worksheet.py 生成，提供350 dpi PNG及PDF/SVG矢量版；英文坐标避免字体缺失，中文解读在工作单图注中。配色一致，色盲友好；曲线另用标记或线型区分。

| 图 | 数据源（本目录） | 用途与观察 | 误差/分布含义 | 限制 |
|---|---|---|---|---|
| fig1-depth-bootstrap | tree-depth.csv、bootstrap-simulation.csv | 深度3后验证表现不升；抽样覆盖率在理论值周围波动 | 右图为1000次模拟分布，非CI | 树深度仅一个划分 |
| fig2-ensemble-stability | model-comparison.csv、bootstrap-predictions.csv | 集成改善单树，降低大幅波动 | 每点为一个验证样本跨30次重训的SD；箱线描述样本间分布 | 非性能CI；未直接测相关系数ρ |
| fig3-tree-count | tree-count-curve.csv | 约10–50棵后平台；不必持续增加树数 | 点估计，无误差条 | 树数横轴对数，固定种子 |
| fig4-importance-correlation | feature-importance.csv、train-feature-correlations.csv | 相关特征分摊并替补，排名因方法而异 | 20次置换均值±SD | 中图仅展示不纯度前7名；完整置换排名在CSV；非因果 |
| fig5-boosting-curves | boosting-curves.csv | AUC稳定可掩盖验证log-loss回升 | 训练/验证曲线，无CI | 最优轮数由同一验证集选择，属于探索 |
| fig6-fair-comparison | cv-folds.csv、cross-validation.csv | 简单基线未被集成超过 | 圆点逐折，菱形均值±折间SD | 相依折，不能直接推断显著性 |

检查：六图均已打开查看；坐标、图例与图注对应；fig4减少横轴刻度以避免数字拥挤；精确值由CSV与工作单表格提供。AUC点图/折线局部缩放且标明刻度，未使用截断柱高暗示性能倍数。重要性条形图从零起。
