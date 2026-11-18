# 第10讲工作单完成计划

## 目标
阅读 lesson9 全部教学材料，使用默认 WDBC 完成 student-worksheet.md，附真实实验数据与论文风格图表。

## 步骤
- [x] 核对工作单、教师参考、课程设计、演示脚本和两套数据说明。
- [x] 提取19页课件与数据准备代码，核实示例树、划分和特征语义。
- [x] 固定 seed=1010，在原划分运行模型比较、共享五折 CV、30次共享 bootstrap、树深度/树数/boosting实验。
- [x] 保存逐折、逐样本预测与汇总数据，生成6幅 PNG/PDF/SVG 图并逐图检查。
- [x] 简明填写 A–H，核对数值、图表链接和备忘录字数（387字符，含260汉字）。

## 决策与边界
- 课程未指定二选一，按 workplan.md 默认选择方案 A WDBC；CTG 材料一并阅读但不另做整套实验。
- 不读取 locked-test-labels.csv 或 all-measurements.csv；不以教师参考数字代替实测。
- 主指标 ROC-AUC，默认分类阈值；折间 SD 仅为描述量，不作独立样本显著性检验。
- 原教师材料、原数据和原演示脚本保持不变；新增可复现分析脚本、结果和图表。

## 问题记录
- 当前工作目录无 .git，改用输入 SHA-256 与文件清单核对改动。
- pubfig 未安装，按 publication-chart-skill 尝试安装；必要时用 matplotlib 完成实际图表。
- pubfig 安装尝试：`python -m pip install --target 'lesson9/10. Ensemble/.work/deps' --no-deps --disable-pip-version-check --retries 0 --timeout 15 pubfig`；索引返回 No matching distribution found。复查仍不可用，采用 Matplotlib。
- 首次打印课件文本受 Windows GBK 编码限制；设置 PYTHONIOENCODING=utf-8 后成功读取19页。
- 本机 Python 环境较课程 requirements 旧（sklearn 1.2.2）；保留原环境并记录精确版本。AdaBoost 默认 SAMME.R 与教师新版本不同，不能直接照抄其参考值。
- 校验脚本首次用 `\s*` 检测空表格，误跨换行匹配两行边界；改为仅匹配水平空白 `[ \t]*`，重新校验。
- 校验导出文件时 Path.with_suffix 需要点前缀；将 pdf/svg 修正为 .pdf/.svg 后重新校验。

## 状态
全部完成。verify_worksheet.py 通过：从保存的预测重算7种模型验证指标、35折CV指标及5种模型稳定性；13份受检查输入哈希未变；6图三种格式有效；工作单链接与表格完整。

## 交付范围
- 修改：student-worksheet.md。
- 新增：scripts/complete_worksheet.py、scripts/verify_worksheet.py；figures/下18份图文件；results/下数据、运行与验证记录、统计口径及图表说明；task_plan.md、notes.md。
- 保持：原数据、教师参考、README、workplan、课件、requirements和原4份脚本；未打开teacher数据表，未解锁测试。

## Word 交付追加计划
- [x] 依据已完成Markdown生成可编辑Word文档，嵌入6张图、13张表、全部问题及答案。
- [x] 使用本机Word渲染10页并逐页查看；移除模板标题蓝线，保持代码块、小表和备忘录完整；核对A–H、备忘录与关键AUC数值。
- [x] 交付docx，保留Markdown和实验数据不变。

Word交付：`第10讲 集成学习与Bootstrapping 学生工作单完成版.docx`；生成脚本：`scripts/build_worksheet_docx.py`。内部最终预览：`.work/word-final/`。
渲染记录：打包render_docx.py因缺少soffice.exe失败；沙箱Word COM因登录会话不可用失败；通过获准的本机Word只读转换成功。首次PowerShell路径语法已改为System.IO.Path.GetDirectoryName。
