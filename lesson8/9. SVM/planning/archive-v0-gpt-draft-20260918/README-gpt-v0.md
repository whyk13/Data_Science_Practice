# 第9讲 SVM 与 Margin

本版日期：2026-09-18。按最新版 pbl-ml-package 重新设计；主载体为学生学习单，Notebook 仅辅助复现。旧 `plan.md`、`svm.md` 保留作历史记录，本版教学实施以以下文件为准。

## 教学文件

- `pbl-svm-margin-student-worksheet.docx`：完整学习单母版，含实际证据、概念读页和个人答题区。
- `pbl-svm-margin-evidence-pack.docx`：同编号证据汇编，不含标准答案。
- `pbl-svm-margin-teaching-plan-flipped-classroom.docx`：教师流程、参考答案、追问和评分。
- `pbl-svm-margin-design-rationale.docx`：内容复核与设计理由。

时间安排：课堂启动45分钟、课外自学约120分钟、翻转课堂45分钟。全程个人作答；随机选1–2位学生白板讲解，不设固定组内角色。学生不必编写新代码。

## 分阶段发放

`student-release/` 中的六份阶段版从同一学习单母版提取，不是额外作业。按以下顺序开放，学生始终保留最初的总答案追踪表：

1. 启动与追踪表；
2. Packet A；
3. Packet B；
4. Packet C；
5. Packet D与综合；
6. 翻转课堂与数据入口。

完成并提交追踪表“T5 冻结方案”行后再开放 Packet D。第一次上课前不要同时公开完整学习单、完整证据包或已执行 Notebook，否则无法维持测试证据的阶段隔离。

`review/` 内含红字模拟试填和教师质检记录，不向学生发放；PDF/PNG是排版复核副本，正式可编辑课件为上述 Word 文件。

## 数据与方法

保留既有计划选定的 UCI Vertebral Column：310条发布记录、6项连续特征，正常100、异常210。真实数据用于离线方法选择；六点算例及双月形模拟分别用于精确几何和二维核边界观察。两种模拟均不声称是真实患者或细胞数据。

来源：[UCI Vertebral Column](https://archive.ics.uci.edu/dataset/212/vertebral%2Bcolumn)，DOI 10.24432/C5K89B，CC BY 4.0。原始文件保存在 `assets/pbl-svm-margin/source-data/`；单位、缺失和记录独立性限制在课件中明确说明。

固定分层划分为训练186、验证62、测试62。缩放只拟合训练集；验证选择参数，测试后不得换核。结果只支持教学与该历史数据内部的方法比较，不证明临床用途、病因或外部人群表现。

## 复现与质检

在本目录运行 `scripts/pbl-svm-margin-build.py`，会重新计算所有证据并生成四份课件和阶段版。依赖 Python、numpy、pandas、scikit-learn、matplotlib、python-docx、nbformat；已用 scikit-learn 1.9.0 执行，具体版本见 `assets/pbl-svm-margin/environment.json`。本地数据齐备后无需联网。

`notebook/pbl-svm-margin-lab.ipynb` 已执行；它调用随包提供的完整分析函数，重新拟合模型、生成图表，并额外核查手算几何、合页损失和二次核恒等式，不是仅展示缓存结果。

`review/` 保存设计覆盖检查、学习单检查、独立模拟学生试填、冷读指代审计及逐页排版检查。模拟试填用于寻找题目缺陷，不是真实学生试教或学习效果证据。重新构建会使既有试填和排版结论失效，必须重新运行发布检查；`package-manifest.json` 记录当前状态。
