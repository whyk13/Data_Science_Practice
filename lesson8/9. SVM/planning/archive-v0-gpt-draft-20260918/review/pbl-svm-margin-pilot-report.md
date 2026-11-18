# Pilot-fill report

- Worksheet: `pbl-svm-margin-student-worksheet.docx`
- Pilot copy: `review/pbl-svm-margin-pilot-fill.docx`
- Date: 2026-09-18
- Persona / independence mode: fresh subagent；生命科学优秀大四直博生，微积分与统计先修，无CS/优化先修。独立审阅仅使用学生可见材料，未读取教师答案、构建脚本或设计文件。
- Lint: 0 errors, 0 warnings；Design: 0 errors, 0 warnings。
- Fields: 107 · answered: 107 · unanswered: 0
- Blockers: **0 block** · 0 friction
- Estimated minutes (sum of field estimates): 218
- Gate: **PASS** (also requires lint and design with 0 errors and time within budget)

## Minutes by part

下表是提取器按文档物理位置分组，不是教学阶段用时。追踪表T0属于启动，T1–T6分别属于课外Q1–Q6；以下校正后的阶段时间用于发布门槛：

| 教学阶段 | 独立估时 | 预算 | 偏差 |
|---|---:|---:|---:|
| 课堂启动（含T0） | 44.5 | 45 | −1.1% |
| 课外自学（含T1–T6、证据卡与综合） | 125.75 | 120 | +4.8% |
| 翻转课堂（含共享讲解、质询与个人修订） | 48 | 45 | +6.7% |

所有阶段及最终各模块估时均在预算20%范围内。当前模块分配为17、10、15、20、13、22分钟，另23分钟证据卡与综合，合计120。没有为迎合预算而下调字段估时。完整首次问题与复核历史见 `cold-read-audit.md`。

| Part | Minutes |
|---|---|
| Phase 1 课堂启动 | 36 |
| 总答案追踪表 | 37 |
| Phase 2 Packet A 几何与数据 | 14 |
| Packet B 支持向量与容错 | 18 |
| Packet C 核选择与验证 | 29 |
| Packet D 新记录与适用范围 | 19 |
| 课外综合 研究备忘录初稿 | 18 |
| Phase 3 翻转课堂与课堂修订 | 48 |

## Blockers by pitfall

| Code | Pitfall | block | friction |
|---|---|---|---|

## Blocker list

| Field | Location | Prompt | Code | Severity | Note | Suggested fix |
|---|---|---|---|---|---|---|

## Decision check

能。试填从启动阶段暂选L，利用A/B限定间隔、支持向量与容错的意义；C阶段核查两类召回率，冻结训练集标准化的线性SVM、C=10。D阶段用38/42与15/20保留L，不因R的测试分较高换核。Q6讨论高p小n和大n的不同限制，综合备忘录调用T1–T6。

## Residual friction and risks

最终0 block、0 friction。已解决：备忘录三段/四段冲突、课堂/课后修订时点冲突、二次核算例指代和综合局部时间不足。每项修订后的实际DOCX均经独立重读，记录保留在冷读审计中。

这是AI模拟试填，不是真实学生试教。估时包括阅读、查图、计算与表达；真实班级阅读英语资料和公式所需时间仍有个体差异。教师应按120分钟量级安排自学，不将可选延伸追加为必做。

作者另做了四份正式Word文件的逐页渲染检查和阶段版分页检查；辅助Notebook16个代码单元全部执行无异常。试填红字副本只保存在review，不发给学生。
