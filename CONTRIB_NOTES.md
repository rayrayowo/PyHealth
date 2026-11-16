# PyHealth 贡献工作台（rayrayowo fork）

> 这个文件只放在 `notes` 分支，**不要**合并进任何提交给上游的 PR 分支。
> 每个接手的 agent / 人：先读完本文件，干完活后更新「进度」和「决策日志」，再 push `notes` 分支。

## 目标

- 在 PyHealth（UIUC SunLab，PyHealth 2.0 论文已被 ICML 2026 收录）持续做有分量的贡献。
- 路线：完成上游 `bounty.yaml` 里的 bounty。历史上完成 bounty 的贡献者多数进入了 PyHealth 2.0 论文作者名单（无成文规则，不保证）。
- 首个目标：**BTY-24 Cardiology Dataset PyHealth 2.0 Migration**（P1, dataset）。

## 仓库与分支约定

| 名称 | 说明 |
|---|---|
| `origin` | https://github.com/rayrayowo/PyHealth （fork，可 push） |
| `upstream` | https://github.com/sunlabuiuc/PyHealth （官方，只读） |
| `master` | 跟随 upstream/master，不在上面直接提交 |
| `notes` | 只放本文件等协作笔记 |
| `bty-24-cardiology` | BTY-24 的实现分支，从最新 upstream/master 拉出；PR 从这里开到上游 |

同步上游：`git fetch --depth=50 upstream master && git checkout master && git merge --ff-only upstream/master && git push origin master`

## BTY-24 任务说明（摘自上游 bounty.yaml）

> Move CardiologyDataset (CPSC 2018, Georgia, PTB, PTB-XL, INCART) onto the 2.0 BaseDataset and keep
> cardiology_detect working, using TUEV as the reference port. Check open PRs #1052, #1187, and #1323
> first to avoid duplicate PTB-XL work.

- 旧代码：`pyhealth/datasets/cardiology.py`（约 128 行，基于 legacy `BaseSignalDataset`）
- 下游任务：`pyhealth/tasks/cardiology_detect.py`（`cardiology_isAD_fn` 等）
- 参考迁移：`pyhealth/datasets/tuev.py` + `configs/tuev.yaml`；`isruc.py`（上游 #1486，2026-11-16 合并）
- 已有：`pyhealth/datasets/ptbxl.py` + `configs/ptbxl.yaml`（PTB-XL 已有 2.0 实现，需决定复用还是对齐）
- #1187、#1323 均已关闭（2026-10/11）；截至 2026-11-16 没有 `[BTY-24]` 标题的 PR。
- 数据：PhysioNet/CinC Challenge 2020/2021 公开数据，无需 MIMIC 认证。测试必须用合成数据（见 docs/how_to_contribute.rst）。

## 上游 PR 规则（必须遵守）

- 详见 `CONTRIBUTING.md` 与 `docs/how_to_contribute.rst`，开工前通读。
- PR 标题以 `[BTY-24]` 开头，才会被 bounty board 识别；开 PR 时勾选 "Allow edits by maintainers"。
- 只改 `docs/ examples/ pyhealth/ test-resources/ tests/`；不要改 `BaseDataset`、Trainer 等核心代码，否则转人工审核。
- CI 规则：新增/修改行 `ruff check` 无告警（88 字符）；每个新增/修改的公开类/函数 docstring 里要有 `>>>` 示例。
  本地检查：`python tools/check_pr_rules.py --base <base_sha> --head HEAD`
- 审核流程：label `status: automatic` 时评论 `/review` 触发 AI 审核，AI 通过后评论 `/merge`；最多 10 次 AI 审核。
- 非小改动先在 issue / Discord 讨论。

## 环境备注

- Python 3.13（要求 >=3.12,<3.14）；venv 在 `/home/user/.venvs/pyhealth`（会话容器内，非持久）。
- 云端容器无法访问 download.pytorch.org，torch 需从 PyPI 安装。

## 进度

- [x] 1. Fork 到 rayrayowo/PyHealth
- [x] 2. 建立 `notes` 分支与本文件
- [ ] 3. 本地环境安装、跑通现有测试
- [ ] 4. 读 TUEV / ISRUC / PTB-XL 迁移写法，写 BTY-24 方案（放在本文件「方案」一节），由用户确认
- [ ] 5. 实现 + 合成数据测试 + 通过 check_pr_rules
- [ ] 6. 用户在 Discord/Slack 说明认领，用户提交 `[BTY-24]` PR 到上游

## 决策日志

- 2026-11-16：选定 PyHealth 作为主攻项目（医疗数据 + 论文署名先例 + 与简历的数据校验主线一致）。
- 2026-11-16：首个任务选 BTY-24（公开数据、纯数据工程、生理信号方向，与运动康复/可穿戴相关）。

## 方案

（第 4 步填写）
