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
- 容器访问不了 huggingface.co，相关测试必然失败；另需 `pip install 'monai>=1.6.1,<2'`。
- 跑测试：`pytest tests/core -q`（和上游 CI 一致，约 15 分钟）；只跑相关的：`pytest tests/core/test_cardiology*.py tests/core/test_isruc.py`

## 进度

- [x] 1. Fork 到 rayrayowo/PyHealth
- [x] 2. 建立 `notes` 分支与本文件
- [x] 3. 本地环境安装、跑通现有测试（2026-11-16：tests/core 3621 通过；42 个失败全是容器环境问题：HuggingFace 被代理拦截、缺 nibabel、以 root 运行导致只读目录测试失效）
- [~] 4. 读 TUEV / ISRUC / PTB-XL 迁移写法，写 BTY-24 方案（放在本文件「方案」一节），由用户确认
- [ ] 5. 实现 + 合成数据测试 + 通过 check_pr_rules
- [ ] 6. 用户在 Discord/Slack 说明认领，用户提交 `[BTY-24]` PR 到上游

## 决策日志

- 2026-11-16：选定 PyHealth 作为主攻项目（医疗数据 + 论文署名先例 + 与简历的数据校验主线一致）。
- 2026-11-16：首个任务选 BTY-24（公开数据、纯数据工程、生理信号方向，与运动康复/可穿戴相关）。
- 2026-11-19：上游 @irvifa 已开 PR #1551 `[BTY-24] Port CardiologyDataset to the PyHealth 2.0 BaseDataset`，内容覆盖我们方案的全部要点（每条记录一个病人、按真实采样率重采样、按字段解析表头）。不再做 BTY-24，避免重复 PR；改选其他 bounty（待用户决定）。

## 方案（BTY-24 草案，已作废：见决策日志 2026-11-19）

### 读旧代码发现的问题（PR 描述里要写清楚，这是我们的亮点）

1. **采样率写死为 500 Hz**：`cardiology_detect.py` 用 `500 * epoch_sec` 切窗。Challenge 2020 里
   PTB 是 1000 Hz、INCART 是 257 Hz，所以这两个来源的"10 秒窗口"其实不是 10 秒。采样率在 `.hea`
   第一行就有（`<record> <n_sig> <fs> <n_samples> ...`）。（待用真实数据核实）
2. **按行号倒数解析表头**：`label_content[-4]` 取 Dx、`[-5]` Sex、`[-6]` Age，表头注释行顺序或数量一变就错位。
   应按 `#Dx:` / `#Sex:` / `#Age:` 前缀解析。
3. **"patient" 其实是 g1/g2… 分卷文件夹**：旧代码把每个 `gN` 文件夹当成一个病人，但这些文件夹只是
   约 1000 条记录一卷的分包，不是病人。按病人划分训练/测试集会因此出错。（待核实，需和维护者讨论 ID 方案）
4. `len(os.listdir(dataset_root)) - 1` 假设目录里恰好多一个 `RECORDS` 文件，有 `index.html` 等就会少算或多算。

### 实现计划（参照 ISRUC #1486 的写法）

1. `pyhealth/datasets/cardiology.py`：`CardiologyDataset(BaseDataset)`
   - `prepare_metadata()` 扫描 `<root>/<source>/g*/<record>.hea`，按键解析 WFDB 表头，写
     `cardiology-pyhealth.csv`（一行一条记录）：`patient_id, record_id, source, signal_file,
     header_file, sampling_rate, n_samples, n_leads, age, sex, dx_codes`
   - 用 `metadata_csv.py` 的工具（root 不可写时落到 cache），与 ISRUC 一致
   - 参数：`sources: list[str]`；旧的 `chosen_dataset=[1,1,1,1,1,1]` 继续接受，发 DeprecationWarning
   - `default_task` 返回新任务类
2. `pyhealth/datasets/configs/cardiology.yaml`：表 `recordings`
3. `pyhealth/tasks/cardiology_detect.py`：新增 `CardiologyDetection(BaseTask)`，参数 `category`
   ∈ {AR, BBBFB, AD, CD, WA}，复用旧代码的 SNOMED 编码集合；按每条记录的真实采样率切窗
   （可选重采样到统一频率）。**旧的 `cardiology_is*_fn` 函数保留不动**，满足 "keep cardiology_detect working"。
4. 测试 `tests/core/test_cardiology.py`：合成 `.hea` + `.mat` 文件，覆盖目录布局、按键解析表头、缺字段、
   混合采样率（500/1000/257）、来源筛选、dev 模式、`default_task`、窗口数量与标签
5. 文档：API 文档条目 + 示例；所有新公开函数带 `>>>` 示例；`ruff` 与 `tools/check_pr_rules.py` 通过

### 开工前要问维护者的问题（用户在 Discord / issue 里问）

- patient_id 方案：每条记录当作一个病人，还是保留旧 ID 以兼容？
- 非 500 Hz 的来源：重采样到 500 Hz，还是按真实采样率切窗？
- PTB-XL 已有 `PTBXLDataset`：Cardiology 里的 `ptb-xl` 来源保持独立，还是复用它？
