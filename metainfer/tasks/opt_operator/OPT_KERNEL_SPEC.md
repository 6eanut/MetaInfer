# opt-kernel / opt-operator 开发规格（dev 用）

> 本文件是**最高优先级**的需求文档，描述你要把 opt-operator 做成什么样。
> 开发 ccb 每次动这个 task 前先通读本文件；若你发现"当前代码行为 / 本文件 / 用户的即时说法"三者冲突，以用户即时说法为准，然后**更新本文件**与代码，保持本文件始终描述真目标。

---

## 0. 定位与一句话

MetaInfer 的 **opt-kernel（= `opt-operator` task）** 负责：给定一个**自然语言描述的算子需求**（要在某硬件上得到一个跑得快的算子，可细到如"GEMM 在 M/N/K 某 shape 下性能最佳"），自动生成"正确性 + 基准"两个 harness，然后做**多轮迭代算子/kernel 优化**，产出经过正确性验证、且性能优于基线的一批（池化）kernel。

它是一款**由 LLM agent 驱动、人机协作确认、结果可自证可信**的算子优化工具。用户是开发者本人；WebUI 是它对外的全部脸面。

---

## 1. 端到端期望流程（用户视角）

### 阶段 A · 需求理解与确认（交互式，人机往返，**发生在 task 创建期、run 之前**）

> **已实现（R1）**：确认是**创建期对话**（`opt-operator-conversation.js` wizard + 后端 `converse.py` 的 interpret/settle）——opt-operator 引导、用户在对话框用自然语言描述 → 逐项解读/引导缺项 → 收敛后呈**可编辑解读卡**，用户点「确认」后前端组装扁平 answers + `raw_request`(对话转写留痕) 走现有 `POST /tasks → launcher.start()` 起真实 run。**run 一启动即无 pending 确认**；`raw_request` 仅作 historical provenance，运行时不再读驱动。

1. 用户用**自然语言**提出需求：目标算子 + 目标硬件 +（可选）细粒度约束，例如"在 K100(gfx928) 上做一个 GEMM，要求 M=4096,N=4096,K=4096 时性能最好"。
2. opt-kernel **理解**这段自然语言 → 回给用户**简单、结构化**的解读（如：算子/问题域、目标硬件、关注的具体 shape 范围、约束）。
3. 用户确认；若有纠正 → 回到 2 继续修正，**直到用户认可需求被正确理解**。
4. 在确认中顺带敲定几个运行参数（每个都可用户改默认）：
   - 迭代优化用哪些 GPU；
   - baseline：**用户提供** 还是 **opt-kernel 自生成**；
   - shape 模式：只针对给定 shape（targeted）还是关心一段范围 / 一般形状（general）。

产出：一份**双方认可的需求契约**，作为本次 run 的单一事实来源，后续所有 harness 与迭代都以它为基准。

### 阶段 B · 生成 correctness + benchmark 两个 harness（4 步）
1. 生成 **correctness harness**：判断"迭代产出的 kernel vs baseline"在用户关心的输入范围内，结果是否满足误差容限（正确性保障）。
2. 派**对抗验证 agent** 审这个 correctness harness：它是否真的能在输入范围内抓出错误 kernel（会不会漏判 / 形同虚设）。
3. 生成 **benchmark harness**：判断优化 kernel 相较 baseline 的性能提升多少（口径、shape 集合与 correctness 一致）。
4. **对抗验证** benchmark harness 的权威性：**有无预热、是否多次运行、是否取稳定统计量（如均值/中位数）**、是否公平（同样编译优化、同样核数）——若权威性不足要修 harness。

### 阶段 C · 迭代优化循环（4 环节）
1. **选一个 kernel**：opt-kernel 维护一个 **kernel 池**，存所有"正确性通过 且 性能≥某门槛"的 kernel；最初可只有 baseline。每个 kernel 带**评分**（一般来自 benchmark harness 数据）。从池中**按评分加权概率**挑一个（评分越高越可能被选中）。
2. **优化**：先分析该 kernel 的源码（可逐步到编译产物/汇编），据此生成一个优化候选。初期可做得简单。
3. **验证**：把候选送进 correctness + benchmark 两个 harness。
   - 正确性不过 或 性能不达标 → 进入 4；
   - 两者都过 且 性能优于基线 → 候选入池；本轮成功，回 1 选下一个。
4. **修正**：基于 harness 给出的验证失败信息，对候选 kernel 做修订 → 回到 3 重新验证。**整个内循环最多 3 次**。3 次后仍不达标 → 丢弃该候选，回到 1 重新选一个 kernel 继续。

循环在达到（每轮/总预算）上限或用户叫停时结束。结束时向用户交付：**当前池内最优 kernel（champion）+ 完整血缘 / 每轮提升记录**。

---

## 2. 术语表

| 词 | 含义 |
|---|---|
| 需求契约 | 阶段 A 双方认可的结构化需求（算子的输入/输出/dtype/shape 范围/误差容限/硬件/约束） |
| baseline | 性能对照基准；用户提供或系统自生成，作为正确性对照与提速分母 |
| oracle / 参考实现 | 正确性的"标准答案"；被冻结(SHA-256)后所有 kernel 对它做符合性校验 |
| correctness harness | 封装"候选 vs 参考 是否满足误差"的检查（含 shape 范围矩阵） |
| benchmark harness | 封装"候选相对 baseline 提速多少"的公平测速（预热/多次/统计量） |
| kernel 池 | 存所有"正确性过且性能达标"kernel 的容器；每个带评分 |
| 评分 | 该 kernel 在 benchmark harness 上的性能度量（shape 集合上的代表值） |
| champion | 当前最优 kernel（从池中按需求口径选出，通常是评分最高） |

---

## 3. 功能需求（分阶段 P0/P1/P2）与验收标准

优先级：**P0 = 本次迭代必须做完且正确**；P1 = 紧随其后；P2 = 可后置/锦上添花。验收标准是"你能证明它做到了"的最低门槛。

> **实现进度（截至 R2）**：FR-0/1 已由 R1 创建期对话满足；FR-2/3 已由 R0 双 harness 对抗审校 + `negative_evidence` 留痕满足；FR-4/5/6/7 已由 R0 kernel 池 + 加权采样 + 3 次内循环 + resume 满足。各 FR 正文仍保留"目标态"描述作为后续增强的参照。

### FR-0 需求输入与确认（阶段 A）—— **已由 R1 创建期对话满足**（原 P1，曾是最缺失项）
- 现状：form.yaml 是一次性结构化表单，**没有**"结构化回显 → 用户纠正 → 再确认"的往返。
- 要做的：支持"用户给自然语言/一段话 + 关键字段表单"开始 → 系统给出**结构化解读卡**（算子、硬件、shape 口径、误差、baseline 来源、GPU、约束）→ 用户在界面上**逐项确认或纠正** → 直到用户点"确认无误"才真正起 run。
- 验收：界面存在"解读 → 可逐项改 → 确认/拒绝"三态；确认后生成的 requirements/契约与用户最后一次改动**完全一致**；用户拒绝时不启动迭代。
- P2：支持纯自然语言一段话直接进（后端抽结构化槽位，让用户改槽位而不是重打）。

### FR-1 契约解析（形状/误差/口径）
- 需求契约要能表达：算子签名（in/out/dtype）、**shape 集合**（一个明确 shape，或一段范围 sweep，或 general）、**误差容限**（abs/rel 阈值）。
- 验收：任给一个契约，能展开成 correctness 用的 case 矩阵与 benchmark 用的同口径 shape 集；targeted 与 general 在代码里语义清晰。
- 现有 `contract.py::OperatorContract` + shape-sweep DSL 是地基，**改造/复用而非推倒**。

### FR-2 correctness harness 生成 + 对抗审校（阶段 B 步骤 1-2）—— **P0 收敛**
- correctness harness 必须覆盖"用户关心的输入范围"，而不是只测一个点。
- 需要一个**对抗验证 agent**：只给 harness + 参考，故意构造/设想会让 harness 漏判的 kernel，检查 harness 是否真能抓错。对抗结论必须**留痕**（作为该 run 可信度证据，WebUI 可展示）。
- 验收：对抗审校输出结论进入 run 记录；能证明 harness 不是 rubber-stamp（对"改错的 kernel"能测出 fail，至少有一个负向用例证据）。
- 现有 `oracle.py`/`reference_lib.py`（生成参考 → 自证 → auto-review → 冻结）需评估是否已构成"对抗"；若只是形式审校，要升级为真对抗并记录证据。

### FR-3 benchmark harness 生成 + 权威性审校（步骤 3-4）—— **P0 收敛**
- benchmark 必须：同硬件/同编译口径、**预热**、**多次运行**、取**稳定统计量**（均值或中位数），shape 集合与 correctness 对齐。
- 需要**对抗/审查 agent** 逐项核上述关键点，结论留痕。
- 验收：harness 元信息（预热次数/重复次数/统计口径）在 run 记录与 WebUI 可见、可复核。

### FR-4 kernel 池 + 评分 + 加权概率选择（阶段 C 环节 1）—— **已由 R0 落地**（原 P0，曾是最错位项）
- 现状：`ledger.py::ChampionLedger` 只保留**单条 champion 链**（贪心跟随最好），**没有多 kernel 并存、没有概率采样**。
- 要做的：引入 **kernel 池**语义——
  - 入池条件：correctness 过 且 性能（相对 baseline）≥ 门槛（可配）。
  - 每个池内 kernel 有**评分**：由 benchmark harness 在其关心的 shape 集合上算出的代表性能（默认几何均值或对 targeted 用该 shape 的 latency；口径待用户定，先用基准方法并**留可配置**）。
  - 选择：按评分**加权概率**采样（高分高概率）。**支持固定随机种子以可复现/可回放**。
- 验收：单测证明"池里有多个 kernel 时，选择是概率分布且分数单调影响概率"；一次 run 的落盘能回放同一序列。
- P1：池容量上限与淘汰（如只留 top-N 或保血缘），超限策略待用户定。

### FR-5 优化动作（环节 2）—— **P0 起步从简，P2 加深**
- 分析候选源码（必要时到编译产物/汇编），产出优化候选。初期允许走"LLM 基于源码与评分反馈提出改动"的简单路径。
- P2：接入编译/汇编产物分析、profile 热点引导、auto-tuning 启发（特化 tile/unroll/constexpr 等 targeted 手段）。
- 验收：一次优化动作有可复现输入（源码+反馈）与可审查输出（候选 + 说明改了啥/为何）。

### FR-6 验证 + 3 次内循环修正（环节 3-4）—— **P0 收敛并落账**
- 候选送双 harness；fail 时基于 harness 反馈修订，**内循环 ≤3 次**；最终不达标丢弃、达标入池。
- 每一条"候选→验证→(修正×n)→结论"都要**可落账、可展示**（这就是 WebUI 最有意思的部分之一）。
- 验收：坏候选被记 failed 并安全跳过/回退，**绝不 crash 整轮**；3 次上限在配置与记录里可见。

### FR-7 运行控制与预算 —— **P0**
- 外层轮次上限（每轮从选 kernel 到入池/丢弃记一轮）、总预算（成本/时间）、GPU 集合，都来自阶段 A 的确认结果。
- resume/冷重启：能从落盘权威源重建池、当前轮、已跑结论，继续而非从头。

---

## 4. 状态机与落盘（SSOT，对齐根 CLAUDE.md）

> **已实现（R0/R1）**：`phases.py` 已从线性流水线 `S_baseline→A…→F→finished` 重映射为"池化演化 + harness 审校"机。**需求确认不在 run 内** —— 它以 R1 的**创建期对话**完成（见 §1 阶段 A），run 前即产出冻结的 requirements.json 作为唯一事实来源；因此 run 状态机**不含任何 confirm/interpret 相位**（那段逻辑只活在 `orchestrator/converse.py`，供创建器对话用）。

已实现的目标状态机（给 WebUI 与 resume 用的单一机，`phases.py::_PHASES`）：
```
harness_setup(一次性: 基线入池为genesis + correctness/benchmark 双harness 对抗自审, STRONG)
  → select_kernel(池内按评分加权概率采样, 固定seed)
    → optimize → verify(双harness: correctness gate + benchmark 全shape)
       →[correctness不过 或 评分低于门槛]→ repair(≤3次, 依harness结构化反馈修) → verify
       →[过 且 ≥门槛]→ admit_to_pool
       →[≤3次后仍不过/低于门槛]→ discarded
  → admit_to_pool|discarded → select_kernel … → finished(预算耗尽, 从select_kernel停)
```
节点：`harness_setup`(STRONG) / `select_kernel` `optimize` `verify` `repair` `admit_to_pool` `discarded`(CHEAP) / `finished`(终态)。
入池条件 = correctness 全 case 过 **且** 相对 baseline 评分 ≥ 门槛（不要求每 shape 都超 incumbent）；champion 每次从池内按需求口径现算选出（派生，不单存）。

落盘规则（**新字段先定权威，禁双写**）：
| 事实 | 权威源建议 | 派生 |
|---|---|---|
| 需求契约（双方认可后冻结） | `requirements.json` 或独立 `contract.json`（写死 immutable） | — |
| 池内 kernel + 评分 | 新 `kernel_pool.jsonl`（append-only，存入池证据：digest/harness结论/评分/血缘） | 当前 champion = 池内最高分（派生，不单存） |
| harness 审校结论 | 对应 phase 的 iteration/记录 | WebUI 展示（派生） |
| 每轮 select/optimize/verify/repair | `iterations/NNN.json` + `timeline.jsonl` | WebUI 血缘/流水（派生） |
- **已落地**：`kernel_pool.jsonl` 是唯一权威（append-only）；`ledger.py::ChampionLedger` 已降为**只读派生**的池血缘视图（沿用旧方法名 `read_all/lineage/…` 但内部改读池文件），**池写、链读，无双向写**。
- 冷重启只能从 `kernel_pool.jsonl` 权威重建池与 champion。

---

## 5. WebUI 规格

用户偏好 **dark** 风格（复用 shell 主题 token），要求 **美观简洁**、**把整个过程可视化**、**有意思的展示、没意思的不展示**。它是对外全部脸面，别做成"工程师调试 dump"。

### 5.1 需求确认界面（阶段 A）—— 新增，核心体验
- 显示**结构化解读卡**（算子 / 硬件 / shape 口径 / 误差 / baseline 来源 / GPU / 约束），每项可**直接编辑 + 打勾确认**；提供"确认无误，开始""驳回，重解读"。与运行参数（GPU、baseline 来源）同屏。
- 形式可参考表单 + 解读摘要二合一，避免让用户觉得"又要填一遍表"。

### 5.2 总览页（run 进行中/结束后）
建议以"**一张演化图讲清来龙去脉**"为目标：
- 状态 stepper（run 内，§4 节点，**不含创建期确认**）：harness_setup → select → optimize → verify → [admit_to_pool|discarded]（当前到哪、是否卡住）。
- **kernel 池视图**：每个池内 kernel 的评分/提速/何时入池/血缘，一眼看出"现在最好的几个是谁、怎么来的"。
- **有意思的部分要突出**：① 每次被采纳的 kernel 相比前代/基线**提升多少、改了哪**；② harness 被对抗审校出的问题与修复（这是"可信度"的卖点）；③ 当前最优(champion)相对基线的整体提速。
- **没意思的不要堆首屏**：agent 原始日志、过程噪音、失败的中间试探默认折叠，需用户点开。
- GPU 占用/空闲、轮次预算消耗做成小而清晰的指示，不喧宾夺主。

### 5.3 数据口径可信
- 每个分数/提速标注**口径与来源**（哪个 benchmark harness、预热/重复/统计量、哪批 shape），避免用户把中间态误读成最终效果。
- 显示"参考来源：用户提供 / 参考库 / 生成+对抗审校"，让用户能信任正确性结论。

### 5.4 复用而非重写
- 复用 shell 主题 token、现有 `/api/{type}/{task_id}` 挂载、`static/opt-operator.css` 的 dark 基座、SSE `/events` 实时更新。
- 大改前先在本地静态审一遍结构与信息层级，必要时请用户给当前截图对照。

---

## 6. 与现有代码的映射 / 改造要点（R0/R1/R2 已落成）

| 文件 | R0/R1/R2 后的真实状态 |
|---|---|
| `orchestrator/pool.py`（新） | **权威 SSOT** `kernel_pool.jsonl`（append-only）+ `PoolEntry`/`sample_kernel`(评分加权, 固定seed)/`champion()`/`quality()`(派生现算) |
| `orchestrator/ledger.py` | 降为**只读派生**的池血缘视图（`ChampionLedger` 内部改读 pool，方法名兼容） |
| `orchestrator/harness.py`（新） | 收敛出 `CorrectnessHarness` / `BenchmarkHarness` 两对象 + 元信息(预热/重复/统计量/shape集/基线) |
| `orchestrator/adversarial.py`（新） | 对抗审校：负向用例扰动 → 断言 harness 测出 fail → `negative_evidence`/`ReviewCheck` 留痕 |
| `orchestrator/oracle.py`+`reference_lib.py` | 参考冻结(SHA-256)+对抗审校留痕；correctness/benchmark 两 harness 生命周期明确 |
| `orchestrator/backend.py`+`conformance.py`+`profiler.py` | 收敛为两个 harness 入口；预热/重复/统计量**可配置 + 入 harness 元信息** |
| `orchestrator/contract.py` | shape-sweep DSL 对接确认后契约；case 矩阵覆盖"用户关心范围" |
| `orchestrator/converse.py`（新） | 创建期对话引擎：interpret/settle 抽槽位、缺项引导、收敛产出扁平 answers+`raw_request` |
| `orchestrator/phases.py` | 已重映射为池化演化机（§4 节点），graph_payload 渲染 stepper |
| `server/_state_readers.py`+`routes.py` | 只读派生：`read_pool`/`read_harness_reviews`/`read_lineage`/…；`/pool` `/harness` 端点；只读不写、不缓存、纯派生 |
| `static/opt-operator-detail.js`+`.css` | 池 top 视图 + Harness 可信度(negative_evidence) + 口径 chip + 有意思/没意思分层 + stepper |
| `static/opt-operator-conversation.js`（新） | 创建期引导对话 wizard + 解读卡确认/驳回 |
| `form.yaml` | `__meta__` 标记开启对话承载；字段作为解读卡底层槽位 |

改造铁律（持续生效）：**两端同步动**——改写入侧字段/阶段名/状态值，必须同步 reader 与前端；新增字段先定权威源；champion/血缘/提速等派生量每次现算、不落盘。

---

## 7. 非目标 / 当前明确不做（避免扩散）

- 暂不做**编译产物/汇编的深度优化引导**（P2，可后置；初期源码级即可）。
- 不做与硬件 vendor 库绑定的自动调优平台；约束缺省"禁 vendor 库、确定性、尊重误差容限"。
- 不做多算子联合/图级优化；本 task 聚焦**单算子/单 kernel**。
- 不做通用"任意算子自动全自动无人化"——**保留人机确认阶段**是本设计有意为之。

---

## 8. 分阶段路线（建议，R0–R2 已完成）

> 历史路线——R0/R1/R2 均已实现并测试覆盖（见 §3 进度、§6 落成映射）。此后若继续，参照本段思路推进 R3+ 或补 P2 增强。

- **R0（P0 收敛，先改"池化演化"内核）**：FR-4/5/6/7——把单 champion 贪心改为 kernel 池 + 加权采样 + 3 次内循环修复 + 落账；补负向用例证明 harness 非 rubber-stamp。让"优化机制符合预期"先成立。双 harness 与对抗审校(FR-2/3)在本阶段按"自证/审查留痕"收敛，不追求 UI 完美。
- **R1（阶段 A 交互式确认）**：FR-0/1——解读卡 + 逐项确认/纠正 + 运行参数同屏，进入真正的"先确认再优化"。WebUI 出确认界面。
- **R2（WebUI 全流程可视化）**：§5 全部——池视图、有意思/没意思分层、口径标注、状态 stepper。
- 每一步都有测试与 resume/坏候选覆盖；每步结束对照 §3 验收标准自查。

---

## 9. 质量与验收红线（对齐根 MetaInfer CLAUDE.md）

- **单一权威源**：每份事实一个权威文件；禁双写/双向同步；WebUI 任何重置/清理走单一路径。
- **派生量不落盘**：champion、提速、血缘视图每次从权威现算。
- **跨主机**：GPU 租约用 `os.link`-based claim；lease 属 orchestrator；worker 永不 release。
- **每改必有测试**；resume/冷重启路径覆盖；坏候选记录 failed 并保留 incumbent、绝不 crash 整轮。
- **提交自查**：新增/改动字段是否已有人存？我读的是权威还是缓存？冷重启后还能取对值吗？有没有把派生量当权威写盘？WebUI/reader 是否同步了我改的口径？
