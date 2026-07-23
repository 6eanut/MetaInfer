# MetaInfer 多机多节点 GPU 执行层 — 架构文档

## 目标

MetaInfer 编排器原本只能在本节点跑子进程，限制了：
1. **多任务 GPU 共享冲突** — 多个内核优化任务并行 perf 时互相干扰。
2. **大模型调试受限** — PP2/TP 大型推理框架必须跨节点才能端到端跑通。

本架构引入"远程提交并执行"机制：worker 进程在远端 GPU 节点运行，
orchestrator 通过共享 NFS 上的文件系统协调。

---

## 顶层文件系统布局

```
$METAINFER_ROOT/
├── nodes/<node_id>/...                   # 既有 per-node 树（不动）
└── cluster/                              # NEW — 跨节点协调状态
    ├── workers/<node_id>.json            # SSOT: worker 身份 + GPU 拓扑
    ├── workers/<node_id>.heartbeat       # mtime = 心跳（永不重写 JSON）
    ├── scoreboard/<node_id>/
    │   ├── gpu-<i>.claim                 # hardlink-claim = GPU slot 互斥锁
    │   └── gpu-<i>.meta.json             # 派生: holder/job/lease_until
    ├── inbox/<worker_node_id>/<job_id>/  # 提交通道
    │   ├── job.json, script.sh / prompt.txt
    │   ├── stdout.log, stderr.log        # worker 边跑边 append
    │   ├── status.json
    │   ├── claimed                       # link-claim marker = 消费者抢占
    │   └── cancel.marker                 # 取消信号（force-kill 时由 reaper 写）
    └── replies/<orchestrator_node_id>/<job_id>.result.json
```

`cluster/` 与 `nodes/` **必须**在同一 NFS 挂载上（hardlink-claim 跨文件系统无效）。

---

## 核心算法决策

### 1. NFS-safe 原子 claim = `os.link`

生产者写 tmp 文件，再 `os.link(tmp, gpu-<i>.claim)`。NFS 服务端 link 是原子的，
多个并发只有一个成功（其他 EEXIST）。

**禁止**使用 `fcntl.flock` 跨主机 — NFS flock 的语义不可靠。

参见 `metainfer/cluster/fs_primitives.py::link_claim`。

### 2. 死锁避免 = 全局排序 + 一次性获取

`acquire_gpus(slots, holder, job_id, deadline_s)`：
- 对所需 slot 集合**确定性排序**（按 (node_id, gpu_idx)）
- 逐个 `link_claim`
- **任何一个失败 → 释放本次已获取的全部 slot（rollback）→ jitter backoff → 重试**
- 直到 deadline 或全部成功

**死锁证明**：每个持有者在持有 slot 后**不再等待其他 slot**（要么全拿，要么
全部释放才重试）。循环等待图无法形成。

参见 `metainfer/cluster/scoreboard.py::acquire_gpus`。

### 3. 租约 = LeaseToken (含 secret)

`acquire_gpus` 返回 `LeaseToken(holder, slots, secret, lease_until)`。
`release_gpus(token)` 校验 secret 与 claim 文件内容一致才 unlink。

- claim 文件 link 后**不可变**
- 续约（`renew_lease`）写到 sibling `.meta.json`（flock 内 atomic rewrite）
- reaper 优先读 `.meta.json` 的 `lease_until`

参见 `metainfer/cluster/scoreboard.py::LeaseToken`。

### 4. 单 reap 路径不变量

**所有**"释放他人 slot"的代码路径**必须**走 `scoreboard.force_release`：

- WebUI force-kill API
- mqueue 的 `reap_orphaned_submissions`（worker_dead / timeout）
- scoreboard 自己的 `reap_expired_claims`

**禁止**写第二个简化版 reaper。镜像 CLAUDE.md `launcher._reap_dead_pid_file`
不变量。

### 5. 死 worker 检测 = 心跳 stale + lease 过期

reap 条件**两者都满足**：
- `lease_until` 已过
- `worker.<holder>.heartbeat` mtime stale（> 60s）

只要 worker 心跳还在，lease 即使过期也不 reap（worker 可能正在续约）。

### 6. 消息队列消费 = `claimed` marker link

worker `consume_next_job` 用 `link_claim(job_dir/claimed, payload)` 抢占；
失败说明别的 worker 拿了。结果 tmp + `os.replace` 到
`replies/<orch>/<job_id>.result.json`。

参见 `metainfer/cluster/mqueue.py::consume_next_job`。

### 7. 流式日志 = append + mtime

worker `Popen(stdout=open(stdout.log, "ab"), stderr=...)`；心跳循环 flush
+ `os.utime`。orchestrator / WebUI 用 `read(offset=N)` 读最新字节。

---

## Job 生命周期

```
[orchestrator]                              [worker]
      │
      ├─ acquire_gpus(slots)
      ├─ submit_job(spec) ──write──→ inbox/<worker>/<job_id>/
      │                       job.json, script.sh
      │
      │                                    ├─ consume_next_job (link claimed marker)
      │                                    ├─ Popen(script.sh) → stdout.log, stderr.log
      │                                    ├─ watch cancel.marker + deadline
      │                                    │    └─ SIGTERM → 5s grace → SIGKILL
      │                                    ├─ child exit → status.json
      │ ←────── result.json ──────── write_result ───┤
      │
      ├─ read_result(job_id)
      └─ release_gpus(token) [in finally]
```

### Status 编码

| status | 含义 |
|---|---|
| `pending` | 已 submit，未 consumed |
| `inflight` | worker 已 claimed，未完成 |
| `done` | exit 0 |
| `failed` | exit ≠ 0 |
| `timeout` | 超时被 SIGTERM |
| `cancelled` | cancel.marker 触发 |
| `worker_dead` | worker 心跳 stale（被 reaper 标记） |

---

## 模块边界

| 模块 | 文件 | 职责 |
|---|---|---|
| FS 原语 | `metainfer/cluster/fs_primitives.py` | atomic write, link_claim, heartbeat |
| 路径 | `metainfer/cluster/paths.py` | 所有 cluster/ 子路径 helper |
| 拓扑 | `metainfer/cluster/topology.py` | nvidia-smi / rocm-smi 探测 |
| Worker 注册 | `metainfer/cluster/worker_registry.py` | SSOT worker 记录 + 心跳 |
| Scoreboard | `metainfer/cluster/scoreboard.py` | GPU 互斥 + lease + reaper |
| 队列 | `metainfer/cluster/mqueue.py` | submit / consume / result / reaper |
| SDK | `metainfer/cluster/sdk.py` | RemoteJob + submit_script/agent/pp2 |
| CLI | `metainfer/cluster/cli.py` | `metainfer-cluster` 命令 |
| Worker daemon | `metainfer/worker/daemon.py` | 心跳 + poll + supervisor |
| Job runner | `metainfer/worker/jobs.py` | Popen + 流式日志 + 超时/取消 |
| HTTP | `metainfer/server/cluster_routes.py` | /api/cluster/* |

---

## SSOT 表（新增）

| 数据 | 权威源 | 派生 / 历史快照 |
|---|---|---|
| Worker 身份 + GPU 拓扑 | `workers/<node_id>.json` | WebUI `/api/cluster/workers` 响应（运行时派生） |
| Worker 心跳 | `workers/<node_id>.heartbeat` mtime | `/api/cluster/workers[].alive` （派生） |
| GPU slot 持有者 | `scoreboard/<n>/gpu-<i>.claim` | `.meta.json`（派生: lease_until 可改） |
| Job spec | `inbox/<w>/<j>/job.json` | WebUI `/api/cluster/jobs` 响应（派生） |
| Job 结果 | `replies/<orch>/<j>.result.json` | WebUI 任务状态（派生） |
| Job 日志 | `inbox/<w>/<j>/stdout.log` / `stderr.log` | 唯一权威，无派生 |

---

## 反模式（**禁止**）

- ❌ 跨主机 flock（用 link_claim）
- ❌ 写第二个 reaper 简化版（必须复用 `force_release`）
- ❌ 重写 worker heartbeat JSON（heartbeat 只 touch mtime）
- ❌ 重写 claim 文件（link 后不可变；用 `.meta.json` 续约）
- ❌ Worker 进程内持有 LeaseToken（lease 属 orchestrator；worker 只报状态）
- ❌ Auto-requeue worker_dead 任务（按用户决策"Surface 为失败"）

---

## 部署模型

- 所有节点共享 NFS（`$METAINFER_ROOT` 同一挂载）
- 编排器节点：跑 `metainfer-server` + 各 task orchestrator
- Worker 节点：跑 `python -m metainfer.worker --node-id <name>`
- LAN 信任（无应用层鉴权）
- Worker 节点需预装 `ccb` 二进制（agent 类 job 复用）
