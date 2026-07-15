<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/MetaInfer/MetaInfer/main/docs/logo.png" alt="MetaInfer" width="200" onerror="this.style.display='none'">
</p> -->

<h1 align="center">MetaInfer</h1>

<h3 align="center">
  <em>Let AI generate high-performance, compact, purpose-built LLM inference frameworks</em>
  <p>
  <em>Describe what you want — MetaInfer builds your optimized inference engine</em>
</h3>

<h4 align="center">
  <em>(An LLM-powered AI Infra optimization toolkit)</em>
</h3>

<p align="center">
  <a href="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml">
    <img src="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/metainfer">
    <img src="https://img.shields.io/pypi/v/metainfer?color=blue" alt="PyPI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-%3E%3D3.9-blue" alt="Python">
  </a>
  <a href="https://github.com/MetaInfer/MetaInfer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="https://star-history.com/#MetaInfer/MetaInfer&Date">
    <img src="https://api.star-history.com/svg?repos=MetaInfer/MetaInfer&type=Date" alt="Star History" width="600">
  </a>
</p>

---

<p align="center">
  <a href="README_CN.md">中文文档</a>
</p>

---

## What is this? (Our vision)

* ***Let AI generate simple, compact inference frameworks that match or surpass SOTA performance. (Killer app)***

  * Today's SOTA inference frameworks are buckling under their own weight.
    * The traditional approach — multi-layer abstractions to support every model, every hardware target, every parallelism strategy, every quantization scheme, every kernel optimization — no longer keeps pace with the pace of LLM innovation.
    * The vast majority of code in an inference framework is unnecessary once your deployment constraints are fixed, yet that compatibility logic drags down performance.

  * Tell MetaInfer what you need, and it writes a high-performance, purpose-built inference framework for your exact setup.
    * Which model? What hardware? What parallelism? What quantization? Optimize for throughput or latency?

* ***AI-assisted kernel optimization***
  * Given a target kernel, the LLM drives an automated [optimize → benchmark] loop.

* ***Model porting***
  * New model unsupported on older GPUs? Point MetaInfer at the upstream framework code and let it backport to a lower-version target.

* ***Embedded / edge inference***
  * MetaInfer targets C/C++ and Rust as first-class output languages — laying the groundwork for edge computing and embodied AI.



### Built-in task types

| Type | Description |
|---|---|
| **`gen-infer-framework`** | Build a model-specific inference server with an OpenAI-compatible HTTP API. An immutable oracle boots `serve.sh`, sends fixed prompts, and dispatches an LLM judge to verdict correctness. |
| **`calc-theoretical-value`** | Compute theoretical FLOPs and memory-traffic for a single forward pass of an LLM. Fully read-only deterministic pipeline: model inspection → memory modeling → compute graph → visualization. |
| **`example`** | Canonical skeleton for new task types. Copy, rename, uncomment `register()`, implement your pipeline — no shared code touched. |

## Quick start

```bash
git clone https://github.com/MetaInfer/MetaInfer.git
cd MetaInfer
pip install -r requirements.txt
./serve.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), click **+ New Task**,
pick a task type, describe your requirements, and the LLM gets to work.

```bash
# Other ways to start
./serve.py --host 0.0.0.0 --port 9000
METAINFER_PORT=9000 ./serve.py
python -m metainfer.server.app
```

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture details, design
principles, and how to add new task types.
