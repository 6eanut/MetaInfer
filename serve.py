#!/usr/bin/env python3
"""MetaInfer WebUI 一键启动入口。

用法:
    ./serve.py                # 前台运行，默认 127.0.0.1:8765
    python serve.py           # 等价
    METAINFER_PORT=9000 ./serve.py
    ./serve.py --host 0.0.0.0 --port 8765

这个脚本不依赖 `pip install` —— 它把仓库根目录加到 sys.path，
然后直接调用 metainfer.server.app:main()。如果你已经 `pip install -e .`
或 `pip install metainfer`，也可以用 `metainfer-web` 命令，等价的。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """把仓库根目录插到 sys.path 最前，让 `import metainfer` 能找到。"""
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _bootstrap_path()

    # 延迟 import，确保 sys.path 已经修好。
    try:
        from metainfer.server.app import create_app
    except ImportError as e:
        sys.stderr.write(
            f"[metainfer] 无法加载 metainfer 包: {e}\n"
            "  检查依赖是否安装：pip install -r requirements.txt\n"
            "  或者：pip install -e .\n"
        )
        return 2

    parser = argparse.ArgumentParser(
        prog="metainfer-serve",
        description="启动 MetaInfer WebUI（前台运行）。",
    )
    parser.add_argument("--host", default=os.environ.get("METAINFER_HOST", "127.0.0.1"),
                        help="监听地址（默认：env METAINFER_HOST 或 127.0.0.1）")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("METAINFER_PORT", "8765")),
                        help="监听端口（默认：env METAINFER_PORT 或 8765)")
    parser.add_argument("--reload", action="store_true",
                        help="开发模式：源代码变更时自动重载")
    args = parser.parse_args()

    # 用环境变量传给 uvicorn —— main() 也会读它们。
    os.environ["METAINFER_HOST"] = args.host
    os.environ["METAINFER_PORT"] = str(args.port)

    import uvicorn
    app = create_app()
    print(f"[metainfer] WebUI 启动中: http://{args.host}:{args.port}")
    print(f"[metainfer] 按 Ctrl-C 退出。打开浏览器访问上面的 URL。")
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info", reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
