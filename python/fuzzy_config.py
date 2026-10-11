"""
共享实验配置加载器 (Shared experiment configuration loader)
=========================================================
本模块负责读取仓库根目录 ``shared/fuzzy_config.json``——它是 Python、MATLAB
以及自动生成的 ``.fis`` 文件三者共同的**唯一数据源 (single source of truth)**。

为什么必须这样做？
------------------
在旧版本工程中，Python 的规则库、MATLAB 类的规则库、``generate_fis.m`` 生成的
``.fis`` 文件以及 README 中的规则表 **四者互不一致**（尤其 ΔKi 规则矩阵），
导致"同一套算法"在不同语言下给出不同的控制量，实验结论无法复现。

现在所有语言都从本 JSON 读取隶属度函数参数与 49 条规则，任何不一致都会被
   python/test_fuzzy_core.py     （Python 单元测试）
   matlab/verify_consistency.m   （MATLAB 与 .fis 交叉验证）
立刻捕获。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

_CACHE: Dict[str, Any] = {}


def config_path() -> str:
    """向上搜索 shared/fuzzy_config.json 的绝对路径。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "shared", "fuzzy_config.json"),
        os.path.join(os.path.dirname(here), "shared", "fuzzy_config.json"),
        os.path.join(os.path.dirname(os.path.dirname(here)), "shared", "fuzzy_config.json"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "未找到 shared/fuzzy_config.json。请保持仓库目录结构完整，"
        "或从仓库根目录运行脚本。已尝试:\n  " + "\n  ".join(candidates)
    )


def load_config() -> Dict[str, Any]:
    """读取并缓存整份配置。"""
    path = config_path()
    if path not in _CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            _CACHE[path] = json.load(fh)
    return _CACHE[path]


def scenario(name: str) -> Dict[str, Any]:
    """读取指定场景（'thermal' 工业加热炉 / 'motor' 直流伺服电机）的全部参数。"""
    cfg = load_config()
    if name not in cfg["scenarios"]:
        raise KeyError(f"场景 '{name}' 不存在，可选: {list(cfg['scenarios'])}")
    return cfg["scenarios"][name]


def fuzzy_block() -> Dict[str, Any]:
    """模糊推理系统定义（隶属度函数 + 规则库）。"""
    return load_config()["fuzzy"]
