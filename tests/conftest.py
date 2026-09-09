# -*- coding: utf-8 -*-
"""pytest 共享配置：把 src/ 加入 sys.path，使测试可直接 import 核心模块。"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
