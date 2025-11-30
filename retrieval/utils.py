import os
import pathlib
from typing import Callable

def cached_index(index_path: pathlib.Path, build_func: Callable, *args, **kwargs):
    if os.path.exists(index_path):
        print(f"✅ 索引已缓存，直接加载: {index_path}")
        return
    print(f"📦 索引未缓存，开始构建: {index_path}")
    build_func(*args, **kwargs)
    if not os.path.exists(index_path):
        print(f"⚠️ 警告：索引构建后未找到缓存文件 {index_path}")

























