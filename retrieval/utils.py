import os
import pathlib
from typing import Callable

def cached_index(index_path: pathlib.Path, build_func: Callable, *args, **kwargs):
    """
    缓存索引的通用工具函数
    :param index_path: 索引文件路径（用于判断是否已缓存）
    :param build_func: 构建索引的函数
    :param args: 构建函数的位置参数
    :param kwargs: 构建函数的关键字参数
    """
    if os.path.exists(index_path):
        print(f"✅ 索引已缓存，直接加载: {index_path}")
        return
    print(f"📦 索引未缓存，开始构建: {index_path}")
    build_func(*args, **kwargs)
    if not os.path.exists(index_path):
        print(f"⚠️ 警告：索引构建后未找到缓存文件 {index_path}")

























