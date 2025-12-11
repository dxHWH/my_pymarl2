import logging
import sys

def setup_logging(log_level=logging.INFO):
    """
    配置一个根记录器 (root logger) 用于向控制台输出。
    """
    # 获取根记录器
    logger = logging.getLogger() 
    
    # 防止重复添加 handlers (如果在 notebook 中多次运行)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)
    
    # 1. 创建控制台处理器 (StreamHandler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # 2. 创建格式化器 (Formatter)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 3. 为处理器设置格式化器
    handler.setFormatter(formatter)
    
    # 4. 将处理器添加到记录器
    logger.addHandler(handler)
    
    return logger