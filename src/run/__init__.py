from .run import run as default_run
from .on_off_run import run as on_off_run
from .dop_run import run as dop_run
from .per_run import run as per_run
from .dual_run import run_dual as dual_run

#创建字典，将不同模式的runer函数注册到字典中，将字典参数与runer下的各个类一一对应起来
REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run
REGISTRY["dual_run"] = dual_run