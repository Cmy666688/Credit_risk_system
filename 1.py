import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import ConfigLoader, get_logger, Evaluator
configs = ConfigLoader("./configs")
print(configs.get("model").xgboost.params.max_depth)   # 应打印 6
