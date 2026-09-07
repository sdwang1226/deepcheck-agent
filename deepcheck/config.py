"""运行时配置：环境变量、代理、路径。所有脚本通过 `from deepcheck import config` 统一初始化。"""
import os
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_indices")
REGISTRY_PATH = os.path.join(INDEX_DIR, "registry.json")
FILINGS_DIR = os.path.join(DATA_DIR, "10k_filings")
EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "eval_results")

EMBEDDING_MODEL = os.getenv("DEEPCHECK_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_LOCAL_ONLY = os.getenv("DEEPCHECK_EMBEDDING_LOCAL_ONLY", "0") == "1"
LLM_MODEL = os.getenv("DEEPCHECK_LLM_MODEL", "deepseek-chat")

# Longbridge 的域名在任何情况下都直连，LLM 是否走代理由 HTTP(S)_PROXY 决定。
_LONGBRIDGE_HOSTS = "open.longportapp.com,openapi.longbridge.com,openapi.longportapp.com"
os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), _LONGBRIDGE_HOSTS]))

if os.getenv("DEEPCHECK_PATCH_TORCH_CHECK") == "1":
    # transformers>=4.5x 要求 torch>=2.6 才允许 torch.load；老版本 torch 环境下可显式开启此绕过。
    import transformers.modeling_utils
    import transformers.utils.import_utils

    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
