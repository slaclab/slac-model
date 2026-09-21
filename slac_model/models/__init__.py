import yaml
from pathlib import Path

CU_HXR_PROFMON_INFO = yaml.safe_load(open(Path(__file__).parent / "data"/"cu_hxr_profmon_info.yaml"))

