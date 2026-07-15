import re
from pathlib import Path

ROOT_FOLDER = Path(re.sub(r"(.*?/Active-Learning-for-Email-Interaction-Dynamics/).*", r"\1", __file__))
