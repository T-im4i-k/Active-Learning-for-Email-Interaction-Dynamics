import re
from pathlib import Path

ROOT_FOLDER = Path(re.sub(r"(.*?/email_outreach/).*", r"\1", __file__))
