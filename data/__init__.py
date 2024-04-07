import subprocess
import os

from .recover import get_untyped, get_recovered
from .get_dataloader import get_train_test_dataloader
from .get_dataloader import get_inf_dataloder

if not os.path.exists("data/generated-or-not"):
    subprocess.run(["bash", "data/download_generated_or_not.sh"])
