import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
RESULT_PATH = os.path.join(ROOT_PATH, "results")
MODEL_PATH = os.path.join(ROOT_PATH, "models")
MODEL_LOG_PATH = os.path.join(MODEL_PATH, "logs")
LOG_PATH = os.path.join(ROOT_PATH, "module_logs")
TEST_PATH = os.path.join(ROOT_PATH, "tests")

DEFAULT_CHECKPOINT_PATH = os.path.join(
    MODEL_PATH, "default", "checkpoints", "last.ckpt"
)

DEFAULT_CHECKPOINT_REMOTE: str = "https://ndownloader.figshare.com/files/40451207"
DEFAULT_MOMENTS_REMOTE: str = "https://ndownloader.figshare.com/files/38956727"
DEFAULT_CHECKPOINT_ARCHIVE_SHA256 = (
    "9335ec4a46a15748df39dd27da27cc272b5aef045c3df24b134f51950feb07a1"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "ee0460a09baadb8188f22c84ede59c4b5b7d55871a76dadf6829a45ed11dc90a"
)
