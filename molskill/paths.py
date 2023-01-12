import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
RESULT_PATH = os.path.join(ROOT_PATH, "results")
MODEL_PATH = os.path.join(ROOT_PATH, "models")
MODEL_LOG_PATH = os.path.join(MODEL_PATH, "logs")
LOG_PATH = os.path.join(ROOT_PATH, "module_logs")
TEST_PATH = os.path.join(ROOT_PATH, "tests")
