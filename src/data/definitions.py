import os

file_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(file_path))

TRAIN_RAW = os.path.join(ROOT_PATH, 'data/raw/training_v2.csv')
TEST_RAW = os.path.join(ROOT_PATH, 'data/raw/unlabeled.csv')


TRAIN_PROCESS = os.path.join(ROOT_PATH, 'data/processed/training_v2.csv')
TEST_PROCESS = os.path.join(ROOT_PATH, 'data/processed/unlabeled.csv')
SOLUTION_TEMPLATE = os.path.join(ROOT_PATH, 'data/raw/solution_template.csv')
