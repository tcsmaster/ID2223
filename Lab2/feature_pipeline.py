
from datasets import Dataset, load_dataset


def load(path:str) -> Dataset:
    ds =load_dataset(path)
    return ds
if __name__ == "__main__":
    ds = load('./datasets')
    print(ds)
