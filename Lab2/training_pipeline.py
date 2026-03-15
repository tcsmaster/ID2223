import modal
from datasets import DatasetDict, KeyDataset
app = modal.app("Hungarian AS(M)R")
image = modal.Image\
        .debian_slim()\
        .apt_install(["ffmpeg"])\
        .
@modal.app(image = image)
def main():


if __name__ == "__main__":
    main()
