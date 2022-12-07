import time
from spacy.cli.train import train


def main():
    config_file = "./config/config.cfg"
    train_file = "./data/resume_zh_spacy/train.spacy"
    dev_file = "./data/resume_zh_spacy/dev.spacy"
    model_dir = "model_dir/cpu"

    # CPU 训练, 用时 244 s
    # GPU 训练坑我,, 用了 475 s, 你这是认真的吗?
    train(
        config_file,
        overrides={
            "paths.train": train_file,
            "paths.dev": dev_file,
        },
        output_path=model_dir,
        use_gpu=-1,
    )


def main_gpu():
    config_file = "./config/config_gpu.cfg"
    train_file = "./data/resume_zh_spacy/train.spacy"
    dev_file = "./data/resume_zh_spacy/dev.spacy"
    model_dir = "model_dir/gpu"

    train(
        config_file,
        overrides={
            "paths.train": train_file,
            "paths.dev": dev_file,
        },
        output_path=model_dir,
        use_gpu=0,  # 似乎是个整数, -1 不使用 GPU, 0 使用第一个 GPU, TODO: 怎么用多 GPU
    )


if __name__ == "__main__":
    start = time.time()
    main()
    # main_gpu()
    print("time cost: ", time.time() - start)
