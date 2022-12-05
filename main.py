from spacy.cli.train import train


def main():
    config_file = "./config/config.cfg"
    train_file = "./data/resume_zh_spacy/train.spacy"
    dev_file = "./data/resume_zh_spacy/dev.spacy"
    model_dir = "model_dir/cpu"

    train(
        config_file,
        overrides={
            "paths.train": train_file,
            "paths.dev": dev_file,
        },
        output_path=model_dir,
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
    )


if __name__ == "__main__":
    # main()
    main_gpu()
