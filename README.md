[TOC]

# 训练 

获取基础配置

https://spacy.io/usage/training

```bash
python -m spacy init fill-config base_config.cfg config.cfg
```

准备数据, 然后训练

```bash
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --output model_dir
```
