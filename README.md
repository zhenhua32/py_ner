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

TODO: GPU 没法用, 

```bash
cupy.cuda.compiler.CompileException: `nvcc` command returns non-zero exit status.
command: ['C:\\Program', 'Files\\NVIDIA', 'GPU', 'Computing', 'Toolkit\\CUDA\\v11.2\\bin\\nvcc.EXE', '-gencode=arch=compute_75,code=sm_75', '--ptx', '-DFIRST_PASS=1', '--std=c++11', '-IC:\\Anaconda3\\envs\\transformers\\lib\\site-packages\\cupy\\_core\\include', '-IC:\\Anaconda3\\envs\\transformers\\lib\\site-packages\\cupy\\_core\\include\\cupy\\_cuda\\cuda-11', '-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include', '-ftz=true', 'C:\\Users\\tzh\\AppData\\Local\\Temp\\tmp187r3plz\\preprocess.cu']
return-code: 1
stdout/stderr:
nvcc fatal   : nvcc cannot find a supported version of Microsoft Visual Studio. Only the versions between 2017 and 2019 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
```

环境又搞崩了, 删了重新建个 env. 现在又卡在 cupy 上了.
还有个问题是在 conda 的其他环境下, python 和 pip 总是指向 base 环境.
