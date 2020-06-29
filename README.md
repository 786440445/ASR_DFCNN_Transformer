# Corpus
- ai_shellv1
- thchs30
- aidatatang
- st-cmds
- primewords

# Deep Convolutional Neural Network SpeechRecognition
- acoustic model is designed by dfcnn and ctc
- language model is designed by transformer encoder
```python
    python3 train.py
    python3 test.py
```
## model structure
![](./img/am_lm.png)

##　详细说明
1. 本工程采用tensorflow框架完成
2. 数据集 aishell, stcmds, thchs30, aidatatang, primewords
3. model_and_log/logs_am 存储了训练好的两个声学模型
    model_05.7.64.hdf5采用上述五个训练好的声学模型
    model_04-14.91.dhf5是在上述五个数据集中添加了随机噪声训练的结果
4. 麦克风输入模块: read_wav.py
5. 常量设置模块 const.py
6. 参数设置模块 hparams.py
    