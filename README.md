# 基于扩散模型的连续故事生成
进入服务器、登陆显卡：`ssh RockyOS8-Login0`<br>
申请显卡：`sbatch demo.sh` （注意：需要80GB A100显卡）<br>
查看显卡：`squeue --me`<br>
登录显卡：`ssh r8a100-c01`（其中`r8a100-c01`为`squeue --me`的回显结果中`NODELIST(REASON)`对应的值）

## 第一环节：LLAMA2-7B模型（文本-文本）
运行方法：
进入环境：`conda activate LLAMA2`<br>
进入路径：`cd /lustre/S/yuxiaoyi/llama_bot/`<br>
执行指令：
```shell
CUDA_VISIBLE_DEVICES=0 python src/cli.py \
    --model_name_or_path /lustre/S/yuxiaoyi/LLAMA2-7b \
    --template llama2 \
    --temperature 0.95 \
    --top_p 0.7 \
    --top_k  50
```

## 第二环节：ARLDM模型（文本-图像）
## 1. raw data转为hdf5格式
### 1.1. 数据环境配置
raw data在`../dataset/split`中，hdf5文件保存在`../oxford_data`中 <br>
```shell
conda create -n raw2hdf5 python=3.11
conda activate raw2hdf5
pip install opencv-python
pip install h5py
pip install pillow
pip install tqdm
```
### 1.2. 运行方法
#### 生成hdf5文件
`cd Synthesizing-Coherent-Story-via-Latent-Diffusion-Models/data_script` <br>
`python oxford_hdf5.py`

## 2. 运行主程序
进入环境：`conda activate arldm`<br>
进入路径：`cd /lustre/S/yuxiaoyi/Synthesizing-Coherent-Story-via-Latent-Diffusion-Models/`<br>
执行指令：`python main.py`<br>
注意：
在`/lustre/S/yuxiaoyi/Synthesizing-Coherent-Story-via-Latent-Diffusion-Models/config.yaml`中按需修改相关参数<br>
训练的结果在：`/lustre/S/yuxiaoyi/device/save_ckpt/`<br>
生成的结果在：`/lustre/S/yuxiaoyi/sample/`<br>
