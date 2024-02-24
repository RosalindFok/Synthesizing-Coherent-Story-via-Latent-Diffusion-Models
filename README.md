# Synthesizing-Coherent-Story-via-Latent-Diffusion-Models
基于扩散模型的连续故事生成

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
#### 运行主程序
进入实验节点`ssh RockyOS8-Login0` <br>
申请显卡`sbatch apply_GPU.sh` <br>
查看显卡`squeue --me` <br>
登录显卡`ssh r8a100-c01`, 其中`r8a100-c01`为`squeue --me`的回显结果中`NODELIST(REASON)`对应的值 <br>
激活环境`conda activate arldm`  <br>
`python main.py` <br>