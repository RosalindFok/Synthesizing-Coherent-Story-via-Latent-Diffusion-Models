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
`python oxford_hdf5.py`
