在本机跑需安装anaconda。可参考https://blog.csdn.net/weixin_42855758/article/details/122795125

安装完后win+R打开cmd
cd 文件目录（如：cd C:\Users\19662\Desktop\models\demo）
如文件目录不在C盘，假设在D盘则先输入D:
进入D盘路径后再执行 cd D:\Users\19662\Desktop\models\demo

正确进入目录后执行
conda create -n torch python=3.8
activate torch
pip install -r requirements.txt
（CPU版本安装方式，GPU版本的torch以及torchvision需另行安装）

（GPU版本安装方法，如只使用CPU可忽略）
如想在gpu上训练需进一步安装CUDA、cuDNN，可参考https://blog.csdn.net/weixin_42496865/article/details/124002488
CUDA以及cuDNN请自行搜索本机显卡匹配版本！！！
GPU版本的torch以及torchvision安装参考https://blog.csdn.net/weixin_44994302/article/details/117962299
可到此寻找对应的版本离线安装（较为方便快捷）whl轮子地址：https://download.pytorch.org/whl/torch_stable.html
或执行如下在线安装方法（torch以及torchvision的版本按需求更换，'+cu'代表cuda即GPU版本）
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

部分显卡不支持CUDA以及cuDNN也可用cpu进行训练（无需焦虑小任务cpu也可完成）

若使用CPU进行训练则在train.py的14行修改为Cuda = False

安装完环境后
python train.py即可执行程序，也可在pycharm或vscode执行

优化器和模型定义在train.py（注释较多仔细查看应该较好理解）
修改模型在nets/model.py
修改训练策略或损失函数在utils/trainer.py（20到48行为训练，53到73行为测试，76行之后为保存权重文件，保存的权重文件会在logs文件夹下）

