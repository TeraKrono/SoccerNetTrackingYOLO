```bash 
conda create -n tracklab pip python=3.9 pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate tracklab
cd sn-gamestate
pip install -e .
mim install mmcv==2.0.1
pip install SoccerNet
```

```
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
                                       split=["train", "valid", "test", "challenge"])
```

```bash
cd data/SoccerNetGS
unzip gamestate-2024/train.zip -d train
unzip gamestate-2024/valid.zip -d valid
unzip gamestate-2024/test.zip -d test
unzip gamestate-2024/challenge.zip -d challenge
cd ../..
```

```bash
conda run tracklab -cn soccernet
```
