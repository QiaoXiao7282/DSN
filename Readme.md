# DSN Official Pytorch implementation


This is code for the paper titled Dynamic Sparse Network for Time Series Classification: Learning What to “See”

## Requirements

The library requires Python 3.6.7, PyTorch v1.4.0, and CUDA v9.0
You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 

## Dataset

[UCR and UEA](http://www.timeseriesclassification.com/) archives and some private datasets. 

[UCI](https://github.com/titu1994/MLSTM-FCN/releases) datasets

## Training

To train models for **UCR 85 Archive**, change the value of --root (e.g., UCR_TS_Archive_2015) and run this command: 

```
python trainer_DSN.py --sparse True --density 0.2 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 47 --c_size 3 --k_size 39

```

To train models for **UCI datasets**, change the value of --root (e.g., UCI) and run this command: 
```
python trainer_DSN.py --sparse True --density 0.2 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 47 --c_size 3 --k_size 39

```

To train models for **UEA 30 Archive**, change the value of --root (e.g., UEA_TS_Archive_2018) and run this command: 
```
python trainer_DSN.py --sparse True --density 0.1 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 59 --c_size 3 --k_size 39

```

#### With the default hyperpermeter setting! no need to search!

Just have a try!!!

  
​        
​    
