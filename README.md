# PseudoLabeling-VAD
This repo contains the Pytorch implementation of our method

## Dependencies

```bash
conda create -n PL_VAD python=3.8
conda activate PL_VAD
pip install -r requirements.txt
```


## Training




### Data download 
**Please download the extracted I3d features for UCF-Crime and XD-Violence dataset from links below:**
> [**UCF-Crime I3D features**](https://github.com/Roc-Ng/DeepMIL)
> 
> [**XD-Violence I3D features**](https://roc-ng.github.io/XD-Violence/)
> 


### Feature Preparation
Please update the following files to replace placeholder paths with the appropriate paths on your local environment.

`list/UCF_Train.list` and `list/UCF_Test.list`

`list/XD_Train.list` and `list/XD_Test.list`


### Train and test the PseudoLabeling-VAD
After the setup, simply run the following command: 

start the visdom for visualizing the training phase


Traing and infer for  UCF-Crime dataset and XD dataset

```
python ./scripts/ucf_main.py
python ./scripts/ucf_infer.py

python ./scripts/xd_main.py
python ./scripts/xd_infer.py
```

## References
We referenced the repos below for the code.

* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [UR-DMU](https://github.com/henrryzh1/UR-DMU)
