This project is the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***.

# Usage:
The data files are in ***json*** folder. We provide several sample trajectories in Beijing Dataset. To use your own dataset, make sure the corresponding ***.json*** and ***.lens*** files are located in json folder.

See the sample trajectories for formatting information.

## Model Training
python train.py
### Parameters:

* model: The model to train (e.g., AttrTTE, DeepTTE, see ***models*** folder)
* task: train/test
* batch_size: the batch_size to train, default 400
* epochs: the epoch to train, default 100
* kernel_size: the kernel size of Geo-Conv, only used when the model contains the Geo-conv part
* pooling_method: attention/mean
* alpha: the weight of combination in multi-task learning
* driver_off: if the driver_off = 1, then all the driver ID is reset to 0, this option is used to show the effectiveness of the driverID embedding.
* week_off: similar with driver_off, used for the weekID embedding
* road_off: whether to use the road information

```
Example:
python main.py --model DeepTTE --batch_size 400 --epochs 100 --kernel_size 3 --pooling_method attention --alpha 0.3 --log_file log_deepTTE
```

