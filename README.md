This project is the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***.
We provide the complete version of code and part of sample data in Beijing. You can replace the sample data with your own data easily. See the samples in data/ for more details.
We further provide a pre-trained model in saved_weights/ folder.

# Usage:

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
* log_file: the path of log file

The training log will be recorded to log_file

## Model Evaluation

### Parameters:
* weight_file: the path of model weight
* result_file: the path to save the result

## Example:
```
Train:
python main.py --model DeepTTE --batch_size 400 --epochs 100 --log_file deeptte_log --pooling_method attention --kernel_size 3 --alpha 0.3

Test:
python main.py --task test --model DeepTTE --batch_size 10 --weight_file ./saved_weights/model_weight --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3
```
