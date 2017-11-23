#! /bin/sh
# train
#python main.py --model DeepTTE --batch_size 400 --epochs 100 --log_file deeptte_log --pooling_method attention --kernel_size 3 --alpha 0.3

# test
python main.py --task test --model DeepTTE --batch_size 10 --weight_file ./saved_weights/model_weight --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3

