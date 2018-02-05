#! /bin/sh
#python main.py --task train  --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file run_log
python main.py --task test --weight_file ./saved_weights/weight --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file run_log

