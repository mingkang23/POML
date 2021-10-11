# Training # ratio means missing ratio (5:0%, 4:20%, 3:40%, 2:60%, 1: 20%)
for seed in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed}
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed} --ratio 2 --max_epoch 20 --hyper_semi 0.5 --th 0.95
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed} --ratio 1 --max_epoch 70 --hyper_semi 0.4 --th 0.93
#CUDA_VISIBLE_DEVICES=1 python mlc_cub_ft_final_v3.py --seed ${seed} --ratio 2 --max_epoch 20 --hyper_semi 0.5 --th 0.95
#CUDA_VISIBLE_DEVICES=1 python mlc_cub_ft_final_v4.py --seed ${seed} --ratio 1 --max_epoch 70 --hyper_semi 0.4 --th 0.93
done

