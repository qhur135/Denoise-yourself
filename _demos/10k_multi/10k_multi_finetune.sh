#!/bin/bash
export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH
######### finetune
# 디렉토리 경로 설정
data_dir="/home/jihyeon/self_sample_data/my_data/PU1K_raw_meshes/sampling/10000"
result_dir="_result/ablation/300/finetune" 
# epoch 100
# 10k -> 40k
## main 코드 내부도 수정!!! 
#### pretrain weight - main_finetune_multi.py에서 직접 입력함
mode="curvature" 
# name_list=('02954340.40f0c6599f0be25fce01c07526cf2aa4' '03046257.143e665cb61b96751311158f08f2982a' '02828884.2b065fc9d62f1540ad5067eac75a07f7' '02691156.37f2f187a1582704a29fef5d2b2f3d7')

# combined_list=()
# for pc_name in "${name_list[@]}"; do
#     combined_list+=("$data_dir/$pc_name.xyz")
# done

# echo "${combined_list[@]}"

for pc_file in $data_dir/*.xyz; do
# for pc_file in "${combined_list[@]}"; do
	if [ -f "$pc_file" ]; then
		pc_name=`basename "$pc_file" .xyz`\
		pc_file="$data_dir/$pc_name.xyz" # 40,000
		# echo $pc_file
		# --- finetune
		python -u $BASE_PATH/multi_finetune_main.py \
		--lr 0.0005 \
		--iterations 10100 \
		--export-interval 100 \
		--pc "$pc_file" \
		--init-var 0.15 \
		--D1 2497 --D2 2497 \
		--save-path "$result_dir/$pc_name/$mode" \
		--sampling-mode $mode \
		--batch-size 8 \
		--k 10 \
		--p1 0.85 --p2 0.2 \
		--force-normal-estimation \
		--mse

		python -u $BASE_PATH/inference.py \
		--lr 0.001 --name result \
		--iterations 16 \
		--export-interval 100 \
		--pc "$pc_file" \
		--init-var 0.15 \
		--D1 2497 --D2 2497 \
		--save-path "$result_dir/$pc_name/$mode" \
		--generator "$result_dir/$pc_name/$mode/generators/model1200.pt" \
		--sampling-mode $mode \
		--batch-size 8 \
		--k 10 \
		--force-normal-estimation \
		--mse
	fi
done


