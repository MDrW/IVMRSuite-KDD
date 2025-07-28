


data_type=(-1 0 1 2 3)

echo "Training Imitation Learning..."
for d in ${data_type[@]}; do

echo "Running Train ecs_v0_t${d}..."
python train_imitation_model.py --batch_size=16 --epochs=50 --expert_data_path=../results/ecs_imitation/expert_data_merge_placement --save_path=../results/ecs_imitation/results --train_set_id=${d} --model_version=ecs_v0_t${d} --is_dense_items --is_use_full_infos --is_gnn_resnet --is_item_use_pre_acts --is_state_merge_placement

echo "v0-------------------($d)---------------------v0"

done

sleep 120

data_type=(-1 0 1 2 3)
echo "Evaling Imitation Learning..."
for d in ${data_type[@]}; do

echo "Running Eval ecs_v0_t${d}..."
python eval_imitation_model.py --train_set_id=${d} --model_version=ecs_v0_t${d} --model_name=model_best --device=0 --model_save_path=../results/ecs_imitation/results/pretrain --log_save_path=../results/ecs_imitation/results/eval_logs --is_use_full_infos --is_gnn_resnet --is_dense_items --is_item_use_pre_acts --is_state_merge_placement

done
