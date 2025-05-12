export EVAL_N_TRAJS=20
export CHECKPOINT_PATH=debug_soft_cl_cta5
export ENV_NAME=franka_sim

declare -A status

mkdir "${CHECKPOINT_PATH}_log"

for (( i = 14000; i <= 14000; i += 1000)); do
    XLA_PYTHON_CLIENT_MEM_FRACTION=.4 /home/qirico/miniconda3/envs/hilserl3/bin/python ../../eval_sim_checkpoints.py \
        --exp_name=$ENV_NAME \
        --checkpoint_path=$CHECKPOINT_PATH \
        --eval_checkpoint_step=$i \
        --eval_n_trajs=$EVAL_N_TRAJS \
        --save_to_txt="${CHECKPOINT_PATH}.txt" \
        >> "${CHECKPOINT_PATH}_log/${i}_2.out" 2>&1
done

echo "Everything done."
