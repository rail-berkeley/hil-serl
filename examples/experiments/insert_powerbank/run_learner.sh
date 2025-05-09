export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_ENTITY=724183484zz@gmail.com
export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
python ../../train_rlpd.py "$@" \
    --exp_name=insert_powerbank \
    --checkpoint_path=first_run \
    --learner \
    --demo_path="demo_data/insert_powerbank_10_demos_2025-04-30_20-44-43.pkl" \



