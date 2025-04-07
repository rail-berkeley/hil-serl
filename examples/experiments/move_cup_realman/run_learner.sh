export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_ENTITY=724183484zz@gmail.com
export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
python ../../train_rlpd.py "$@" \
    --exp_name=grab_bottole \
    --checkpoint_path=first_run \
    --learner \
    --demo_path="demo_data/grab_bottole_10_demos_2025-03-29_10-19-37.pkl" \



