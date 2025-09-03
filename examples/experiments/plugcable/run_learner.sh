export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_ENTITY=724183484zz@gmail.com
export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
python ../../train_rlpd.py "$@" \
    --exp_name=plug_cable \
    --checkpoint_path=first_cable \
    --learner \
    # --demo_path="/home/amigo/ChaoW/ws/hil-serl/examples/demo_data/grab_earphone_20_demos_2025-08-04_11-10-27.pkl" \



