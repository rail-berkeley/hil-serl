export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_ENTITY=zjq19990202@gmail.com
export WANDB_API_KEY=0a88034024ec899c79a9809b84c3a4f01c8b191d
python ../../train_rlpd.py "$@" \
    --exp_name=plug_usb \
    --checkpoint_path=first_plug \
    --learner \
    # --demo_path="/home/amigo/ChaoW/ws/hil-serl/examples/demo_data/grab_earphone_20_demos_2025-08-04_11-10-27.pkl" \



