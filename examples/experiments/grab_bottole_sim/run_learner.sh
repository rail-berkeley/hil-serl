export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=grab_bottole_sim \
    --checkpoint_path=first_run \
    --learner \


    #--demo_path=... \
