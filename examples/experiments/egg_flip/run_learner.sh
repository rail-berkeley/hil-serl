export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=egg_flip \
    --checkpoint_path=../../experiments/egg_flip/debug \
    --demo_path=... \
    --learner \