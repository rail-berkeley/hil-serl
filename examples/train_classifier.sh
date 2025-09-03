export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && 
python ./train_reward_classifier.py"$@" \
    --exp_name=plug_cable \