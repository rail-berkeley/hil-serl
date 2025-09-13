export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && 
python ./record_cls_imgs.py "$@" \
    --exp_name=plug_usb \
# python ./record_success_fail.py "$@" \
    
