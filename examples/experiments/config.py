from abc import abstractmethod
from typing import List

class DefaultTrainingConfig:
    """Default training configuration. """

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.97

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    
    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
