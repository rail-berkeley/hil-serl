from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.cube_reach.config import TrainConfig as CubeReachTrainConfig
from experiments.cube_reach2.config import TrainConfig as CubeReach2TrainConfig
from experiments.cube_reach3.config import TrainConfig as CubeReach3TrainConfig
from experiments.franka_sim.config import TrainConfig as FrankaSimTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "cube_reach": CubeReachTrainConfig,
                "cube_reach2": CubeReach2TrainConfig,
                "cube_reach3": CubeReach3TrainConfig,
                "franka_sim": FrankaSimTrainConfig,
               }