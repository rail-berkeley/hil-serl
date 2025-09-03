# from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
# from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
# from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
# from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
#from experiments.object_handover_sim.config import (
#    TrainConfig as ObjectHandoverSimTrainConfig,
#)

# from experiments.object_handover_agilex.config import (
#    TrainConfig as ObjectHandoverAgilexTrainConfig,
# )

from experiments.plugcable.config import TrainConfig as plug_cableTrainingConfig

CONFIG_MAPPING = {
    # "ram_insertion": RAMInsertionTrainConfig,
    # "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    # "object_handover": ObjectHandoverTrainConfig,
    # "egg_flip": EggFlipTrainConfig,
    #"object_handover_sim": ObjectHandoverSimTrainConfig,
    # "object_handover_agilex": ObjectHandoverAgilexTrainConfig,
    "plug_cable": plug_cableTrainingConfig
    #"grab_bottole_sim": GrabBottoleSimTrainConfig,
}
