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
from experiments.grab_bottole.config import TrainConfig as GrabBottoleAgilexTrainConfig
#from experiments.grab_bottole_sim.config import TrainConfig as GrabBottoleSimTrainConfig
from experiments.move_cup_realman.config import TrainConfig as MoveCupRealmanTrainConfig
from experiments.lift_cup.config import TrainConfig as LiftCupTrainConfig
from experiments.place_power_bank.config import TrainConfig as PlacePowerBankTrainConfig
from experiments.pick_pen.config import TrainConfig as PickPenTrainConfig
from experiments.insert_battery.config import TrainConfig as InsertBatteryTrainConfig
from experiments.insert_powerbank.config import TrainConfig as InsertPowerbankTrainConfig

CONFIG_MAPPING = {
    # "ram_insertion": RAMInsertionTrainConfig,
    # "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    # "object_handover": ObjectHandoverTrainConfig,
    # "egg_flip": EggFlipTrainConfig,
    #"object_handover_sim": ObjectHandoverSimTrainConfig,
    # "object_handover_agilex": ObjectHandoverAgilexTrainConfig,
    "grab_bottole": GrabBottoleAgilexTrainConfig,
    "move_cup": MoveCupRealmanTrainConfig,
    "lift_cup": LiftCupTrainConfig,
    "place_power_bank": PlacePowerBankTrainConfig,
    "pick_pen": PickPenTrainConfig,
    "insert_battery": InsertBatteryTrainConfig,
    "insert_powerbank": InsertPowerbankTrainConfig
    #"grab_bottole_sim": GrabBottoleSimTrainConfig,
}
