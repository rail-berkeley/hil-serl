from .continuous.bc import BCAgent
from .continuous.sac import SACAgent
from .continuous.sac_hybrid_single import SACAgentHybridSingleArm
from .continuous.sac_hybrid_dual import SACAgentHybridDualArm

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
    "sac_hybrid_single": SACAgentHybridSingleArm,
    "sac_hybrid_dual": SACAgentHybridDualArm,
}
