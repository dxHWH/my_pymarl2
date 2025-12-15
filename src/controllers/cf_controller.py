from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
import numpy as np

# This multi-agent controller shares parameters between agents
class CFController(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CFController, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # if self.q_self != None:
        #     print("qvals: ", qvals.shape)
        #     print("q_self: ", self.q_self.shape)

        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        # if self.q_self_out != None:
        #     mask_q_self = self.q_self_out[bs]
        #     mask_q_self[avail_actions[bs] == 0.0] = -float("inf")

        #     chosen_actions_self = self.action_selector.pick_random*self.action_selector.random_actions + \
        #     (1-self.action_selector.pick_random)*mask_q_self.max(dim=2)[1]
        #     self.q_self = chosen_actions_self
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, agent_inter, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if self.agent.q_self != None:
            self.q_self_out = self.agent.q_self
        # agent_outs, agent_inter, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        self.inter = agent_inter
        return agent_outs