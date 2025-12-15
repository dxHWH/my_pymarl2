import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DMAQ_QattenMixer(nn.Module):
    def __init__(self, args, beta_eta=0.01,gamma=0.99):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.beta = args.beta
        self.register_buffer('beta', th.tensor(0))  
        self.beta_eta = beta_eta                   
        self.gamma = gamma                              
        self.lambda_update_steps = 0                       

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args)
        # self.msg_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i,q_self_out,max_q_self):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)
        q_self_out = q_self_out.view(-1, self.n_agents) 
        max_q_self = max_q_self.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).clone().detach()
        adv_q_self = (q_self_out - max_q_self).view(-1, self.n_agents).clone().detach()
        adv_m2 = adv_q_self - adv_q
        adv_m1 = max_q_self - max_q_i
        combined = th.cat([adv_m1, adv_m2], dim=-1)

        adv_w_final,adv_w_msg2= self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)
        adv_w_msg2 = adv_w_msg2.view(-1, self.n_agents)
        # adv_w_msg1 = adv_w_msg1.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot1 = th.sum(adv_q * (adv_w_final - 1.), dim=1)
            adv_tot2 = th.sum(adv_q_self * (adv_w_msg2 - 1.), dim=1)
        else:
            adv_tot1 = th.sum(adv_q * adv_w_final, dim=1)
            adv_tot2 = th.sum(adv_q_self * adv_w_msg2, dim=1)

        return adv_tot1 + self.args.beta * adv_tot2

    def calc(self, agent_qs, states, actions=None, max_q_i=None,q_self_out=None,max_q_self=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i,q_self_out,max_q_self)
            return adv_tot

    def update_beta(self, M1, M2, Q_values):
            delta_group = th.mean(M1 + M2)    
            maxQ = th.max(Q_values)
            minQ = th.min(Q_values)
            q_range = self.gamma * (maxQ - minQ) + 1e-8

            delta_beta = self.beta_eta * (delta_group / q_range)
            
            beta_prime = self.beta + delta_beta
            
            beta_prime_clipped = th.clamp(beta_prime, 0.0, 1.0)
            
            self.beta_update_steps += 1
            self.beta = (beta_prime_clipped.detach() + 
                        self.lambda_update_steps * self.beta) / (self.beta_update_steps + 1)
        
    def forward(self, agent_qs, states, actions=None, max_q_i=None,q_self_out=None,max_q_self=None, is_v=False):
        bs = agent_qs.size(0)
        #agent_qs.retain_grad()
        #global_Grad.x = agent_qs
        # print("Agent Qs: ", agent_qs.shape)
    
        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        # print("W_final: ", w_final.shape)
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            if max_q_i is not None:
                # print("max_q_i: ", max_q_i.shape)
                max_q_i = max_q_i.view(-1, self.n_agents)
                max_q_i = w_final * max_q_i + v
                if q_self_out is not None:
                    # print("q_self_out: ", q_self_out.shape)
                    # print("max_q_self: ", max_q_sel f.shape)
                    q_self_out = q_self_out.view(-1, self.n_agents)
                    q_self_out = w_final * q_self_out + v
                    max_q_self = max_q_self.view(-1, self.n_agents)
                    max_q_self = w_final * max_q_self + v
            else:
                print("max_q_i is None, skipping view and multiplication")
                
        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, q_self_out = q_self_out,max_q_self=max_q_self,is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies


class Qatten_Weight(nn.Module):
    def __init__(self, args):
        super(Qatten_Weight, self).__init__()

        self.name = 'cf_weight'
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.mixing_embed_dim
        # print("Embedding dim: ", self.embed_dim)
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)  # query
            if self.args.nonlinear:  # add qs
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                              nn.ReLU(),
                                              nn.Linear(hypernet_embed, self.n_head))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        # print("States shape: ", states.shape)
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = th.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs, head_num, self.n_agents)
            head_attend *= w_head

        head_attend = th.sum(head_attend, dim=1)

        if not self.args.state_bias:
            v *= 0.
 
        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies


class DMAQ_SI_Weight(nn.Module):
    def __init__(self, args):
        super(DMAQ_SI_Weight, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        self.key_extractors2 = nn.ModuleList()
        self.agents_extractors2 = nn.ModuleList()
        self.action_extractors2 = nn.ModuleList()
        self.meg1_weight_extractor = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention
            if getattr(args, "adv_hypernet_layers", 1) == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 2:
                self.meg1_weight_extractor.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1))) 
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.key_extractors2.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1))) 
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
                self.agents_extractors2.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors2.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 3:
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(adv_hypernet_embed, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                                                            nn.ReLU(),
                                                            nn.Linear(adv_hypernet_embed, self.n_agents)))  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)  
        data = th.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        all_head_key2 = [k_ext(states) for k_ext in self.key_extractors2]
        all_head_agents2 = [k_ext(states) for k_ext in self.agents_extractors2]
        all_head_action2 = [sel_ext(data) for sel_ext in self.action_extractors2]
        # all_head_meg1 = [sel_ext(states) for sel_ext in self.meg1_weight_extractor] 

        head_attend_weights = []
        head_attend_weights2 = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key2, all_head_agents2, all_head_action2):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights2.append(weights)
        
        # head_msg1_weight = [th.abs(one_head_meg1) + 1e-10 for one_head_meg1 in all_head_meg1]
        # head_msg1 = th.stack(head_msg1_weight, dim=1)
        # head_msg1 = head_msg1.view(-1, self.num_kernel, self.n_agents)
        # head_msg1 = th.sum(head_msg1, dim=1)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)

        head_attend2 = th.stack(head_attend_weights2, dim=1)
        head_attend2 = head_attend2.view(-1, self.num_kernel, self.n_agents)
        head_attend2 = th.sum(head_attend2, dim=1)

        return head_attend, head_attend2
    # , head_msg1