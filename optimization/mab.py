import sys
import numpy as np
from itertools import product


class MAB:
    def __init__(self, arms_initial_dist_dict, exploit=0.8):
        self.exploit = exploit
        self.arms = arms_initial_dist_dict.keys()
        self.params_space = {arm: arms_initial_dist_dict[arm]['param_space'] for arm in self.arms}
        self.priors = {arm: arms_initial_dist_dict[arm]['initial_prior'] for arm in self.arms}
        
        self.posteriors = self.priors
        self.maps = self._maps_update()
        self.draws = []
        
        self.burnin_arm = 1
        self.burnin2_arm = 1
    
    def _burnin(self):
        burnin_arm = self.burnin_arm
        self.burnin_arm = (self.burnin_arm % 5) + 1
        return 'burnin', burnin_arm
    
    def _exploit(self):
        if np.random.uniform() < self.exploit:
            if self.draws:
                maps = {arm: self.maps[arm]['prob'] for arm in self.arms}
                return 'exploit', max(maps, key=maps.get)
            else:
                return 'exploit', np.random.choice(list(self.arms))
        else:
            return 'explore', np.random.choice(list(self.arms))
    
    def choose_arm(self):
        if len(self.draws) > 1_000:
            return self._exploit()
        return self._burnin()
    
    def _burnin2(self):
        burnin_arm = self.burnin_arm
        burnin2_arm = self.burnin2_arm
        
        self.burnin2_arm = (self.burnin2_arm % 5) + 1
        if self.burnin2_arm == 1:
            self.burnin_arm = (self.burnin_arm % 5) + 1
        return 'burnin', burnin_arm, burnin2_arm
    
    def _exploit2(self):
        if np.random.uniform() < self.exploit:
            if self.draws:
                maps = {arm: self.maps[arm]['prob'] for arm in self.arms}
                return 'exploit', *max(maps, key=maps.get)
            else:
                return 'exploit', *list(np.random.permutation(list(self.arms))).pop()
        else:
            return 'explore', *list(np.random.permutation(list(self.arms))).pop()
    
    def choose_arm2(self):
        if len(self.draws) > 10_000:
            return self._exploit2()
        return self._burnin2()
    
    def likelihood(self, arm, observation):
        likelihood_dict = {}
        for bound in self.params_space[arm]:
            if observation == 1:
                likelihood_dict[bound] = bound
            else:
                likelihood_dict[bound] = 1 - bound
        return likelihood_dict
    
    def likelihood2(self, arm, observation):
        likelihood_dict = {}
        for bound in self.params_space[arm]:
            if observation == 1:
                likelihood_dict[bound] = bound
            else:
                likelihood_dict[bound] = 1 - bound
        return likelihood_dict
    
    def _maps_update(self):
        return {arm: {'params': max(self.posteriors[arm], key=self.posteriors[arm].get),
                      'prob': max(self.posteriors[arm].values())} for arm in self.arms}  # todo: what if there are ties?
    
    def posterior_update(self, arm, observation, exploit):
        self.draws.append((arm, observation, exploit))
        
        l = self.likelihood(arm, observation)
        p = self.priors[arm]
        
        posterior = {}
        posterior_sum = 0
        for bounds in self.params_space[arm]:
            posterior[bounds] = l[bounds] * p[bounds]
            posterior_sum += l[bounds] * p[bounds]
        
        self.posteriors[arm] = {bounds: posterior[bounds] / posterior_sum for bounds in self.params_space[arm]}
        self.priors[arm] = self.posteriors[arm]
        
        self.maps = self._maps_update()

    def posterior_update2(self, arm, observation, exploit):
        self.draws.append((arm, observation, exploit))
        
        l = self.likelihood2(arm, observation)
        p = self.priors[arm]
        
        posterior = {}
        posterior_sum = 0
        for bounds in self.params_space[arm]:
            posterior[bounds] = l[bounds] * p[bounds]
            posterior_sum += l[bounds] * p[bounds]
        
        self.posteriors[arm] = {bounds: posterior[bounds] / posterior_sum for bounds in self.params_space[arm]}
        self.priors[arm] = self.posteriors[arm]
        
        self.maps = self._maps_update()
