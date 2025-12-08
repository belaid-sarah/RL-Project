from .utils import BaseEnv
import random

class LineWorld(BaseEnv):
    def __init__(self, length=30):  # Augmenté à 30
        super().__init__()
        self.length = length
        self.start = 0
        self.goal = length - 1
        self.state = self.start
        self.done = False
        
        # Complexité VRAIMENT accrue
        self.obstacles = self._generate_obstacles()
        self.moving_obstacles = self._generate_moving_obstacles()
        self.traps = self._generate_traps()
        self.rewards = self._generate_rewards()
        self.keys = self._generate_keys()
        self.doors = self._generate_doors()
        self.ice_zones = self._generate_ice_zones()
        self.wind_zones = self._generate_wind_zones()
        self.portals = self._generate_portals()
        
        self.collected_keys = set()
        self.visited = set()
        self.step_count = 0
        self.max_steps = length * 4
        self.total_reward = 0
        self.energy = 100  # Nouvelle mécanique
        
    def _generate_obstacles(self):
        """Obstacles statiques (25%)"""
        obstacles = set()
        num_obstacles = int(self.length * 0.25)
        while len(obstacles) < num_obstacles:
            pos = random.randint(2, self.length-3)
            obstacles.add(pos)
        return obstacles
    
    def _generate_moving_obstacles(self):
        """Obstacles qui bougent (dynamique)"""
        moving = []
        num_moving = max(3, self.length // 8)
        for _ in range(num_moving):
            pos = random.randint(3, self.length-4)
            direction = random.choice([-1, 1])
            speed = random.randint(1, 2)  # Vitesse variable
            moving.append({
                'pos': pos,
                'dir': direction,
                'speed': speed,
                'wait': 0
            })
        return moving
    
    def _generate_traps(self):
        """Pièges (forte pénalité)"""
        traps = set()
        num_traps = self.length // 6
        while len(traps) < num_traps:
            pos = random.randint(2, self.length-3)
            if pos not in self.obstacles:
                traps.add(pos)
        return traps
    
    def _generate_rewards(self):
        """Récompenses à collecter"""
        rewards = {}
        num_rewards = self.length // 5
        for _ in range(num_rewards):
            while True:
                pos = random.randint(2, self.length-3)
                if pos not in self.obstacles and pos not in self.traps:
                    rewards[pos] = random.uniform(1.0, 3.0)
                    break
        return rewards
    
    def _generate_keys(self):
        """Clés nécessaires pour ouvrir des portes"""
        keys = {}
        num_keys = min(3, self.length // 4)
        for i in range(num_keys):
            attempts = 0
            while attempts < 50:
                if self.length > 6:
                    pos = random.randint(2, max(2, self.length // 2))
                else:
                    pos = random.randint(2, max(2, self.length - 1))
                if pos not in self.obstacles and pos not in self.traps:
                    keys[pos] = f"key_{i}"
                    break
                attempts += 1
        return keys
    
    def _generate_doors(self):
        """Portes qui bloquent sans clé"""
        doors = {}
        for i in range(min(3, self.length // 4)):
            pos = (self.length // 4) * (i + 1)
            if pos < self.length and pos not in self.obstacles:
                doors[pos] = f"key_{i}"
        return doors
    
    def _generate_ice_zones(self):
        """Zones de glace : glisse plusieurs cases"""
        ice = set()
        num_zones = max(1, min(2, self.length // 10))
        for _ in range(num_zones):
            if self.length > 8:
                start = random.randint(2, max(2, self.length-6))
            else:
                start = random.randint(2, max(2, self.length-2))
            length = random.randint(1, min(3, self.length - start))
            for i in range(length):
                if start + i < self.length:
                    ice.add(start + i)
        return ice
    
    def _generate_wind_zones(self):
        """Zones de vent (probabiliste)"""
        wind = {}
        num_zones = self.length // 7
        for _ in range(num_zones):
            pos = random.randint(3, self.length-4)
            if pos not in self.obstacles:
                wind[pos] = {
                    'direction': random.choice([-1, 1]),
                    'strength': random.uniform(0.5, 1.0)  # Probabilité d'être affecté
                }
        return wind
    
    def _generate_portals(self):
        """Portails bidirectionnels"""
        portals = {}
        if self.length > 6:
            num_pairs = max(1, min(2, self.length // 12))
            for _ in range(num_pairs):
                attempts = 0
                while attempts < 50:
                    pos1 = random.randint(2, max(2, self.length // 2))
                    pos2 = random.randint(max(self.length // 2, 2), max(2, self.length-2))
                    if pos1 != pos2 and pos1 not in self.obstacles and pos2 not in self.obstacles:
                        portals[pos1] = pos2
                        portals[pos2] = pos1
                        break
                    attempts += 1
        return portals
    
    def _update_moving_obstacles(self):
        """Met à jour les obstacles mobiles"""
        for obs in self.moving_obstacles:
            if obs['wait'] > 0:
                obs['wait'] -= 1
                continue
            
            new_pos = obs['pos'] + obs['dir'] * obs['speed']
            
            # Rebondir aux bords
            if new_pos <= 1 or new_pos >= self.length - 2:
                obs['dir'] *= -1
                obs['wait'] = random.randint(0, 2)  # Pause aléatoire
            else:
                obs['pos'] = new_pos

    def reset(self):
        self.state = self.start
        self.done = False
        self.step_count = 0
        self.total_reward = 0
        self.energy = 100
        self.collected_keys = set()
        self.visited = set()
        self.visited.add(self.start)
        
        # Régénérer
        self.obstacles = self._generate_obstacles()
        self.moving_obstacles = self._generate_moving_obstacles()
        self.traps = self._generate_traps()
        self.rewards = self._generate_rewards()
        self.keys = self._generate_keys()
        self.doors = self._generate_doors()
        self.ice_zones = self._generate_ice_zones()
        self.wind_zones = self._generate_wind_zones()
        self.portals = self._generate_portals()
        
        return self._get_state()
    
    def _get_state(self):
        """État enrichi pour l'agent"""
        return {
            'position': self.state,
            'energy': self.energy,
            'keys': tuple(sorted(self.collected_keys)),
            'nearby_obstacles': self._get_nearby_obstacles(),
            'steps': self.step_count
        }
    
    def _get_nearby_obstacles(self):
        """Détecte obstacles mobiles proches (observation partielle)"""
        nearby = []
        for obs in self.moving_obstacles:
            if abs(obs['pos'] - self.state) <= 3:
                nearby.append(obs['pos'] - self.state)  # Position relative
        return tuple(sorted(nearby))

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}
        
        self.step_count += 1
        self._update_moving_obstacles()
        
        old_state = self.state
        
        # Actions: 0=gauche, 1=droite, 2=rester, 3=sauter, 4=sprint (3 cases, coûte énergie)
        energy_cost = 1
        
        if action == 0:  # Gauche
            self.state = max(0, self.state - 1)
        elif action == 1:  # Droite
            self.state = min(self.length - 1, self.state + 1)
        elif action == 2:  # Rester (regagne énergie)
            self.energy = min(100, self.energy + 5)
            energy_cost = 0
        elif action == 3:  # Sauter
            self.state = min(self.length - 1, self.state + 2)
            energy_cost = 2
        elif action == 4:  # Sprint
            if self.energy >= 10:
                self.state = min(self.length - 1, self.state + 3)
                energy_cost = 10
            else:
                energy_cost = 0  # Pas assez d'énergie
        else:
            raise ValueError("Action invalide")
        
        self.energy = max(0, self.energy - energy_cost)
        reward = -0.1  # Coût par step
        
        # Vérifier obstacles statiques
        if self.state in self.obstacles:
            self.state = old_state
            reward = -1.5
        
        # Vérifier obstacles mobiles
        for obs in self.moving_obstacles:
            if abs(self.state - obs['pos']) < 1:
                reward -= 3.0
                self.energy = max(0, self.energy - 10)
        
        # Vérifier portes
        if self.state in self.doors:
            required_key = self.doors[self.state]
            if required_key not in self.collected_keys:
                self.state = old_state
                reward = -2.0
        
        # Zone de glace : glisse automatiquement
        if self.state in self.ice_zones and self.state != old_state:
            slide_distance = random.randint(2, 4)
            direction = 1 if self.state > old_state else -1
            self.state = max(0, min(self.length-1, self.state + direction * slide_distance))
            reward -= 0.5  # Pénalité pour perte de contrôle
        
        # Zone de vent (probabiliste)
        if self.state in self.wind_zones:
            wind = self.wind_zones[self.state]
            if random.random() < wind['strength']:
                self.state = max(0, min(self.length-1, self.state + wind['direction']))
        
        # Portails
        if self.state in self.portals:
            self.state = self.portals[self.state]
            reward += 0.5
        
        # Exploration
        if self.state not in self.visited:
            self.visited.add(self.state)
            reward += 0.2
        
        # Pièges
        if self.state in self.traps:
            reward -= 4.0
            self.energy = max(0, self.energy - 20)
        
        # Clés
        if self.state in self.keys:
            key_name = self.keys[self.state]
            if key_name not in self.collected_keys:
                self.collected_keys.add(key_name)
                reward += 2.0
                del self.keys[self.state]
        
        # Récompenses
        if self.state in self.rewards:
            reward += self.rewards[self.state]
            del self.rewards[self.state]
        
        # Goal
        if self.state == self.goal:
            # Doit avoir toutes les clés
            if len(self.collected_keys) == 3:
                efficiency = max(0, (self.max_steps - self.step_count) / self.max_steps * 10)
                reward = 30.0 + efficiency
                self.done = True
            else:
                reward = -5.0  # Arrivé sans toutes les clés
        
        # Timeout ou plus d'énergie
        if self.step_count >= self.max_steps or self.energy <= 0:
            self.done = True
            reward = -15.0
        
        self.total_reward += reward
        
        return self._get_state(), reward, self.done, {
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'energy': self.energy,
            'keys_collected': len(self.collected_keys),
            'exploration': len(self.visited) / self.length
        }

    def sample_action(self):
        return random.randint(0, 4)

    def n_actions(self):
        return 5