from .utils import BaseEnv
import random

class GridWorld(BaseEnv):
    def __init__(self, width=10, height=10):
        super().__init__()
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width-1, height-1)
        self.state = self.start
        self.done = False
        
        # Complexité accrue pour RL
        self.obstacles = self._generate_obstacles()
        self.traps = self._generate_traps()
        self.rewards_cells = self._generate_rewards()
        self.moving_traps = self._generate_moving_traps()
        self.visited = set()
        self.step_count = 0
        self.max_steps = width * height * 2
        self.total_reward = 0
        
    def _generate_obstacles(self):
        """Génère des murs complexes (25-30% de la grille)"""
        obstacles = set()
        num_obstacles = int(self.width * self.height * 0.28)
        
        # Créer des "murs" en lignes pour forcer des chemins
        for i in range(self.width // 3):
            wall_x = random.randint(2, self.width-3)
            wall_length = random.randint(3, self.height-2)
            start_y = random.randint(0, self.height - wall_length)
            for y in range(start_y, start_y + wall_length):
                pos = (wall_x, y)
                if pos != self.start and pos != self.goal:
                    obstacles.add(pos)
        
        # Ajouter des obstacles aléatoires
        while len(obstacles) < num_obstacles:
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        
        return obstacles
    
    def _generate_traps(self):
        """Génère des pièges (forte pénalité)"""
        traps = set()
        num_traps = max(self.width // 2, 5)
        while len(traps) < num_traps:
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if pos != self.start and pos != self.goal and pos not in self.obstacles:
                traps.add(pos)
        return traps
    
    def _generate_rewards(self):
        """Génère des récompenses à collecter"""
        rewards = {}
        num_rewards = max(self.width // 2, 5)
        for _ in range(num_rewards):
            while True:
                pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                if (pos != self.start and pos != self.goal and 
                    pos not in self.obstacles and pos not in self.traps):
                    rewards[pos] = random.uniform(0.3, 1.0)
                    break
        return rewards
    
    def _generate_moving_traps(self):
        """Génère des pièges qui bougent (complexité dynamique)"""
        moving = []
        num_moving = max(1, min(2, self.width // 5))
        for _ in range(num_moving):
            attempts = 0
            while attempts < 100:
                pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                if (pos != self.start and pos != self.goal and 
                    pos not in self.obstacles and pos not in self.traps):
                    direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                    moving.append({'pos': pos, 'dir': direction})
                    break
                attempts += 1
        return moving
    
    def _update_moving_traps(self):
        """Déplace les pièges mobiles"""
        for trap in self.moving_traps:
            x, y = trap['pos']
            dx, dy = trap['dir']
            new_x, new_y = x + dx, y + dy
            
            # Rebondir sur les bords ou obstacles
            if (new_x < 0 or new_x >= self.width or 
                new_y < 0 or new_y >= self.height or
                (new_x, new_y) in self.obstacles):
                trap['dir'] = (-dx, -dy)
            else:
                trap['pos'] = (new_x, new_y)

    def reset(self):
        self.state = self.start
        self.done = False
        self.step_count = 0
        self.total_reward = 0
        self.visited = set()
        self.visited.add(self.start)
        
        # Régénérer l'environnement
        self.obstacles = self._generate_obstacles()
        self.traps = self._generate_traps()
        self.rewards_cells = self._generate_rewards()
        self.moving_traps = self._generate_moving_traps()
        
        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0, True, {}
        
        self.step_count += 1
        
        # Mettre à jour les pièges mobiles tous les 2 steps
        if self.step_count % 2 == 0:
            self._update_moving_traps()
        
        x, y = self.state
        
        # Déplacement
        if action == 0:  # Haut
            y = max(0, y-1)
        elif action == 1:  # Bas
            y = min(self.height-1, y+1)
        elif action == 2:  # Gauche
            x = max(0, x-1)
        elif action == 3:  # Droite
            x = min(self.width-1, x+1)
        else:
            raise ValueError("Action invalide")
        
        new_state = (x, y)
        reward = -0.04  # Coût par step (encourager l'efficacité)
        
        # Vérifier les obstacles
        if new_state in self.obstacles:
            new_state = self.state  # Reste sur place
            reward = -0.75
        else:
            self.state = new_state
            
            # Bonus pour explorer de nouvelles cellules
            if self.state not in self.visited:
                self.visited.add(self.state)
                reward += 0.1
        
        # Vérifier les pièges statiques
        if self.state in self.traps:
            reward += -2.0
        
        # Vérifier les pièges mobiles
        for trap in self.moving_traps:
            if self.state == trap['pos']:
                reward += -3.0
                break
        
        # Vérifier les récompenses bonus
        if self.state in self.rewards_cells:
            reward += self.rewards_cells[self.state]
            del self.rewards_cells[self.state]
        
        # Vérifier le goal
        if self.state == self.goal:
            # Bonus si arrivé rapidement
            efficiency_bonus = max(0, (self.max_steps - self.step_count) / self.max_steps * 5)
            reward = 20.0 + efficiency_bonus
            self.done = True
        
        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True
            reward = -10.0
        
        self.total_reward += reward
        
        return self.state, reward, self.done, {
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'visited_cells': len(self.visited),
            'rewards_collected': self.width // 2 - len(self.rewards_cells),
            'exploration_ratio': len(self.visited) / (self.width * self.height)
        }

    def sample_action(self):
        return random.randint(0, 3)

    def n_actions(self):
        return 4