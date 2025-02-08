import numpy as np
from scipy import spatial
import pygame as pg
from pygame.constants import QUIT
from time import sleep


screen_b = screen_h = 750
dt = 1

particles_count = 3000
kind = 6

kind_color = [
    (255, 0, 0), 
    (0, 255, 0), 
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

# interaction_matrix = np.zeros((kind, kind))
interaction_matrix = [
    [+10, -5,  -5, +10, +3, +5],
    [ +5, +10, +5, +10, +3, +5],
    [ +5,  +5, -5, +10, +3, +5],
    [ +5,  +5, +5,  -5, +3, +5],
    [ -5,  +5, -5,  +5, +3, +5],
    [ -5,  -5, +5,  +5, -5, +5]
]


particle_radius = 3

interaction_radius = 30

draw_interactions = False
interaction_line_width = 3


def draw_particles(screen, particles, kinds):
    for i in range(particles_count):
        pg.draw.circle(
            screen, 
            kind_color[kinds[i]], 
            (int(particles[i][0]), int(particles[i][1])), 
            particle_radius
        )


def get_force(distance : float, kind_i : int, kind_j : int):
    repulsion_radius = 5
    repulsion_strength = -5
    attraction_max_radius = 20
    attraction_radius = 30
    attraction_max = interaction_matrix[kind_i][kind_j]
    
    if distance < repulsion_radius:
        return distance * repulsion_strength/repulsion_radius - repulsion_strength
    if distance < attraction_max_radius:
        return (distance - repulsion_radius) * attraction_max/(attraction_max_radius - repulsion_radius)
    if distance < attraction_radius:
        return (attraction_radius - distance) * attraction_max/(attraction_radius - attraction_max_radius)
    return 0


def interaction(particles : np.ndarray, kinds : np.ndarray, i : int, j : int):      
    distance = np.linalg.norm(particles[i] - particles[j])
    a = get_force(distance, kinds[i], kinds[j])
    direction = particles[j] - particles[i]
    direction /= np.linalg.norm(direction)
    return a * direction


def main():
    pg.init()
    pg.font.init()
    game_font = pg.font.SysFont('Roboto', 20)
    
    
    pg.display.set_caption("Particles")
    screen = pg.display.set_mode([screen_b, screen_h])
    
    particles = np.random.rand(particles_count, 2)
    particles *= np.array([screen_b, screen_h])
    
    velocities = np.zeros((particles_count, 2))
    kinds = np.random.randint(0, kind, particles_count)
    
    while True:
        screen.fill((0, 0, 0))
        next_particles = particles.copy()
        
        kd_tree = spatial.KDTree(particles)
        interactions = kd_tree.query_pairs(interaction_radius)
        for i, j in interactions:
            velocities[i] += interaction(particles, kinds, i, j) * 0.1
            velocities[j] += interaction(particles, kinds, j, i) * 0.1
                
        velocities *= 0.9 # friction
        next_particles += velocities
        next_particles %= np.array([screen_b, screen_h])
        
        particles = next_particles
        
        draw_particles(screen, particles, kinds)
        if draw_interactions:
            textsurface = game_font.render(f"{len(interactions)} interactions found", False, (255, 255, 255))
            screen.blit(textsurface,(0,0))
            for i, j in interactions:
                pg.draw.line(screen, (255, 0, 0), (int(particles[i][0]), int(particles[i][1])), (int(particles[j][0]), int(particles[j][1])), width=3)
        
        pg.display.flip()
        sleep(0.01)
        
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                return
    

if __name__ == "__main__":
    main()
