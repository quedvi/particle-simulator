import numpy as np
from scipy import spatial
import pygame as pg
from pygame.constants import QUIT
from time import sleep


screen_b = screen_h = 1000
dt = 1

particles_count = 2000
kind = 5

kind_color = [
    (255, 0, 0), 
    (0, 255, 0), 
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

# interaction_matrix = np.zeros((kind, kind))
# interaction_matrix = [
#     [ +3,  -2, -3,  -1, +3, +5],
#     [ +2,  -1, -3,  -1, +3, +5],
#     [ -3,  -3, -1,  +1, +3, +5],
#     [ -1,  -1, -1,  -1, +3, +5],
#     [ -5,  +5, -5,  +5, +3, +5],
#     [ -5,  -5, +5,  +5, -5, +5]
# ] * 2
s = -5
a = 2
b = -2
z = 0
# interaction_matrix = [
#     [ a, a, b, s,  z, z],
#     [ b, a, a, s,  z, z],
#     [ a, b, a, s,  z, z],
#     [ s, s, s, s,  z, z],
#     [ z, z, z, z,  a, z],
#     [ z, z, z, z,  z, a]
# ] * 3


a = 2
b = -2
interaction_matrix = [
    [ a, b, 0, 0,  s, z],
    [ 0, a, b, 0,  s, z],
    [ 0, 0, a, b,  s, z],
    [ 0, 0, 0, a,  s, z],
    [ s, s, s, s,  s, z],
    [ z, z, z, z,  z, a]
] * 3


# interaction_matrix = [
#     [ a, a, s, s,  z, z],
#     [ b, a, s, s,  z, z],
#     [ s, s, s, s,  z, z],
#     [ s, s, s, s,  z, z],
#     [ z, z, z, z,  a, z],
#     [ z, z, z, z,  z, a]
# ] * 3


particle_radius = 3

interaction_radius = 70 #30
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
    attraction_max = interaction_matrix[kind_i][kind_j]
    return min(attraction_max/distance * 10, 20)
    
    
    # repulsion_radius = 5
    # repulsion_strength = -5
    # attraction_max_radius = 20
    # attraction_radius = 30
    # attraction_max = interaction_matrix[kind_i][kind_j]
    
    # if distance < repulsion_radius:
    #     return distance * repulsion_strength/repulsion_radius - repulsion_strength
    # if distance < attraction_max_radius:
    #     return (distance - repulsion_radius) * attraction_max/(attraction_max_radius - repulsion_radius)
    # if distance < attraction_radius:
    #     return (attraction_radius - distance) * attraction_max/(attraction_radius - attraction_max_radius)
    # return 0


def interaction(particles : np.ndarray, kinds : np.ndarray, i : int, j : int):      
    distance = np.linalg.norm(particles[i] - particles[j])
    
    if distance < 10: #(repulsion)
        a = -10.0
    else:
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
        
        kd_tree = spatial.KDTree(particles)
        # interactions = kd_tree.query_pairs(interaction_radius)
        interactions = kd_tree.query_ball_point(particles, interaction_radius, workers=-1) # increase workers
        for i in range(particles_count):
            for j in interactions[i]:
                if i == j:
                    continue
                velocities[i] += interaction(particles, kinds, i, j) * 0.15
            # velocities[i] += interaction(particles, kinds, i, j) * 0.033
            # velocities[j] += interaction(particles, kinds, j, i) * 0.033
                
        velocities *= 0.75 # friction
        particles += velocities
        
        for i in range(particles_count):
            if particles[i][0] < 0 or particles[i][0] >= screen_b:
                velocities[i][0] = -velocities[i][0]
            if particles[i][1] < 0 or particles[i][1] >= screen_h:
                velocities[i][1] = -velocities[i][1]
                
        # particles %= np.array([screen_b, screen_h])
        
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
