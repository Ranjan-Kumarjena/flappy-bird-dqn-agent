import gymnasium as gym
import numpy as np
import pygame
import flappy_bird_gymnasium

# creating our env

env=gym.make("FlappyBird-v0",render_mode="human")
state,info=env.reset()
done=False
# Initialize Pygame keybord

pygame.init()

# Gym has has already created a window 
screen=pygame.display.get_surface()

while not done:
    action=0 # default --> 0 is no flap & 1 is flap 
      
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # flap 
                
    state , reward , done , truncated , info = env.step(action)
    env.render()
    
env.close()
pygame.quite()
