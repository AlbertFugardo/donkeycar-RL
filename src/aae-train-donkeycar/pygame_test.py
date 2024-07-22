import os
import pygame

os.environ['SDL_AUDIODRIVER'] = 'directx'
os.environ["SDL_VIDEODRIVER"]="x11"
pygame.init()
# Define the background colour 
# using RGB color coding. 
(width, height) = (300, 200)
screen = pygame.display.set_mode((width, height))
pygame.display.flip()
while True:
    pass
"""
# Fill the background colour to the screen 
screen.fill(background_colour) 
  
# Update the display using flip 
pygame.display.flip() 
  
# Variable to keep our game loop running 
running = True
  
# game loop 
while running: 
    print("running")
# for loop through the event queue   
    for event in pygame.event.get(): 
      
        # Check for QUIT event       
        if event.type == pygame.QUIT: 
            running = False
            
"""