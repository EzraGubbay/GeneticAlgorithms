import pygame
import random

# Initialize PyGame
pygame.init()
N = 100
CELL_SIZE = 10
screen = pygame.display.set_mode((N * CELL_SIZE, N * CELL_SIZE))
pygame.display.set_caption("Magic Cube Visualizer")
clock = pygame.time.Clock()


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    # Draw the grid
    for row in range(N):
        for col in range(N):
            color = random_color()
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            pygame.draw.rect(screen, color, rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
