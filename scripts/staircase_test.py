import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2AFC Staircase Procedure")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Parameters
square_size = 100
step_size = 10


# Staircase procedure
def staircase(response, current_value):
    if response:
        return max(current_value - step_size, 0)
    else:
        return min(current_value + step_size, 255)


def draw_square(position, color):
    pygame.draw.rect(screen, color, (*position, square_size, square_size))


def main():
    red_intensity = 128
    position_left = ((width // 4) - (square_size // 2), height // 2 - square_size // 2)
    position_right = ((3 * width // 4) - (square_size // 2), height // 2 - square_size // 2)

    # Generate colors initially
    color_left = (red_intensity + random.choice([-1, 1]) * step_size, 0, 0)
    color_right = (red_intensity, 0, 0)

    while True:
        screen.fill(WHITE)

        draw_square(position_left, color_left)
        draw_square(position_right, color_right)

        pygame.display.flip()

        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                if position_left[0] <= mouse_x <= position_left[0] + square_size and position_left[1] <= mouse_y <= \
                        position_left[1] + square_size:
                    response = color_left[0] > color_right[0]
                    red_intensity = staircase(response, red_intensity)
                elif position_right[0] <= mouse_x <= position_right[0] + square_size and position_right[1] <= mouse_y <= \
                        position_right[1] + square_size:
                    response = color_right[0] > color_left[0]
                    red_intensity = staircase(response, red_intensity)

                # Update colors only after user input
                color_left = (red_intensity + random.choice([-1, 1]) * step_size, 0, 0)
                color_right = (red_intensity, 0, 0)


if __name__ == "__main__":
    main()
