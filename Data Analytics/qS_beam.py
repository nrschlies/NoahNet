import cv2
import numpy as np
from copy import deepcopy

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return image, binary

def detect_grid(binary_image):
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(binary_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return binary_image

def extract_cells(image, num_rows, num_cols):
    cells = []
    height, width, _ = image.shape
    cell_height = height // num_rows
    cell_width = width // num_cols

    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            y1, y2 = i * cell_height, (i + 1) * cell_height
            x1, x2 = j * cell_width, (j + 1) * cell_width
            cell = image[y1:y2, x1:x2]
            row.append(cell)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cells.append(row)
    return cells

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

def get_color_name(avg_color, predefined_colors):
    min_dist = float('inf')
    color_name = 'unknown'
    for name, bgr in predefined_colors.items():
        dist = np.linalg.norm(np.array(avg_color) - np.array(bgr))
        if dist < min_dist:
            min_dist = dist
            color_name = name
    return color_name

def recognize_colors(cells, predefined_colors, padding=10):
    color_matrix = []
    for row in cells:
        color_row = []
        for cell in row:
            h, w, _ = cell.shape
            sample_region = cell[padding:h-padding, padding:w-padding]
            avg_color = np.mean(sample_region.reshape(-1, 3), axis=0)
            color_name = get_color_name(avg_color, predefined_colors)
            color_row.append(color_name)
        color_matrix.append(color_row)
    return color_matrix

def detect_queens(cells):
    board_with_queens = []
    for row in cells:
        row_with_queens = []
        for cell in row:
            h, w, _ = cell.shape
            center = (h // 2, w // 2)
            half_size = 4  # Half of 9x9 grid size
            grid_area = cell[center[0] - half_size:center[0] + half_size + 1, center[1] - half_size + 1]
            if np.any(grid_area == 0):  # Check if there are any black pixels in the 9x9 grid
                row_with_queens.append(True)
            else:
                row_with_queens.append(False)
        board_with_queens.append(row_with_queens)
    return board_with_queens

def setup_initial_board(color_matrix, queen_matrix):
    board = [['empty' for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            color_name = color_matrix[i][j]
            if queen_matrix[i][j]:
                board[i][j] = (color_name, 'Q')
            else:
                board[i][j] = (color_name, ' ')
    for i in range(9):
        for j in range(9):
            if board[i][j][1] == 'Q':
                mark_attacks(board, i, j)
    return board

def mark_attacks(board, row, col):
    n = len(board)
    # Mark row and column
    for i in range(n):
        if board[row][i][1] != 'Q':
            board[row][i] = (board[row][i][0], 'X')
        if board[i][col][1] != 'Q':
            board[i][col] = (board[i][col][0], 'X')
    
    # Mark diagonals
    for i in range(n):
        for j in range(n):
            if abs(row - i) == abs(col - j) and board[i][j][1] != 'Q':
                board[i][j] = (board[i][j][0], 'X')

def mark_color(board, color):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j][0] == color and board[i][j][1] == ' ':
                board[i][j] = (color, 'X')

def print_board(board, color_map):
    for row in board:
        print(' '.join([f'{color_map[color]}{status}' for color, status in row]))
    print()

def is_valid(board, row, col):
    return board[row][col][1] == ' '

def dfs(board, color_positions, color_order, fixed_positions, current_color_index, queens_placed, color_map):
    if queens_placed == len(color_order):
        return board

    color = color_order[current_color_index]

    # If a queen for the current color is already placed, skip to the next color
    if any(board[row][col][0] == color and board[row][col][1] == 'Q' for row, col in color_positions[color]):
        return dfs(board, color_positions, color_order, fixed_positions, (current_color_index + 1) % len(color_order), queens_placed, color_map)

    valid_positions = [(row, col) for row, col in color_positions[color] if (row, col) not in fixed_positions and is_valid(board, row, col)]

    if not valid_positions:
        print(f"(INCORRECT, BACKTRACK) for color {color} after checking all positions.")
        return None

    for row, col in valid_positions:
        new_board = deepcopy(board)
        new_board[row][col] = (color, 'Q')
        mark_attacks(new_board, row, col)
        mark_color(new_board, color)
        print(f"Trying position ({row}, {col}) for color {color}")
        print_board(new_board, color_map)

        result = dfs(new_board, color_positions, color_order, fixed_positions, (current_color_index + 1) % len(color_order), queens_placed + 1, color_map)
        if result:
            return result

    print(f"(INCORRECT, BACKTRACK) for color {color} after checking positions: {valid_positions}")
    return None





def solve_queens(color_matrix, predefined_colors, queen_matrix):
    n = len(color_matrix)
    board = setup_initial_board(color_matrix, queen_matrix)
    color_positions = {color: [] for color in predefined_colors.keys()}
    color_order = list(predefined_colors.keys())
    fixed_positions = set()

    for i in range(n):
        for j in range(n):
            color_name = color_matrix[i][j]
            if queen_matrix[i][j]:
                fixed_positions.add((i, j))
            if color_name in predefined_colors:
                color_positions[color_name].append((i, j))

    color_map = {color: str(index) for index, color in enumerate(predefined_colors.keys())}
    
    print("Initial color positions:", color_positions)
    print("Fixed positions:", fixed_positions)
    
    solved_board = dfs(board, color_positions, color_order, fixed_positions, 0, 0, color_map)
    if solved_board:
        return solved_board, color_map
    else:
        print("No solution exists.")
        return None, None

def print_solved_board(board, color_map):
    for row in board:
        print(' '.join(['Q' if status == 'Q' else '.' for _, status in row]))

# Main code
image_path = 'input_puzzle.jpg'

image, binary = preprocess_image(image_path)
if image is not None and binary is not None:
    grid_image = detect_grid(binary)
    cells = extract_cells(image, num_rows=9, num_cols=9)
    if cells:
        predefined_colors = {
            'light_blue': hex_to_bgr('#a5d2d8'),
            'gray': hex_to_bgr('#dfdfdf'),
            'blue': hex_to_bgr('#97beff'),
            'light_green': hex_to_bgr('#b3dea0'),
            'orange': hex_to_bgr('#ff7b60'),
            'purple': hex_to_bgr('#bba2e2'),
            'yellow': hex_to_bgr('#ffc993'),
            'pink': hex_to_bgr('#dea0bf'),
            'light_yellow': hex_to_bgr('#e5f387')
        }
        color_matrix = recognize_colors(cells, predefined_colors, padding=10)
        queen_matrix = detect_queens(cells)
        board = setup_initial_board(color_matrix, queen_matrix)
        print("Initial Board:")
        print_board(board, {color: str(index) for index, color in enumerate(predefined_colors.keys())})
        solved_board, color_map = solve_queens(color_matrix, predefined_colors, queen_matrix)
        if solved_board:
            print("Solved Board:")
            print_solved_board(solved_board, color_map)
        else:
            print("Solution could not be found.")
else:
    print("Error: Image processing failed.")
