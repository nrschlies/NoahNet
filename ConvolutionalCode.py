import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph for the trellis diagram
G = nx.DiGraph()

# Define the number of states and time steps
states = ['00', '01', '10', '11']
time_steps = [0, 1, 2, 3]

# Add nodes for each state at each time step
for t in time_steps:
    for state in states:
        # Using LaTeX style subscripts for time step labels
        G.add_node(f"{state}_{t}", label=f"${state}_{{{t}}}$", pos=(t, int(state, 2)))

# Define transitions between states
transitions = [
    ('00_0', '00_1'), ('00_0', '10_1'),
    ('01_0', '00_1'), ('01_0', '10_1'),
    ('10_0', '01_1'), ('10_0', '11_1'),
    ('11_0', '01_1'), ('11_0', '11_1'),
    
    ('00_1', '00_2'), ('00_1', '10_2'),
    ('01_1', '00_2'), ('01_1', '10_2'),
    ('10_1', '01_2'), ('10_1', '11_2'),
    ('11_1', '01_2'), ('11_1', '11_2'),

    ('00_2', '00_3'), ('00_2', '10_3'),
    ('01_2', '00_3'), ('01_2', '10_3'),
    ('10_2', '01_3'), ('10_2', '11_3'),
    ('11_2', '01_3'), ('11_2', '11_3'),
]

# Add transitions as edges in the graph
G.add_edges_from(transitions)

# Get node positions for plotting
pos = nx.get_node_attributes(G, 'pos')

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 6))

# Set fixed aspect ratio
ax.set_aspect('equal')

# Draw the trellis diagram
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_color='black', 
        labels=nx.get_node_attributes(G, 'label'), arrows=True, arrowstyle='->', edge_color='black', font_weight='bold', ax=ax)

# Add the description using LaTeX style subscripts for a professional look
plt.text(0.5, -0.1, 
         "This is a convolutional code trellis diagram showing the state transitions of a 4-state\n"
         "convolutional encoder over 3 time steps. Each circle represents a state at a given time step,\n"
         "while the arrows depict how states transition based on the input bits.\n"
         "States: $00$, $01$, $10$, $11$ are shown at each time step, and the trellis visualizes all possible\n"
         "state transitions that can occur as data passes through the encoder.",
         horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)

# Set x and y axis limits to control figure size
plt.xlim(-0.5, 3.5)  # Keeps time steps range consistent
plt.ylim(-0.5, 3.5)  # Keeps state positioning range consistent

# Adjust plot limits to fit the description
plt.subplots_adjust(bottom=0.2)

# Adding a title with professional font
plt.title(r'Convolutional Code Trellis Diagram', fontsize=16, fontweight='bold')

plt.show()
