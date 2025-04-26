#!python3
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import pow
import sys

def calculate_position(round_idx, i, num_teams):
    """Calculate the y-position for a node with dynamic spacing to avoid overlap."""
    p = pow(2, round_idx)
    spacing_factor = max(1, num_teams / 8)  # Ensure proper vertical spacing
    return (-(p * i) - ((p - 1) / 2.0)) * spacing_factor  # Apply spacing factor

def generate_bracket(teams, matchdecider:any=None):
    """Generate a tournament bracket structure."""
    bracket = [teams]
    while len(bracket[-1]) > 1:
        new_round = []
        for i in range(0, len(bracket[-1]), 2):
            teams = [bracket[-1][i], bracket[-1][i+1]]
            if matchdecider == None:
                winner = random.choice(teams)
                new_round.append(winner)
            else:
                winner = matchdecider(teams[0], teams[1])
                new_round.append(winner)
        bracket.append(new_round)
    return bracket

def draw_bracket(bracket):
    """Draw the tournament bracket with proper scaling and spacing."""
    G = nx.DiGraph()
    pos = {}
    num_teams = len(bracket[0])  # Get the number of teams for spacing adjustments

    for round_idx, round in enumerate(bracket):
        for i, team in enumerate(round):
            y = calculate_position(round_idx, i, num_teams)
            pos[f"R{round_idx}_T{i}"] = (round_idx*1.25, y)
            G.add_node(f"R{round_idx}_T{i}", label=team)

    # **Determine Dynamic Figure Size**
    min_y = min(y for _, y in pos.values())
    max_y = max(y for _, y in pos.values())
    height_needed = abs(max_y - min_y) / 10  # Scale height dynamically

    # **Set Proper Min/Max Heights**
    max_height = 16
    min_height = 6
    dynamic_height = min(max_height, max(min_height, height_needed))

    # **Adjust Font Size to Prevent Overlap**
    base_font_size = 12
    font_size = max(5, base_font_size - (num_teams * 0.2))

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    for round_idx in range(1, len(bracket)):
        for i in range(len(bracket[round_idx - 1]) // 2):
            parent1 = f"R{round_idx - 1}_T{2 * i}"
            parent2 = f"R{round_idx - 1}_T{2 * i + 1}"
            child = f"R{round_idx}_T{i}"
            
            x1, y1 = pos[parent1]
            x2, y2 = pos[parent2]
            xc, yc = pos[child]
            
            # **Draw Orthogonal Edges**
            ax.plot([x1, x1 + 0.5, x1 + 0.5, xc], [y1, y1, yc, yc], 'k-')
            ax.plot([x2, x2 + 0.5, x2 + 0.5, xc], [y2, y2, yc, yc], 'k-')
            
            G.add_edge(parent1, child)
            G.add_edge(parent2, child)

    # **Draw Nodes as Labels (With Dynamic Font Size)**
    for node, (x, y) in pos.items():
        ax.text(x, y, G.nodes[node]['label'], ha='center', va='center', fontsize=font_size,
                bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))

    # **Fix Overlapping Issues by Expanding Y-Limits**
    ax.set_xlim(-2, len(bracket)+2)
    ax.set_ylim(min_y - 3, max_y + 3)

    ax.axis('off')
    plt.show()
def main(arguments=sys.argv[1:], stream=sys.stdin):
    teams = []
    round_data = []
    
    for line in stream:
        line = line.strip()
        if line:  # If the line is not blank, add it to the current round
            round_data.append(line)
        elif round_data:  # If we encounter a blank line and have data, start a new round
            teams.append(round_data)
            round_data = []
    
    if round_data:  # Add the last round if the file doesn't end in a blank line
        teams.append(round_data)
    
    if not teams or (len(teams[0]) & (len(teams[0]) - 1)) != 0:
        print("Number of teams in the first round must be a power of 2 (e.g., 2, 4, 8, 16, 32, 64)")
        return
    
    # bracket = generate_bracket(teams, matchdecider=None)
    # bracket = [[f'{str(y)[:21]:<21}' for y in x] for x in bracket]
    draw_bracket(teams)

if __name__ == "__main__":
    main()
