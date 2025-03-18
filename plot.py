import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_tsp_path(path, coordinates, image_path, cost):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()

    # Plot the map
    ax.imshow(img)

    # Plot the nodes
    for city, (x, y) in zip(path, coordinates):
        ax.plot(x, y, 'bo')

    # plot the edges (lines connecting the nodes)
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        ax.plot([x1, x2], [y1, y2], 'r-')

    # connect the last and first cities
    x1, y1 = coordinates[-1]
    x2, y2 = coordinates[0]
    ax.plot([x1, x2], [y1, y2], 'r-')

    # annotate the cost
    cost_text = 'Total Cost: {:.2f}'.format(cost)
    ax.text(0.5, -0.1, cost_text, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', boxstyle='round4'))

    plt.title('TSP Path')
    plt.show()


def plot_genetic_diversity(genetic_diversity_values):
    """
    Plot the genetic diversity over generations.

    Parameters:
    - genetic_diversity_values: List of genetic diversity values for each generation.
    """
    generations = range(1, len(genetic_diversity_values) + 1)
    plt.plot(generations, genetic_diversity_values, marker='o')
    plt.title('Genetic Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Genetic Diversity')
    plt.xticks(generations)
    plt.show()