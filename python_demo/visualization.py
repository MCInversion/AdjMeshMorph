import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

def plot_mesh(vertices, elements, displacements, scale=1):
    """
    Plots the original and deformed mesh.

    :param vertices: Array of vertex positions.
    :param elements: Array of elements (vertex indices).
    :param displacements: Array of vertex displacements.
    :param scale: Scaling factor for displacements for visualization.
    """
    # Create figure
    plt.figure(figsize=(12, 6))

    # Subplot for original mesh
    plt.subplot(1, 2, 1)
    plot_elements(vertices, elements, title="Original Mesh")

    # Subplot for deformed mesh
    plt.subplot(1, 2, 2)
    
    # Ensure displacements are reshaped to match the vertices array
    num_vertices = len(vertices)
    reshaped_displacements = displacements.reshape(num_vertices, 2)

    # Apply scaling and add to the original vertices
    deformed_vertices = vertices + scale * reshaped_displacements
    plot_elements(deformed_vertices, elements, title="Deformed Mesh")

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_elements(vertices, elements, title="Mesh"):
    """
    Helper function to plot elements of the mesh.

    :param vertices: Array of vertex positions.
    :param elements: Array of elements (vertex indices).
    :param title: Title of the subplot.
    """
    for element in elements:
        polygon = plt.Polygon(vertices[element], edgecolor='black', alpha=0.3, fill=None)
        plt.gca().add_patch(polygon)

    plt.scatter(vertices[:, 0], vertices[:, 1], color='red')  # Vertex points
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.axis('equal')

# ======================================================================================================================

def animate_mesh(vertices, elements, displacements_over_time, scale=1, interval=50, delta_t=1):
    """
    Creates an animation of the mesh deformation over time with time display.

    :param vertices: Array of vertex positions.
    :param elements: Array of elements (vertex indices).
    :param displacements_over_time: List of displacement arrays over time.
    :param scale: Scaling factor for displacements for visualization.
    :param interval: Time interval between frames in milliseconds.
    :param delta_t: Time step size used in the simulation.
    """
    fig, ax = plt.subplots()
    ax.set_xlim((np.min(vertices[:, 0]) - 1, np.max(vertices[:, 0]) + 1))
    ax.set_ylim((np.min(vertices[:, 1]) - 1, np.max(vertices[:, 1]) + 1))
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    title = ax.set_title('Mesh Deformation Over Time')

    polygons = [Polygon(vertices[element], edgecolor='black', alpha=0.3, fill=None) for element in elements]
    for polygon in polygons:
        ax.add_patch(polygon)
    points, = ax.plot([], [], 'ro')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        for polygon, element in zip(polygons, elements):
            polygon.set_xy(vertices[element])
        points.set_data(vertices[:, 0], vertices[:, 1])
        time_text.set_text('')
        return polygons + [points, time_text]

    def animate(i):
        reshaped_displacements = displacements_over_time[i].reshape(len(vertices), 2)
        deformed_vertices = vertices + scale * reshaped_displacements

        for polygon, element in zip(polygons, elements):
            polygon.set_xy(deformed_vertices[element])

        points.set_data(deformed_vertices[:, 0], deformed_vertices[:, 1])
        time_text.set_text(f't = {i * delta_t:.2f}s')
        return polygons + [points, time_text]

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(displacements_over_time), interval=interval, blit=True)
    plt.show()

    return anim