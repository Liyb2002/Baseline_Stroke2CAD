import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely import affinity


def compute_normal(face_vertices, other_point):
    if len(face_vertices) < 3:
        raise ValueError("Need at least three points to define a plane")


    p1 = np.array(face_vertices[0].position)
    p2 = np.array(face_vertices[1].position)
    p3 = np.array(face_vertices[2].position)

    # Create vectors from the first three points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the cross product to find the normal
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("The points do not form a valid plane")
    normal_unit = normal / norm

    # Use the other point to check if the normal should be flipped
    reference_vector = other_point.position - p1
    if np.dot(normal_unit, reference_vector) > 0:
        normal_unit = -normal_unit  # Flip the normal if it points towards the other point

    return normal_unit.tolist()


#----------------------------------------------------------------------------------#


def round_position(position, decimals=3):
    return tuple(round(coord, decimals) for coord in position)



#----------------------------------------------------------------------------------#




def find_target_verts(target_vertices, edges) :
    target_pos_1 = round_position(target_vertices[0])
    target_pos_2 = round_position(target_vertices[1])
    target_positions = {target_pos_1, target_pos_2}
    
    for edge in edges:
        verts = edge.vertices()

        edge_positions = {
            round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
            round_position([verts[1].X, verts[1].Y, verts[1].Z])
            }
        
        if edge_positions == target_positions:
            return edge
        
    return None




#----------------------------------------------------------------------------------#

def find_rectangle_on_plane(points, normal):
    """Find a new rectangle on the same plane as the given larger rectangle, with a translation.
    
    Args:
        points: List of 4 numpy arrays representing the vertices of the larger rectangle.
        normal: A numpy array representing the normal vector of the rectangle's plane.
    
    Returns:
        np.ndarray: An array of 4 vertices representing a new rectangle on the same plane.
    """
    assert len(points) == 4, "The input rectangle should have exactly 4 points."
    normal = np.array(normal) / np.linalg.norm(normal)
    points = np.array(points)
    
    # Calculate the center of the larger rectangle
    center = np.mean(points, axis=0)

    # Compute the local axes of the plane:
    # Assume the two diagonals define the axes directions for the rectangle
    diagonal1 = points[1] - points[0]
    diagonal2 = points[2] - points[1]

    # Normalize the diagonals to use them as local axes
    axis1 = diagonal1 / np.linalg.norm(diagonal1)
    axis2 = diagonal2 / np.linalg.norm(diagonal2)

    # Random scale factor between 0.2 and 0.7
    scale_factor = np.random.uniform(0.2, 0.7)
    
    # Compute new rectangle vertices based on the local axes
    half_length1 = np.linalg.norm(diagonal1) / 2 * scale_factor
    half_length2 = np.linalg.norm(diagonal2) / 2 * scale_factor

    smaller_vertices = [
        center + half_length1 * axis1 + half_length2 * axis2,
        center - half_length1 * axis1 + half_length2 * axis2,
        center - half_length1 * axis1 - half_length2 * axis2,
        center + half_length1 * axis1 - half_length2 * axis2,
    ]
    smaller_vertices = np.array(smaller_vertices)

    # Random translation factor between 0.2 and 0.7
    translation_factor1 = np.random.uniform(0.2, 0.7)
    translation_factor2 = np.random.uniform(0.2, 0.7)
    
    # Translate the new rectangle on the plane using the local axes
    translation_vector = translation_factor1 * axis1 + translation_factor2 * axis2
    translated_vertices = smaller_vertices + translation_vector

    return translated_vertices
