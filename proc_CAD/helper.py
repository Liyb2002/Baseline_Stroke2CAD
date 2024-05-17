import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely import affinity
import pyrr


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
        if len(verts) ==2 :
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


def find_triangle_on_plane(points, normal):

    four_pts = find_rectangle_on_plane(points, normal)
    idx1, idx2 = 0, 1
    point1 = four_pts[idx1]
    point2 = four_pts[idx2]

    point3 = 0.5 * (four_pts[2] + four_pts[3])

    return [point1, point2, point3]


def find_triangle_to_cut(points, normal):

    points = np.array(points)
    
    # Randomly shuffle the indices to choose three points
    start_index = np.random.randint(0, 4)

    # Determine the indices of the three points
    indices = [(start_index + i) % 4 for i in range(3)]

    
    # Use the second point as the pin point
    pin_index = indices[1]
    pin_point = points[pin_index]
    
    # Interpolate between the pin point and the other two points
    point1 = 0.5 * (pin_point + points[indices[0]])
    point2 = 0.5 * (pin_point + points[indices[2]])

    return [pin_point, point1, point2]

def random_circle(points, normal):
    four_pts = find_rectangle_on_plane(points, normal)

    pt = random.choice(four_pts)

    return pt




#----------------------------------------------------------------------------------#




def project_points(feature_lines, obj_center, img_dims=[1000, 1000]):

    obj_center = np.array(obj_center)
    cam_pos = obj_center + np.array([2,2,2])
    up_vec = np.array([0,1,0])
    view_mat = pyrr.matrix44.create_look_at(cam_pos,
                                            np.array([0, 0, 0]),
                                            up_vec)
    near = 0.001
    far = 1.0
    view_edges = []
    total_view_points = []

    print("-------------------------")
    for edge_info in feature_lines:
        view_points = []
        vertices = edge_info['vertices']
        for p in vertices:
            p -= obj_center
            hom_p = np.ones(4)
            hom_p[:3] = p
            proj_p = np.matmul(view_mat.T, hom_p)
            view_points.append(proj_p)
            total_view_points.append(proj_p)
        view_edges.append(np.array(view_points))
    
    print("view_edges", view_edges)


    
    #for f_line in view_edges:
    #    plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    #plt.show()
    total_view_points = np.array(total_view_points)
    max = np.array([np.max(total_view_points[:, 0]), np.max(total_view_points[:, 1]), np.max(total_view_points[:, 2])])
    min = np.array([np.min(total_view_points[:, 0]), np.min(total_view_points[:, 1]), np.min(total_view_points[:, 2])])

    #proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=min[0], right=max[0], bottom=min[1], top=max[1],
    #                                                                          near=near, far=far)
    max_dim = np.maximum(np.abs(max[0]-min[0]), np.abs(max[1]-min[1]))
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=-max_dim/2, right=max_dim/2, bottom=-max_dim/2, top=max_dim/2,
                                                                              near=near, far=far)

    total_projected_points = []
    projected_edges = []
    for f_line in view_edges:
        projected_points = []
        for p in f_line.copy():
            proj_p = np.matmul(proj_mat, p)
            proj_p[:3] /= proj_p[-1]
            total_projected_points.append(proj_p[:2])
            projected_points.append(proj_p[:2])
        projected_edges.append(np.array(projected_points))
    total_projected_points = np.array(total_projected_points)

    # screen-space
    # scale to take up 80% of the image
    max = np.array([np.max(total_projected_points[:, 0]), np.max(total_projected_points[:, 1])])
    min = np.array([np.min(total_projected_points[:, 0]), np.min(total_projected_points[:, 1])])
    bbox_diag = np.linalg.norm(max - min)
    screen_diag = np.sqrt(img_dims[0]*img_dims[0]+img_dims[1]*img_dims[1])
    scaled_edges = []
    for f_line in projected_edges:
        scaled_points = []
        for p in f_line:
            p[1] *= -1
            p *= 0.5*screen_diag/bbox_diag
            #p *= 0.8*500/bbox_diag
            p += np.array([img_dims[0]/2, img_dims[1]/2])
            scaled_points.append(p)
        scaled_edges.append(np.array(scaled_points))

    return scaled_edges

