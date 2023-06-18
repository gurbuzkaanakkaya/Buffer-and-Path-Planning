import sys
import math
import numpy as np
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from collections import deque
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from concurrent.futures import ThreadPoolExecutor

def create_matrix(coordinates):

    """
       Create a matrix from a given list of coordinates.

       Args:
           coordinates (list): List of coordinate values.

       Returns:
           matrix (numpy array): Matrix representation of the coordinates.
           max_element_width (int): Width of the maximum element in the matrix.
    """
    
    row_count         = len(coordinates)
    column_count      = len(coordinates)
    matrix            = np.array([[0 for _ in range(column_count)] for _ in range(row_count)])
    max_element_width = len(str(np.max(matrix)))

    return matrix, max_element_width

def create_graph_from_matrix(matrix):

    """
        Creates a graph from a given matrix representation.

        Args:
            matrix: A 2D matrix representing the weights between vertices.

        Returns:
            graph: A NetworkX graph object with nodes and weighted edges.
    """
    
    num_vertices = len(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))

    for i in range(num_vertices):
        for j in range(num_vertices):
            weight = matrix[i][j]
            if weight != 0 and weight != sys.maxsize:
                graph.add_edge(i, j, weight=weight)

    return graph

def shortest_path(graph, source, target):

    """
        Find the shortest path between two nodes in a graph using Dijkstra's algorithm.

        Args:
            graph (NetworkX Graph): The graph in which to find the shortest path.
            source: The starting node for the path.
            target: The target node for the path.

        Returns:
            path: A list representing the shortest path from the source to the target.
    """
    
    path = nx.dijkstra_path(graph, source, target)

    return path

def show_matrix(matrix, element_with):

    # Print the matrix with proper alignment of elements.
    for row in matrix:
        row_str = ' '.join(str(element).rjust(element_with) for element in row)
        print(row_str)

def is_vertex_convex(vertices, vertex_index):

    """
        Check if a vertex in a list of vertices is convex.

        Args:
            vertices (list): List of 2D vertex coordinates.
            vertex_index (int): Index of the vertex to check.

        Returns:
            is_convex (bool): True if the vertex is convex, False otherwise.
    """
    
    size_of_list = len(vertices)

    previous_index = (vertex_index - 1) % size_of_list
    next_index = (vertex_index + 1) % size_of_list

    coordinate_x1, coordinate_y1 = vertices[previous_index]
    coordinate_x2, coordinate_y2 = vertices[vertex_index]
    coordinate_x3, coordinate_y3 = vertices[next_index]

    cross_product = (coordinate_x2 - coordinate_x1) * (coordinate_y3 - coordinate_y2) - \
                    (coordinate_y2 - coordinate_y1) * (coordinate_x3 - coordinate_x2)

    return cross_product > 0

def haversine_distance(latitude1, longitude1, latitude2, longitude2):

    """
        Calculate the haversine distance between two sets of latitude and longitude coordinates.

        Args:
            latitude1 (float): Latitude of the first point.
            longitude1 (float): Longitude of the first point.
            latitude2 (float): Latitude of the second point.
            longitude2 (float): Longitude of the second point.

        Returns:
            distance (float): Haversine distance between the two points in meters.
    """

    earth_radius = 6371000

    latitude1_radian = math.radians(latitude1)
    longitude1_radian = math.radians(longitude1)
    latitude2_radian = math.radians(latitude2)
    longitude2_radian = math.radians(longitude2)

    delta_lat = latitude2_radian - latitude1_radian
    delta_lon = longitude2_radian - longitude1_radian

    center_angle_of_circle_arc = math.sin(delta_lat / 2) ** 2 + math.cos(latitude1_radian) * \
                                 math.cos(latitude2_radian) * math.sin(delta_lon / 2) ** 2
    total_angle_of_the_circular_arc = 2 * math.atan2(math.sqrt(center_angle_of_circle_arc),
                                                     math.sqrt(1 - center_angle_of_circle_arc))

    distance = earth_radius * total_angle_of_the_circular_arc
    return distance

def buffered_point(poly_list, distance=1000):

    """
       Create a buffer around the points of a polygon.

       Args:
           poly_list (list): List of polygons, where each polygon is represented as a list of vertices.
           distance (float): Distance of the buffer around the points (default: 100).

       Returns:
           buff_vertices_l (list): List of polygons with buffered points, where each polygon is represented as a list of vertices.
           buff_vertices_t (list): List of buffered points, where each point is represented as a tuple (x, y).
    """

    buff_vertices_l = []
    buff_vertices_t = []
    for poly in poly_list:
        size_of_list = len(poly)
        poly_coord = deque()
        for index, current_vertex in enumerate(poly):
            control_vertices = [(poly[index - 1][0], poly[index - 1][1]), (current_vertex[0], current_vertex[1]),
                                (poly[(index + 1) % size_of_list][0], poly[(index + 1) % size_of_list][1])]

            first_second_vertex_distance  = haversine_distance(current_vertex[0],
                                                               current_vertex[1],
                                                               poly[index - 1][0],
                                                               poly[index - 1][1])
            second__third_vertex_distance = haversine_distance(current_vertex[0],
                                                               current_vertex[1],
                                                               poly[(index + 1) % size_of_list][0],
                                                               poly[(index + 1) % size_of_list][1])

            third_first_x_dist = poly[(index + 1) % size_of_list][0] - poly[index - 1][0]
            third_first_y_dist = poly[(index + 1) % size_of_list][1] - poly[index - 1][1]

            total_rate = first_second_vertex_distance + second__third_vertex_distance

            point_of_bisector_x = poly[index - 1][0] + ((third_first_x_dist / total_rate) *
                                                        first_second_vertex_distance)
            point_of_bisector_y = poly[index - 1][1] + ((third_first_y_dist / total_rate) *
                                                        first_second_vertex_distance)

            if (point_of_bisector_x, point_of_bisector_y) == (current_vertex[0], current_vertex[1]):
                new_point_x = current_vertex[0] + 0.000000000000001
                current_vertex = (new_point_x, current_vertex[1])

            bisector_distance_vertex = haversine_distance(current_vertex[0],
                                                          current_vertex[1],
                                                          point_of_bisector_x,
                                                          point_of_bisector_y)

            if not is_vertex_convex(control_vertices, 1):
                new_point_x = current_vertex[0] + (current_vertex[0] -
                                                point_of_bisector_x) * distance / bisector_distance_vertex
                new_point_y = current_vertex[1] + (current_vertex[1] -
                                                point_of_bisector_y) * distance / bisector_distance_vertex

            else:
                new_point_x = current_vertex[0] - (current_vertex[0] -
                                                point_of_bisector_x) * distance / bisector_distance_vertex
                new_point_y = current_vertex[1] - (current_vertex[1] -
                                                point_of_bisector_y) * distance / bisector_distance_vertex

            poly_coord.append((new_point_x, new_point_y))
            buff_vertices_t.append((new_point_x, new_point_y))
        buff_vertices_l.append(poly_coord)

    return buff_vertices_l, buff_vertices_t

def shapely_polygon(vertices):

    """
        Create polygons using the Polygon class from the Shapely library.

        Args:
            vertices (list): A list containing the corner vertices of the polygon.

        Returns:
            poly_list (list): A list containing the polygons, where each polygon is represented using Shapely's Polygon class.
    """
    
    poly_list = []
    for poly in vertices:
        poly_list.append(Polygon(poly))
    return poly_list

def create_linestring(all_vertices):

    """
        Create LineString objects between all pairs of vertices.

        Args:
            all_vertices (list): A list of vertex coordinates.

        Returns:
            list_of_line (dict): A dictionary where the keys are the WKT representations of the LineString objects,
                                 and the values are tuples of the corresponding vertex indices.
    """

    list_of_line = {}
    for index1 in range(len(all_vertices)):
        start_point_coords = all_vertices[index1]
        for index2 in range(index1 + 1, len(all_vertices)):
            end_point_coords = all_vertices[index2]
            linestring = LineString([start_point_coords, end_point_coords])
            list_of_line[linestring.wkt] = (index1, index2)

    return list_of_line

def intersection(shapely_poly_list, all_vertices, linestring_wkt):

    """
       Checks the intersections between polygons and LineString objects and updates the weight matrix.

       Args:
           shapely_poly_list (list): A list containing Shapely Polygon objects.
           all_vertices (list):      A list containing the coordinates of all vertices.
           linestring_wkt (dict):    A dictionary where the WKT representation of LineString objects is the key,
                                     and the values represent the indices of the vertices.

       Returns:
           None
    """

    INF_VALUE = 99999999
    for line_str, wkt_index in linestring_wkt.items():
        line = wkt.loads(line_str)
        for polygon in shapely_poly_list:
            intersect = polygon.intersection(line)

            if intersect.is_empty or isinstance(intersect, (Point, MultiPoint)):
                dist_points = haversine_distance(all_vertices[int(wkt_index[0])][0],
                                                 all_vertices[int(wkt_index[0])][1],
                                                 all_vertices[int(wkt_index[1])][0],
                                                 all_vertices[int(wkt_index[1])][1])

                weight_matrix[wkt_index[0]][wkt_index[1]] = dist_points
                weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points

            elif isinstance(intersect, LineString):
                if wkt_index[0] == 0 and wkt_index[1] == 1:
                    center_x = (intersect.coords[0][0] + intersect.coords[1][0]) / 2
                    center_y = (intersect.coords[0][1] + intersect.coords[1][1]) / 2
                    distance = polygon.boundary.distance(Point(center_x, center_y))

                    if distance == 0:
                        dist_points = haversine_distance(all_vertices[int(wkt_index[0])][0],
                                                         all_vertices[int(wkt_index[0])][1],
                                                         all_vertices[int(wkt_index[1])][0],
                                                         all_vertices[int(wkt_index[1])][1])
                        weight_matrix[wkt_index[0]][wkt_index[1]] = dist_points
                        weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points
                        weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points
                        break

                    else:
                        weight_matrix[wkt_index[0]][wkt_index[1]] = INF_VALUE
                        weight_matrix[wkt_index[1]][wkt_index[0]] = INF_VALUE
                        break

                elif wkt_index[0] + 1 == wkt_index[1] \
                        and (all_vertices[wkt_index[0]] in polygon.exterior.coords
                             and all_vertices[wkt_index[1]] in polygon.exterior.coords):
                    dist_points = haversine_distance(all_vertices[int(wkt_index[0])][0],
                                                     all_vertices[int(wkt_index[0])][1],
                                                     all_vertices[int(wkt_index[1])][0],
                                                     all_vertices[int(wkt_index[1])][1])

                    weight_matrix[wkt_index[0]][wkt_index[1]] = dist_points
                    weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points
                    break

                else:
                    weight_matrix[wkt_index[0]][wkt_index[1]] = INF_VALUE
                    weight_matrix[wkt_index[1]][wkt_index[0]] = INF_VALUE
                    break

            else:
                weight_matrix[wkt_index[0]][wkt_index[1]] = INF_VALUE
                weight_matrix[wkt_index[1]][wkt_index[0]] = INF_VALUE
                break


def process_line(shapely_poly_list, all_vertices, linestring_wkt):

    """
       Process the line by splitting it into two halves and running intersection calculations in parallel.

       Args:
           shapely_poly_list (list): A list containing Shapely Polygon objects.
           all_vertices (list): A list containing the coordinates of all vertices.
           linestring_wkt (dict): A dictionary where the WKT representation of LineString objects is the key,
                                 and the values represent the indices of the vertices.

       Returns:
           None
    """

    half_length = len(linestring_wkt) // 2
    first_half = dict(list(linestring_wkt.items())[:half_length])
    second_half = dict(list(linestring_wkt.items())[half_length:])

    with ThreadPoolExecutor() as executor:
        executor.submit(intersection, shapely_poly_list, all_vertices, first_half)
        executor.submit(intersection, shapely_poly_list, all_vertices, second_half)


if __name__ == '__main__':
    # Define lists to store coordinate and polygon information
    coordinate_list = []
    polygon_list = []

    # Read the 'Coordinates' file
    with open('Coordinates', 'r') as file:
        coordinate_info = file.readlines()
        data = [line.strip().split(',') for line in coordinate_info]

    temp_list = []
    polygon_start_index = 2

    # Process the data and add coordinates to the coordinate_list and polygons to the polygon_list
    for coordinate in data:
        if float(coordinate[0]) == 0:
            polygon_list.append(temp_list)
            temp_list = []
        else:
            x_coordinate = float(coordinate[0])
            y_coordinate = float(coordinate[1])
            coordinate_list.append((x_coordinate, y_coordinate))
            if polygon_start_index > 0:
                polygon_start_index -= 1
            else:
                temp_list.append((x_coordinate, y_coordinate))

    # Create the weight matrix and determine the maximum width
    weight_matrix, max_width = create_matrix(coordinate_list)

    # Buffer the points to create new vertices lists
    vertices_list, vertices_list_t = buffered_point(polygon_list)
    vertices_list_t.insert(0, coordinate_list[0])
    vertices_list_t.insert(1, coordinate_list[1])

    # Create Shapely Polygon objects
    shapely_polygon_list = shapely_polygon(vertices_list)

    # Create LineString objects and store them as a dictionary with the vertex indices
    linestring_list = create_linestring(vertices_list_t)

    # Perform intersection operations in parallel
    process_line(shapely_polygon_list, vertices_list_t, linestring_list)

    # Create a graph from the weight matrix
    graph = create_graph_from_matrix(weight_matrix)

    # Perform the shortest path finding
    path = shortest_path(graph, 0, 1)

    # Get the coordinates of the shortest path and store them as a list
    pd_path_coord = []
    for vertex in path:
        pd_path_coord.append(vertices_list_t[vertex])

    # Create the LineString object for the shortest path
    line = LineString(pd_path_coord)

    # Create Point objects for the start and target points
    start_target_point = [Point(vertices_list_t[0]), Point(vertices_list_t[1])]

    # Create a GeoDataFrame with the data
    data = gpd.GeoDataFrame(
        geometry=[polygon for polygon in shapely_polygon_list] + [line] + [point for point in start_target_point])

    # Create a plot and display the data
    fig, ax = plt.subplots()
    data[data.geometry.type == 'Polygon'].plot(ax=ax, color='lightblue', edgecolor='black')
    data[data.geometry.type == 'LineString'].plot(ax=ax, color='red')
    data[data.geometry.type == 'Point'].plot(ax=ax, color='green')
    plt.show()

    # Uncomment the following lines if needed to display the matrix size and other output
    
    # print(f"{len(shapely_polygon_list)} polygons, {len(vertices_list_t)} points")
    # show_matrix(weight_matrix, max_width)
