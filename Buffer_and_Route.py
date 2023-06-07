import math
import time
import numpy as np
import sys
from shapely import wkt
import networkx as nx
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import matplotlib.pyplot as plt

def create_matrix(coordinates):
    row_count         = len(coordinates)
    column_count      = len(coordinates)
    matrix            = np.array([[0 for _ in range(column_count)] for _ in range(row_count)])
    max_element_width = len(str(np.max(matrix)))

    return matrix, max_element_width

def create_graph_from_matrix(matrix):
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
    path = nx.dijkstra_path(graph, source, target)
    return path

def show_matrix(matrix, element_with):
    for row in matrix:
        row_str = ' '.join(str(element).rjust(element_with) for element in row)
        print(row_str)

def is_vertex_convex(vertices, vertex_index):
    size_of_list = len(vertices)

    previous_index               = (vertex_index - 1) % size_of_list
    next_index                   = (vertex_index + 1) % size_of_list

    coordinate_x1, coordinate_y1 = vertices[previous_index]
    coordinate_x2, coordinate_y2 = vertices[vertex_index]
    coordinate_x3, coordinate_y3 = vertices[next_index]

    cross_product                = (coordinate_x2 - coordinate_x1) * (coordinate_y3 - coordinate_y2) - \
                                   (coordinate_y2 - coordinate_y1) * (coordinate_x3 - coordinate_x2)

    return cross_product > 0



def haversine_distance(latitude1, longitude1, latitude2, longitude2):
    earth_radius = 6371000

    latitude1_radian                = math.radians(latitude1)
    longitude1_radian               = math.radians(longitude1)
    latitude2_radian                = math.radians(latitude2)
    longitude2_radian               = math.radians(longitude2)

    delta_lat                       = latitude2_radian  - latitude1_radian
    delta_lon                       = longitude2_radian - longitude1_radian

    center_angle_of_circle_arc      = math.sin(delta_lat / 2) ** 2 + math.cos(latitude1_radian) *\
                                      math.cos(latitude2_radian) * math.sin(delta_lon / 2) ** 2
    total_angle_of_the_circular_arc = 2 * math.atan2(math.sqrt(center_angle_of_circle_arc),
                                      math.sqrt(1 - center_angle_of_circle_arc))

    distance                        = earth_radius * total_angle_of_the_circular_arc
    return distance

def buffered_point(poly_list, distance = 100):
    buff_vertices_l = []
    buff_vertices_t = []
    poly_coord = []
    for poly in poly_list:
        size_of_list = len(poly)
        for index in range(len(poly)):
            control_vertices = []

            control_vertices.append((poly[index - 1][0]                     , poly[index - 1][1]))
            control_vertices.append((poly[index][0]                         , poly[index][1]))
            control_vertices.append((poly[(index + 1) % size_of_list][0]    , poly[(index + 1) % size_of_list][1]))

            first_second_vertex_distance  = haversine_distance(poly[index][0],
                                                               poly[index][1],
                                                               poly[index - 1][0],
                                                               poly[index - 1][1])
            second__third_vertex_distance = haversine_distance(poly[index][0],
                                                               poly[index][1],
                                                               poly[(index + 1) % size_of_list][0],
                                                               poly[(index + 1) % size_of_list][1])

            third_first_x_dist            = poly[(index + 1) % size_of_list][0] - poly[index - 1][0]
            third_first_y_dist            = poly[(index + 1) % size_of_list][1] - poly[index - 1][1]

            total_rate                    = first_second_vertex_distance        + second__third_vertex_distance

            point_of_bisector_x           = poly[index - 1][0]                  + ((third_first_x_dist / total_rate) *
                                                                                    first_second_vertex_distance)
            point_of_bisector_y           = poly[index - 1][1]                  + ((third_first_y_dist / total_rate) *
                                                                                    first_second_vertex_distance)
            
            if (point_of_bisector_x, point_of_bisector_y) == (poly[index][0], poly[index][1]):
                new_point_x = poly[index][0] + 0.000000000000001
                poly[index] = (new_point_x, poly[index][1])
                
            bisector_distance_vertex = haversine_distance(poly[index][0],
                                                          poly[index][1],
                                                          point_of_bisector_x,
                                                          point_of_bisector_y)

            if not is_vertex_convex(control_vertices, 1):
                new_point_x = poly[index][0] + (poly[index][0] -
                              point_of_bisector_x) * distance / bisector_distance_vertex
                new_point_y = poly[index][1] + (poly[index][1] -
                              point_of_bisector_y) * distance / bisector_distance_vertex

            else:
                new_point_x = poly[index][0] - (poly[index][0] -
                              point_of_bisector_x) * distance / bisector_distance_vertex
                new_point_y = poly[index][1] - (poly[index][1] -
                              point_of_bisector_y) * distance / bisector_distance_vertex


            poly_coord.append((new_point_x, new_point_y))
            buff_vertices_t.append((new_point_x, new_point_y))
        buff_vertices_l.append(poly_coord)
        poly_coord = []

    return buff_vertices_l, buff_vertices_t


def shapely_polygon(vertices):
    poly_list = []
    for poly in vertices:
        poly_list.append(Polygon(poly))
    return poly_list

def create_linestring(all_vertices):
    list_of_line = {}

    for index1 in range(len(all_vertices)):
        start_point_coords = all_vertices[index1]
        for index2 in range(index1+1, len(all_vertices)):
            end_point_coords = all_vertices[index2]
            linestring = LineString([start_point_coords, end_point_coords])
            list_of_line[linestring.wkt] = (index1, index2)
            #print(f'{linestring} {index1} {index2}')

    return list_of_line


def intersection(shapely_poly_list, all_vertices, linestring_wkt):
    inf_value = 99999999
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
                if wkt_index[0] + 1 == wkt_index[1]:
                    dist_points = haversine_distance(all_vertices[int(wkt_index[0])][0],
                                                     all_vertices[int(wkt_index[0])][1],
                                                     all_vertices[int(wkt_index[1])][0],
                                                     all_vertices[int(wkt_index[1])][1])

                    weight_matrix[wkt_index[0]][wkt_index[1]] = dist_points
                    weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points
                    break

                else:
                    weight_matrix[wkt_index[0]][wkt_index[1]] = inf_value
                    weight_matrix[wkt_index[1]][wkt_index[0]] = inf_value
                    break
                    
            #   Alternatif Çözüm   #
            
            #elif isinstance(intersect, LineString):
            #    center_x = (intersect.coords[0][0] + intersect.coords[1][0]) / 2
            #    center_y = (intersect.coords[0][1] + intersect.coords[1][1]) / 2
            #    distance = polygon.boundary.distance(Point(center_x, center_y))

            #    if distance == 0:
            #        dist_points = haversine_distance(all_vertices[int(wkt_index[0])][0],
            #                                         all_vertices[int(wkt_index[0])][1],
            #                                         all_vertices[int(wkt_index[1])][0],
            #                                         all_vertices[int(wkt_index[1])][1])

            #        weight_matrix[wkt_index[0]][wkt_index[1]] = dist_points
            #        weight_matrix[wkt_index[1]][wkt_index[0]] = dist_points
            #        break

            #    else:
            #        weight_matrix[wkt_index[0]][wkt_index[1]] = inf_value
            #        weight_matrix[wkt_index[1]][wkt_index[0]] = inf_value
            #        break

            else:
                weight_matrix[wkt_index[0]][wkt_index[1]] = inf_value
                weight_matrix[wkt_index[1]][wkt_index[0]] = inf_value
                break

def process_line(shapely_poly_list, all_vertices, linestring_wkt):
    half_length = len(linestring_wkt) // 2
    first_half = dict(list(linestring_wkt.items())[:half_length])
    second_half = dict(list(linestring_wkt.items())[half_length:])

    with ThreadPoolExecutor() as executor:
        executor.submit(intersection, shapely_poly_list, all_vertices, first_half)
        executor.submit(intersection, shapely_poly_list, all_vertices, second_half)

if __name__ == '__main__':
    coordinate_list = []
    polygon_list = []

    with open('Coordinates', 'r') as file:
        coordinate_info = file.readlines()
        data = [line.strip().split(',') for line in coordinate_info]

    temp_list = []
    polygon_start_index = 2
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

    weight_matrix, max_width         = create_matrix(coordinate_list)
    vertices_list, vertices_list_t   = buffered_point(polygon_list)
    vertices_list_t.insert(0, coordinate_list[0])
    vertices_list_t.insert(1, coordinate_list[1])

    shapely_polygon_list             = shapely_polygon(vertices_list)
    linestring_list                  = create_linestring(vertices_list_t)
    process_line(shapely_polygon_list, vertices_list_t, linestring_list)
    
    graph = create_graph_from_matrix(weight_matrix)
    path = shortest_path(graph, 0, 1)
    pd_path_coord = []
    for vertex in path:
        pd_path_coord.append(vertices_list_t[vertex])

    line = LineString(pd_path_coord)

    start_target_point = [Point(vertices_list_t[0]), Point(vertices_list_t[1])]

    data = gpd.GeoDataFrame(geometry=[polygon for polygon in shapely_polygon_list] + [line] + [point for point in start_target_point])
    fig, ax = plt.subplots()
    data[data.geometry.type == 'Polygon'].plot(ax=ax, color='lightblue', edgecolor='black')
    data[data.geometry.type == 'LineString'].plot(ax=ax, color='red')
    data[data.geometry.type == 'Point'].plot(ax=ax, color='green')
    plt.show()
    #print(f"{len(shapely_polygon_list)} poligon  {len(vertices_list_t)} nokta")
    #show_matrix(weight_matrix, max_width)
