# Buffer_and_Route
• First, we import the required libraries and functions.

• The create_matrix function creates a weight matrix based on a list of coordinates. It determines the dimensions of the matrix and the maximum element width.

• The create_graph_from_matrix function creates a graph from the weight matrix. The nodes in the graph correspond to the indices in the matrix, and the edges have the values from the         weight matrix.

• The shortest_path function calculates the shortest path on the given graph. It uses the Dijkstra algorithm.

• The show_matrix function prints the weight matrix on the screen using the matrix and element width.

• The is_vertex_convex function checks whether a vertex of a polygon is convex or not.

• The haversine_distance function calculates the haversine distance between two points.

• The buffered_point function creates new points by buffering each corner of the polygon.

![BUFF](https://github.com/gurbuzkaanakkaya/Buffer_and_Route/assets/103320421/bcf62727-dffc-47eb-80f8-ac0845fa3858)


• The shapely_polygon function creates a Shapely polygon object from the given polygon.

• The create_linestring function generates all possible line combinations of the points and stores them in a dictionary.

• The intersection function performs intersection calculations based on the polygon list, point list, and line list. It updates the weight matrix accordingly.

• The main function starts. We read the coordinate list and polygon list.

• We create the matrix and buffer the point list.

• Shapely polygons are created, and the line list is generated.

• Intersection calculations are performed.

• We create the graph and calculate the shortest path.

• We visualize the results.


![Result](https://github.com/gurbuzkaanakkaya/Buffer_and_Route/assets/103320421/0e21ffcc-0edf-4e77-81bc-d57eb19374ca)
