#lang dssl2

# Final project: Trip Planner
let eight_principles = ["Know your rights.", "Acknowledge your sources.", "Protect your work.", "Avoid suspicion.", "Do your own work.", "Never falsify a record or permit another person to do so.", "Never fabricate data, citations, or experimental results.", "Always tell the truth when discussing your work with your instructor."]
###
import cons
import 'project-lib/dictionaries.rkt'
import 'project-lib/graph.rkt'
import 'project-lib/binheap.rkt'
  ##i imported
import sbox_hash
### Basic Types ###

#  - Latitudes and longitudes are numbers:
let Lat?  = num?
let Lon?  = num?

#  - Point-of-interest categories and names are strings:
let Cat?  = str?
let Name? = str?

### Raw Item Types ###

#  - Raw positions are 2-element vectors with a latitude and a longitude
let RawPos? = TupC[Lat?, Lon?]

#  - Raw road segments are 4-element vectors with the latitude and
#    longitude of their first endpoint, then the latitude and longitude
#    of their second endpoint
let RawSeg? = TupC[Lat?, Lon?, Lat?, Lon?]

#  - Raw points-of-interest are 4-element vectors with a latitude, a
#    longitude, a point-of-interest category, and a name
let RawPOI? = TupC[Lat?, Lon?, Cat?, Name?]

### Contract Helpers ###

# ListC[T] is a list of `T`s (linear time):
let ListC = Cons.ListC
# List of unspecified element type (constant time):
let List? = Cons.list?


####my work


# Define the struct for mapping positions to vertices and vice versa
struct PositionMap:
    let graph
    let position_to_node_id
    let node_id_to_position


#####




interface TRIP_PLANNER:

    # Returns the positions of all the points-of-interest that belong to
    # the given category.
    def locate_all(
            self,
            dst_cat:  Cat?           # point-of-interest category
        )   ->        ListC[RawPos?] # positions of the POIs

    # Returns the shortest route, if any, from the given source position
    # to the point-of-interest with the given name.
        
    def plan_route( self, src_lat:  Lat?,          # starting latitude 
    src_lon:  Lon?,          # starting longitude
     dst_name: Name? )   ->        ListC[RawPos?] # path to goal
     
   # Finds no more than `n` points-of-interest of the given category
    # nearest to the source position.
    def find_nearby(
            self,
            src_lat:  Lat?,          # starting latitude
            src_lon:  Lon?,          # starting longitude
            dst_cat:  Cat?,          # point-of-interest category
            n:        nat?           # maximum number of results
        )   ->        ListC[RawPOI?] # list of nearby POIs


class TripPlanner (TRIP_PLANNER):
    let position_map
    let poi_hash_table
    let raw_points_of_interest
    
    
    def __init__(self, raw_road_segments, raw_points_of_interest):
        
        # Step 1: Process the raw road segments to create the position mappings
        self.position_map = self.process_raw_road_segments(raw_road_segments)
        
        # Step 2: Add edges to the graph using the position-to-vertex mappings
        self.populate_graph_with_edges(raw_road_segments)
        
        
        # Initialize the hash table for POIs
        self.poi_hash_table = HashTable(len(raw_points_of_interest), make_sbox_hash())  # Hash table for POIs
        self.initialize_poi_hash_table(raw_points_of_interest)
        
        self.raw_points_of_interest=raw_points_of_interest
        
        
        
        
    ###now helpers to setup poi hash
    def initialize_poi_hash_table(self, raw_pois):
    # Iterate over the raw POIs and add them to the hash table
        for raw_poi in raw_pois:
        # Extract the category from the raw POI
            let category = raw_poi[2]

        # Check if the category already exists in the hash table
            if not self.poi_hash_table.mem?(category):
            # If not, initialize a new list (or equivalent structure) for this category
                self.poi_hash_table.put(category, cons(raw_poi, None))  # Start a new list with raw_poi
            else:
                # If the category exists, retrieve the existing list and add the new POI
                let existing_pois = self.poi_hash_table.get(category)
                let new_pois = cons(raw_poi, existing_pois)  # Create a new list with raw_poi at the front
                self.poi_hash_table.put(category, new_pois)  # Update the list in the hash table


    
        
        
        
        
   ######
        
        
    ####setting up grapgh helpers
        
    def process_raw_road_segments(self, raw_road_segments):
        let unique_positions_list = self.find_unique_positions(raw_road_segments)    
    
    
    
        let position_to_node_id = AssociationList()  # Setup association lists in struct
        let node_id_to_position = AssociationList()
    
    # Populate the dictionaries with positions and corresponding node IDs
        let current = unique_positions_list
        let node_id = 0
        while current is not None:
            position_to_node_id.put(current.data, node_id)
            node_id_to_position.put(node_id, current.data)
            current = current.next
            node_id = node_id + 1
            
        let graph = WUGraph(node_id+1)  ###intitializze grpagh
    
        return PositionMap(graph, position_to_node_id, node_id_to_position)

        
    ####get list of unique points
    def find_unique_positions(self, raw_road_segments):
        let count=0
        let unique_positions = None  # This will store the unique positions as a linked list

    # Function to check if the position is already in the unique_positions list
        def is_unique(position, unique_positions):
            let current = unique_positions
            while current is not None:
                if current.data[0] == position[0] and current.data[1] == position[1]:
                    return False  # Position already in list
                current = current.next
            return True  # Position not in list

    # Iterate through each road segment
        for segment in raw_road_segments:
        # Extract the start and end positions
            let start_position = [segment[0], segment[1]]
            let end_position = [segment[2], segment[3]]

        # If the start position is unique, add it to the list
            if is_unique(start_position, unique_positions):
                unique_positions = cons(start_position, unique_positions)
                count=count+1

        # If the end position is unique, add it to the list
            if is_unique(end_position, unique_positions):
                unique_positions = cons(end_position, unique_positions)
                count=count+1

        return unique_positions


    

    def populate_graph_with_edges(self, raw_road_segments):
        for segment in raw_road_segments:
            let start_pos = [segment[0], segment[1]]
            let end_pos = [segment[2], segment[3]]
            let start_node_id = self.position_map.position_to_node_id.get(start_pos) ###extracting node id that maps the position to a vertex
            let end_node_id = self.position_map.position_to_node_id.get(end_pos)
            let weight = self.calculate_distance(start_pos, end_pos)
            
            self.position_map.graph.set_edge(start_node_id, end_node_id, weight)

    def calculate_distance(self, pos1, pos2):
            # Extract the latitude and longitude from the positions
        let lat1 = pos1[0] 
        let lon1 = pos1[1]
        let lat2=pos2[0]
        let lon2 = pos2[1]
    
        # Calculate the differences
        let delta_lat = lat2 - lat1
        let delta_lon = lon2 - lon1
    
        # Use the Pythagorean theorem to calculate Euclidean distance
        let distance = (delta_lat**2 + delta_lon**2)**0.5
        return distance
        
        ####end of functions that help setup my map
        
    #belman for plan route
        
   
    def bellman_ford(self, src_vertex, dst_vertex):
        let MAX_VALUE = 999999
        # Initialize arrays with DSSL2 syntax
        let dist = [MAX_VALUE; self.position_map.graph.len()]
        let prev = [None; self.position_map.graph.len()]

        dist[src_vertex] = 0

        # Relax all edges V - 1 times
        for _ in range(self.position_map.graph.len() - 1):
            for u in range(self.position_map.graph.len()):
                # Iterate over the linked list of adjacent vertices
                let current_adjacent = self.position_map.graph.get_adjacent(u)
                while current_adjacent is not None:
                    let v = current_adjacent.data
                    if dist[u] + self.position_map.graph.get_edge(u, v) < dist[v]:
                        dist[v] = dist[u] + self.position_map.graph.get_edge(u, v)
                        prev[v] = u
                    current_adjacent = current_adjacent.next

        # Check for negative weight cycles
        for u in range(self.position_map.graph.len()):
            let current_adjacent = self.position_map.graph.get_adjacent(u)
            while current_adjacent is not None:
                let v = current_adjacent.data
                if dist[u] + self.position_map.graph.get_edge(u, v) < dist[v]:
                    print("Graph contains a negative weight cycle")
                    return None
                current_adjacent = current_adjacent.next
                
        # Check if the destination is reachable
        if dist[dst_vertex] == MAX_VALUE:
            return None  # Destination is unreachable

        let path = None
        let u = dst_vertex
        while u is not None:
            let position = self.position_map.node_id_to_position.get(u)
            path = cons(position, path)  # Insert at the front of the linked list
            u = prev[u]

        return path
        
        
       
        
        
    #methods
        
    # Helper method to check if a position is already in the list
    def is_position_in_list(self, position, lst):
        let current = lst
        while current is not None:
            if current.data[0] == position[0] and current.data[1] == position[1]:
                return True  # Position found in the list
            current = current.next
        return False  # Position not found in the list

    def locate_all(self, dst_cat: Cat?) -> ListC[RawPos?]:
        # Initialize an empty linked list for storing positions
        let positions = None

        # Check if the specified category exists in the hash table
        if self.poi_hash_table.mem?(dst_cat):
            # Retrieve the linked list of POIs for the specified category
            let pois = self.poi_hash_table.get(dst_cat)

            # Iterate over each POI in the linked list
            while pois is not None:
                # Extract the position from the current raw POI
                let position = [pois.data[0], pois.data[1]]

                # Check if this position is already in the positions list
                # If not, add the position to the front of the list
                if not self.is_position_in_list(position, positions):
                    positions = cons(position, positions)
                
                # Move to the next POI in the list
                pois = pois.next

        # Return the linked list of positions
        return positions
        
    def plan_route(self, src_lat:  Lat?, src_lon:  Lon?, dst_name: Name? )  -> ListC[RawPos?]:
        # Initialize variables for the source and destination vertices
        let src_vertex = None
        let dst_vertex = None

    # Find the source vertex using the source latitude and longitude
        src_vertex = self.position_map.position_to_node_id.get([src_lat, src_lon])

    # Check if the source vertex is valid
        if src_vertex is None:
        # Handle the case where the starting position is not found
            return None  # or an appropriate error handling
            
            
        # Iterate over the raw POIs to find the destination vertex by name
        for poi in self.raw_points_of_interest:
            if poi[3] == dst_name:  # poi[3] is the name of the POI
                dst_vertex = self.position_map.position_to_node_id.get([poi[0], poi[1]])
                break

    # Check if the destination vertex is valid
        if dst_vertex is None:
        # Handle the case where the destination POI is not found
            return None  # or an appropriate error handling
            
       #at this point we have both vertexes in hand now we need to find shortest path
            
        return self.bellman_ford(src_vertex, dst_vertex)   #return bellman funshun
        
        
      ##3rd one
    
    # Helper method to sort POIs by distance (since DSSL2 does not support native sorting)
    def sort_pois_by_distance(self, pois):
        if pois is None or pois.next is None:
            return pois
    
        let sorted = None
        let current = pois
        while current is not None:
            let temp = current
            current = current.next
            sorted = self.insert_in_sorted_order(sorted, temp)

        return sorted

# Helper method to insert a POI into a sorted linked list
    def insert_in_sorted_order(self, sorted_list, poi):
        if sorted_list is None or sorted_list.data[0] > poi.data[0]:
            poi.next = sorted_list
            return poi

        let current = sorted_list
        while current.next is not None and current.next.data[0] < poi.data[0]:
            current = current.next

        poi.next = current.next
        current.next = poi
        return sorted_list
        
    def calculate_path_length(self, path):
        if path is None or path.next is None:
            return 0  # Path is empty or has only one vertex

        let total_length = 0
        let current = path

        while current.next is not None:
            let current_pos = current.data
            let next_pos = current.next.data

        # Use the calculate_distance method to get the distance between consecutive vertices
            total_length = total_length+ self.calculate_distance(current_pos, next_pos)

            current = current.next

        return total_length
        
    
        
    def find_nearby(self, src_lat: Lat?, src_lon: Lon?, dst_cat: Cat?, n: int?) -> ListC[RawPOI?]:
        let src_vertex = self.position_map.position_to_node_id.get([src_lat, src_lon])
        if src_vertex is None:
            return None  # Starting position is not on the graph

        let pois_in_category = self.poi_hash_table.get(dst_cat) if self.poi_hash_table.mem?(dst_cat) else None
        if pois_in_category is None:
            return None  # No POIs in the given category

        let pois_with_path_length = None
        let current_poi = pois_in_category

        while current_poi is not None:
            let poi_vertex = self.position_map.position_to_node_id.get([current_poi.data[0], current_poi.data[1]])
            let path = self.bellman_ford(src_vertex, poi_vertex)
            if path is not None:
                let path_length = self.calculate_path_length(path)  # Get the path length
                pois_with_path_length = cons([path_length, current_poi.data], pois_with_path_length)

            current_poi = current_poi.next

    # Sort the POIs by the length of their shortest path
        let sorted_pois = self.sort_pois_by_distance(pois_with_path_length)  
        let result = None
        let count = 0

        while sorted_pois is not None and count < n:
            result = cons(sorted_pois.data[1], result)
            sorted_pois = sorted_pois.next
            count =count+ 1

        return result
            
        
        
       
        
        
        
        
   ######
        
        
#   ^ YOUR WORK GOES HERE


def my_first_example():
    return TripPlanner([[0,0, 0,1], [0,0, 1,0]],
                       [[0,0, "bar", "The Empty Bottle"],
                        [0,1, "food", "Pierogi"]])

test 'My first locate_all test':
    assert my_first_example().locate_all("food") == \
        cons([0,1], None)

test 'My first plan_route test':
   assert my_first_example().plan_route(0, 0, "Pierogi") == \
       cons([0,0], cons([0,1], None))

test 'My first find_nearby test':
    assert my_first_example().find_nearby(0, 0, "food", 1) == \
        cons([0,1, "food", "Pierogi"], None)
        
        
test 'Locate all non-existent category test':
    assert my_first_example().locate_all("non-existent category") == None

test 'Locate all with multiple POIs in category test':
    let planner = TripPlanner([[0,0, 0,1]], [[0,0, "bar", "Bar1"], [0,1, "bar", "Bar2"]])
    assert planner.locate_all("bar") == cons([0,0], cons([0,1], None))
    
test 'Plan route to non-existent POI test':
    assert my_first_example().plan_route(0, 0, "Non-existent POI") == None

test 'Find nearby with large limit test':
    assert my_first_example().find_nearby(0, 0, "bar", 10) == cons([0,0, "bar", "The Empty Bottle"], None)

test 'Find nearby with no POIs in category test':
    assert my_first_example().find_nearby(0, 0, "cafe", 1) == None

    
#testing
def my_larger_example():
    # Define more complex road segments
    let road_segments = [
        [0,0, 0,1], [0,1, 0,2], [0,2, 1,2],
        [1,2, 1,1], [1,1, 1,0], [1,0, 0,0],
        [0,1, 1,1], [0,2, 1,1]
    ]

    # Define a more diverse set of POIs
    let pois = [
        [0,0, "bar", "Bar A"],
        [0,1, "bar", "Bar B"],
        [1,2, "restaurant", "Restaurant C"],
        [1,1, "restaurant", "Restaurant D"],
        [1,0, "cafe", "Cafe E"],
        [0,2, "cafe", "Cafe F"]
    ]

    return TripPlanner(road_segments, pois)

    
test 'Locate all bars in larger example test':
    let example = my_larger_example()
    assert example.locate_all("bar") == cons([0,0], cons([0,1], None))

    
test 'Locate all in non-existent category in larger example test':
    let example = my_larger_example()
    assert example.locate_all("hotel") == None
    
test 'Plan route to specific POI in larger example test':
    let example = my_larger_example()
    assert example.plan_route(0, 0, "Restaurant C") == cons([0,0], cons([1,0], cons([1,1], cons([1,2], None))))
    
test 'Find nearby cafes in larger example test':
    let example = my_larger_example()
    assert example.find_nearby(1, 1, "cafe", 1) == cons([1,0, "cafe", "Cafe E"], None)

test 'Find nearby restaurants with large limit in larger example test':
    let example = my_larger_example()
    assert example.find_nearby(0, 0, "restaurant", 10) == cons([1,2, "restaurant", "Restaurant C"], cons([1,1, "restaurant", "Restaurant D"], None))
    
test 'unreachable':
    let tp = TripPlanner(
      [[0, 0, 1.5, 0],
       [1.5, 0, 2.5, 0],
       [2.5, 0, 3, 0],
       [4, 0, 5, 0]],
      [[1.5, 0, 'bank', 'Union'],
       [3, 0, 'barber', 'Tony'],
       [5, 0, 'barber', 'Judy']])
    let result = tp.plan_route(0, 0, 'Judy')
    result = Cons.to_vec(result)
    assert result == []
    
    
test 'plannearby failure':
    let tp = TripPlanner(
      [[0, 0, 1.5, 0],
       [1.5, 0, 2.5, 0],
       [2.5, 0, 3, 0],
       [4, 0, 5, 0]],
      [[1.5, 0, 'bank', 'Union'],
       [3, 0, 'barber', 'Tony'],
       [4, 0, 'food', 'Jollibee'],
       [5, 0, 'barber', 'Judy']])
    let result = tp.find_nearby(0, 0, 'barber', 2)
    
    assert result == cons([3, 0, 'barber', 'Tony'], None)
    
test 'plannearby advanced':
    let tp = TripPlanner(
      [[-1.1, -1.1, 0, 0],
       [0, 0, 3, 0],
       [3, 0, 3, 3],
       [3, 3, 3, 4],
       [0, 0, 3, 4]],
      [[0, 0, 'food', 'Sandwiches'],
       [3, 0, 'bank', 'Union'],
       [3, 3, 'barber', 'Judy'],
       [3, 4, 'barber', 'Tony']])
    let result = tp.find_nearby(-1.1, -1.1, 'barber', 1)
   
    assert result == cons([3, 4, 'barber', 'Tony'], None)
    
    

    