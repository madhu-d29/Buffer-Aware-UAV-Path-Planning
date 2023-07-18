import matplotlib.pyplot as plt
import numpy as np
import random
import math
import xlrd
import xlwt
wrt=xlwt.Workbook(encoding="utf-8")
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

Visit_sequence=[]
#added for adjust function only
visit_sequence=[]
buffer_status =[]
center_list =[]
radii=[]
#Adjust function parameters end

max_round=1
search_time=1
service_time=5
nw_lifetime=0
sheet_name="data_gen_rate"
sheet=wrt.add_sheet(sheet_name)
f_alive = open("alive_round.txt","w")
f_data = open("data_collection.txt","w")
f_energy=open("Avg_energy.txt","w")
f_rounds=open("rounds.txt","w")


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.isAlive=1
        self.res_energy=E_init
        self.isRP = False
        self.RP_id=id
        self.service=0
        self.visited = False
        self.data_size=200
        self.buffer_size=3000
        self.buffer_status=0
        self.data_gen_rate=10
        self.data_timer=20
        self.deadline=200
        self.neighbors = []
        self.RP_members=[]
class UAV:
    def __init__(self,dist_capacity,velocity,height):
        self.dist_capacity=dist_capacity
        self.velocity=velocity
        self.height=height
        self.data_collected=0

#Used for RP information storing
data= {
        'locations':[],
        'time_matrix':[],
        'time_windows':[],
        'num_vehicles':1,
        'depot':0,
        'id':[],
        };

# Parameters
m = 600 # number of nodes
R_com=15 # communication range of node

E_init = 1.0 # initial energy of node
k_CP= 100 # control packet lenght
E_th = 0.0001  #Threshold value for residual energy
E_elec = 0.000000050;
E_fs = .00000000001
E_DA = .00000000005
#E_sensing = 0
E_mp = .0000013
E_TX =0.0 # transmision energy variable
E_RX = 0.0 # receiving energy variable
'''
x_range = range(0, 100)
y_range = range(0, 100)

# Specify the number of coordinates you want to generate
num_coordinates = 199

# Generate random coordinates
coordinates=[(50,50)]
for _ in range(num_coordinates):
    coordinates.append((random.choice(x_range), random.choice(y_range)))


# Initialize nodes with the given coordinates
nodes = [Node(i, x, y) for i, (x, y) in enumerate(coordinates)]
'''
read = xlrd.open_workbook('Nodes_position_new2.xls')
x_pos=read.sheet_by_index(0)
y_pos=read.sheet_by_index(1)
X=x_pos.col_values(0)
Y=y_pos.col_values(0)

# Specify the number of coordinates you want to generate
num_coordinates = m

# Generate random coordinates
coordinates=[]
for i in range(num_coordinates):
    coordinates.append((X[i],Y[i]))
# Initialize nodes with the given coordinates
nodes = [Node(i, x, y) for i, (x, y) in enumerate(coordinates)]

uav=UAV(200,5,10)
R_max = (R_com**2-uav.height**2)**(1/2)
Dist_matrix = [[0 for x1 in range(m)] for y1 in range(m)]
for i in range (0,m):
    for j in range (0,m):
        if (i == j):
            Dist_matrix[i][j] = 0
        elif (i < j):
            Dist_matrix[i][j] = ((nodes[i].x - nodes[j].x)**(2) + (nodes[i].y - nodes[j].y)**(2))**(1/2.0)
        else:
            Dist_matrix[i][j] = Dist_matrix[j][i] 
            
def find_neighbors(node, nodes, R_max):
    neighbors = []
    del neighbors[:]
    if node.isAlive:
        for other_node in nodes:
            if node != other_node and other_node.isRP == False and other_node.isAlive==True:
                dist = np.sqrt((node.x - other_node.x)**2 + (node.y - other_node.y)**2)
                if dist < R_max:
                    neighbors.append(other_node)
    else:
        del node.neighbors[:]
    for node in nodes:
        if node.isAlive:
            E_TX = k_CP*(E_elec + E_fs*(R_max**2))
            node.res_energy=node.res_energy-E_TX
            
    return neighbors

def find_RP_members(node, nodes, R_max):
    RP_mem= []
    del RP_mem[:]
    if node.isAlive:
        for other_node in nodes:
            if node != other_node and other_node.isRP == False and other_node.isAlive==True:
                dist = np.sqrt((node.x - other_node.x)**2 + (node.y - other_node.y)**2)
                if dist < R_max:
                    RP_mem.append(other_node)
    else:
        del node.RP_members[:]       
    return RP_mem

def select_RP(nodes):
    temp_RP = nodes.copy()
    
    while temp_RP:
        for node in temp_RP:
            node.RP_members = find_RP_members(node, temp_RP, R_max)
        
        temp_RP.sort(key=lambda node: len(node.RP_members), reverse=True)
        
        if temp_RP[0].isAlive:
            current_RP = temp_RP[0]
            current_RP.isRP = True
            current_RP.service=service_time

            for neighbor in current_RP.RP_members:
                temp_RP.remove(neighbor)

            temp_RP.remove(current_RP)
        else:
            temp_RP.remove(temp_RP[0])
    
    for node in nodes:
        if node.isRP:
            
            E_TX = k_CP*(E_elec + E_fs*(R_max**2))
            node.res_energy=node.res_energy-E_TX 
            for nd in node.RP_members:
                E_TX = k_CP*(E_elec + E_fs*(Dist_matrix[node.id][nd.id]**2))
                nd.res_energy=nd.res_energy-E_TX
                nd.RP_id=node.id
               

def set_data_rate(nodes):
    data_proximity=20
    data_gen_type=[]
    del data_gen_type[:]
    data_gen_type=[25,30,40,45,50,35,55,70,80,100]
    data_gen_loc_x=[]
    del data_gen_loc_x[:]
    data_gen_loc_x=[random.randint(0,100) for i in range(0,10)]
    data_gen_loc_y=[]
    del data_gen_loc_y[:]
    data_gen_loc_y=[random.randint(0,100) for i in range(0,10)]
    zone=random.randint(0,4)
    zone_id=[]
    del zone_id[:]
    zone_id=[random.randint(0,9) for i in range(0,4)]
    for i in range(1,len(nodes)):
        if nodes[i].isAlive:
            for j in range(0,zone):
                d=np.sqrt((nodes[i].x - data_gen_loc_x[zone_id[j]])**2 + (nodes[i].y - data_gen_loc_y[zone_id[j]])**2)
                if d< data_proximity:
                    nodes[i].data_gen_rate=data_gen_type[zone_id[j]]
                    #print(nodes[i].data_gen_rate)
                    nodes[i].data_size=nodes[i].data_gen_rate*nodes[i].data_timer*random.randint(8,10)/10
        #else:
        #    nodes[i].data_gen_rate=0
        
def set_RP_aggregation(nodes):
    for node in nodes:
        if node.isRP and node.data_gen_rate<=20:
            for neigh in node.RP_members:
                #print (nodes[i].neighbors[j])
                if node.data_size<int(node.buffer_size*0.9):
                    node.data_size=int(node.data_size+neigh.data_size*0.8) #0.8 represents aggregation rate
                    
                    E_TX = neigh.data_size*(E_elec + E_fs*(Dist_matrix[neigh.RP_id][neigh.id]**2))
                    neigh.res_energy=neigh.res_energy-E_TX
                    
                    E_RX = E_elec*neigh.data_size
                    node.res_energy=node.res_energy-E_RX
                    
                    neigh.data_size=0
                
                
                
                
                if node.data_size>node.buffer_size:
                    node.data_size=node.buffer_size
                    node.buffer_status=1
            

def set_node_deadline(nodes):
    for i in range (1, len(nodes)):
        if nodes[i].isAlive:
            nodes[i].deadline=(nodes[i].buffer_size-nodes[i].data_size)/nodes[i].data_gen_rate
        else:
            nodes[i].deadline=-1
 
def find_visit_sequence(data, manager, routing, solution):
    #time_dimension = routing.GetDimensionOrDie('Time')
    #total_time = 0  
    for vehicle_id in range(data['num_vehicles']):
        temp_visit_sequnce=[]
        del temp_visit_sequnce[:]
        temp_visit_sequnce=[0]
        temp_visit_sequnce2=[]
        del temp_visit_sequnce2[:]
        
        index = routing.Start(vehicle_id)
        #plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            #time_var = time_dimension.CumulVar(index)
            #plan_output += '{0} Time({1},{2}) -> '.format(
                #manager.IndexToNode(index), solution.Min(time_var),
                #solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
            temp_visit_sequnce.append(manager.IndexToNode(index))       
        
        for i in range(0, len(temp_visit_sequnce)):
            temp_visit_sequnce2.append(data['id'][temp_visit_sequnce[i]])
            
            
        #time_var = time_dimension.CumulVar(index)
        #plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),solution.Min(time_var),solution.Max(time_var))
        Visit_sequence.append(temp_visit_sequnce2)
        #visit_sequence.append(temp_visit_sequnce)
        #plan_output += 'Time of the route: {}min\n'.format(
            #solution.Min(time_var))
        #print(plan_output)
        #total_time += solution.Min(time_var)
    #print('Total time of all routes: {}min'.format(total_time))
    '''
    # Print neighbors of each RP
    for node in nodes:
        if node.isRP:
            print("RP:", node.id, "Neighbors:", [neighbor.id for neighbor in node.neighbors])
    '''
    # Mark nodes in Visit_sequence and their neighbors as visited
    for seq in Visit_sequence:
        for node_id in seq:
            nodes[node_id].visited = True
            uav.data_collected=nodes[node_id].data_size+uav.data_collected
            E_TX = nodes[node_id].data_size*(E_elec + E_fs*(uav.height**2))
            nodes[node_id].res_energy=nodes[node_id].res_energy-E_TX
            nodes[node_id].data_size=0
            for neighbor in nodes[node_id].RP_members:
                neighbor.visited = True
                uav.data_collected=neighbor.data_size+uav.data_collected
                E_TX = neighbor.data_size*(E_elec + E_fs*(Dist_matrix[neighbor.id][neighbor.RP_id]**2))
                neighbor.res_energy=neighbor.res_energy-E_TX
                neighbor.data_size=0  
    
    '''            
    
    for seq in Visit_sequence:
        centers.append([seq.x])
    #del centers[:]
    #centers=[0 for _ in range(len(temp_visit_sequnce))]
    print (len(temp_visit_sequnce))
    for i in range(0, len(temp_visit_sequnce)):
        centers.append([0,0])
    for i in range(0, len(temp_visit_sequnce)):
        print (temp_visit_sequnce[i])
        centers[temp_visit_sequnce[i]]=[nodes[data['id'][temp_visit_sequnce[i]]].x,nodes[data['id'][temp_visit_sequnce[i]]].y]
                                    
    '''
'''                   
def plot(nodes):
    plt.figure(figsize=(8, 8))

    for node in nodes:
        if node == nodes[0]:  # checks if node is the first one
            plt.plot(node.x, node.y, 'gs')  # 'gs' stands for green square
        elif node.isRP:
            plt.plot(node.x, node.y, 'ro')
            for neighbor in node.RP_members:
                if not neighbor.isRP:
                    plt.plot([node.x, neighbor.x], [node.y, neighbor.y], 'b--')
                    plt.plot(neighbor.x, neighbor.y, 'bo')

    plt.show()
'''
def plot(nodes):
    plt.figure(figsize=(8, 8))

    for node in nodes:
        if node == nodes[0]:  # checks if node is the first one
            plt.plot(node.x, node.y, 'o',color="#FF1493")  # 'gs' stands for green square
        elif node.isRP:
            #plt.plot(node.x, node.y, 'ro')
            for neighbor in node.RP_members:
                if not neighbor.isRP:
                    plt.plot([node.x, neighbor.x], [node.y, neighbor.y], '-',color="#B4CDCD")
                    plt.plot(neighbor.x, neighbor.y, 'o',color="#009ACD")
            plt.plot(node.x, node.y, 'o',color="#FF1493")
    plt.xlabel('X-Coordinate',fontsize=30)
    plt.ylabel('Y-Coordinate',fontsize=30)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig("RP600.pdf",bbox_inches='tight')
    plt.show()
    
def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0  
    for vehicle_id in range(data['num_vehicles']):
        #temp_visit_sequnce=[]
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
            #temp_visit_sequnce.append(manager.IndexToNode(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        #Visit_sequence.append(temp_visit_sequnce)
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        print(plan_output)
        total_time += solution.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))



def plot_solution(data, manager, routing, solution):
    """Plots the solution on a 2D graph."""
    plt.figure(figsize=(10, 10))
    # Plot locations.
    for i, location in enumerate(data['locations']):
        plt.scatter(location[0], location[1], c='#B4CDCD', s=20,marker="s")
        plt.text(location[0]+1, location[1]+1, str(data['id'][i]), fontsize=8, color='black')
    for node in nodes:
        if node.isRP:
            for neighbor in node.RP_members:
                if not neighbor.isRP:
                    plt.plot([node.x, neighbor.x], [node.y, neighbor.y], '-',color="#B4CDCD")
                    plt.plot(neighbor.x, neighbor.y, 'o',color="#B4CDCD")
    # Plot the routes.
    time_dimension = routing.GetDimensionOrDie('Time')
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        del route[:]
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(data['locations'][node])
            index = solution.Value(routing.NextVar(index))
        node = manager.IndexToNode(index)
        route.append(data['locations'][node])
        plt.plot([loc[0] for loc in route], [loc[1] for loc in route], linestyle='-')
        
        for i in range(len(route)-1):
            plt.annotate('', xytext=route[i], xy=route[i+1], arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        
    # Mark the visited nodes with a diff color 
    for node_index in route:
        plt.scatter(node_index[0], node_index[1], c='#FF1493', s=200, marker='o') 

    #plt.title('Routes') 
    #plt.xlabel('X')
    #plt.ylabel('Y') 
    plt.xlabel('X-Coordinate',fontsize=30)
    plt.ylabel('Y-Coordinate',fontsize=30)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.axis('square')
    plt.savefig("Path600.pdf")
    plt.show()
    
def path_formation():
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
   # data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])
    #print (manager)
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    #print(routing)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        100,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add a large penalty for violating the time window.
    penalty = 1000000

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddDisjunction([index], penalty)  # This line is new.

    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

     # Setting first solution heuristic and local search metaheuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = search_time

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        find_visit_sequence(data, manager, routing, solution) 
        #print_solution(data, manager, routing, solution)
        plot_solution(data, manager, routing, solution)
#Visit_sequence=[]

def adjust_path():
    current_position =np.array([nodes[0].x,nodes[0].y])
    # Initialize the path length and plots
    path_length = 0
    circle_plots = []
    center_plots = []
    path_plots = []
    
    # Initialize service times for all points
    #service_times = [0 for _ in range(len(Visit_sequence))]
    #service_times = [5 for i in range(1,len(Visit_sequence)-1)] 
    
    # Define a function to calculate the distance between two points
    def distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    # Define a function to find the nearest point on a circle to a given point
    def find_nearest_point(center, radius, point):
        d = distance(center, point)
        x = center[0] + radius*(point[0]-center[0])/d
        y = center[1] + radius*(point[1]-center[1])/d
        return np.array([x,y])

    # Define a function to calculate the angle between three points
    def angle(point1, center, point2):
        point1=np.array(point1)
        point2=np.array(point2)
        v1 = point1 - center
        v2 = point2 - center
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:  # checking if norms are not zero
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1, 1)  # ensure the value is within the valid range
            return np.arccos(cos_angle)
        else:
            return np.nan  # or some default value
        
    chord_lengths = [0 for _ in range(len(Visit_sequence[0]))]
    travel_times = [0 for _ in range(len(Visit_sequence[0]))]
    
    for i in range(1, len(Visit_sequence[0])):
        next_center = [nodes[Visit_sequence[0][i]].x,nodes[Visit_sequence[0][i]].y]
        
        if nodes[Visit_sequence[0][i]].data_size>=nodes[Visit_sequence[0][i]].buffer_size:
            extra_travel_time=2*R_max/uav.velocity
            if nodes[Visit_sequence[0][i]].service <= extra_travel_time:
                nodes[Visit_sequence[0][i]].service=0
            else:
                nodes[Visit_sequence[0][i]].service=nodes[Visit_sequence[0][i]].service-extra_travel_time
        else:
            next_point =find_nearest_point(next_center,R_max,current_position)
            
         # Calculate the chord length if the angle is greater than 90 degrees
        angle_rad = angle(current_position, [nodes[Visit_sequence[0][i-1]].x, nodes[Visit_sequence[0][i-1]].y], next_center)
        if angle_rad > np.pi / 2:
            start_point = find_nearest_point([nodes[Visit_sequence[0][i-1]].x, nodes[Visit_sequence[0][i-1]].y], R_max, current_position)
            end_point = find_nearest_point([nodes[Visit_sequence[0][i-1]].x, nodes[Visit_sequence[0][i-1]].y], R_max, next_center)
            chord_length = distance(start_point, end_point)
            chord_lengths[Visit_sequence[0][i-1]] = chord_length
            chord_travel_time = chord_length /uav.velocity
            travel_times[Visit_sequence[0][i-1]] = chord_travel_time
            
            # Deduct service time for the chord
            if nodes[Visit_sequence[0][i-1]].service <= chord_travel_time:
                nodes[Visit_sequence[0][i-1]].service = 0
            else:
                nodes[Visit_sequence[0][i-1]].service -= chord_travel_time
                
        path_length += distance(current_position, next_point)
        path_plots.append(plt.plot([current_position[0], next_point[0]], [current_position[1], next_point[1]], color='g'))
        current_position = next_point
        
        # Plot the current circle and center
        circle_plots.append(plt.gca().add_artist(plt.Circle([nodes[Visit_sequence[0][i]].x,nodes[Visit_sequence[0][i]].y], R_max, color='b', fill=False)))
        center_plots.append(plt.scatter(nodes[Visit_sequence[0][i]].x,nodes[Visit_sequence[0][i]].y, color='b'))
    
    # Create ordered lists based on visit sequence
    #service_times_ordered = [nodes[Visit_sequence[0][i]].service for i in range (0,len(Visit_sequence[0]))]
    #chord_lengths_ordered = [round(chord_lengths[i],2) for i in range (0,len(Visit_sequence[0])-1)]
    #travel_times_ordered = [round(travel_times[i],2) for i in range (0,len(Visit_sequence[0])-1)]
     
    # What is printed?
    '''
    print("Path length:", path_length)
    print("Chord lengths (order of visit):", chord_lengths_ordered)
    print("Travel time for each chord (order of visit):", travel_times_ordered)
    print("Service times (order of visit):", service_times_ordered)
    
    plt.axis('equal')
    plt.show()
    '''
            
def adjust_path2(uav_index):  
     #centers = np.array([[40,50],[42,66],[20,80],[28,52],[8,40],[33,32],[42,10],[50,35],[92,30],[68,60],[62,80]])
     #radii = np.array([0, 4,4,4,4,4,4,4,4,4,4])
     
     # Define the buffer status of the circles
     '''
     buffer_status = []
     del buffer_status[:]
     for i in range(0, len(Visit_sequence[uav_index])):
         centers.append([nodes[Visit_sequence[uav_index][i]].x, nodes[Visit_sequence[uav_index][i]].y] )
     # Define the visit sequence
     visit_sequence = Visit_sequence[0]
     
     for i in range(0,len(visit_sequence)-1):
         buffer_status[i]=nodes[visit_sequence[i]].buffer
     for i in range (0, len(visit_sequence)-1):
         centers.append([nodes[visit_sequence[i]].x,nodes[visit_sequence[i]].y])
         radii.append(R_max)
         
     '''  
     fig = plt.figure(figsize=(10, 10))
     for i in range(0, len(Visit_sequence[uav_index])-1):
          center_list.append([int(nodes[Visit_sequence[uav_index][i]].x), int(nodes[Visit_sequence[uav_index][i]].y)]) 
          radii.append(R_max)
          if i==0:
              buffer_status.append(1)
          else:
              buffer_status.append(nodes[Visit_sequence[uav_index][i]].buffer_status)
          visit_sequence.append(i)
     visit_sequence.append(0)
     radii[0]=0
     
     centers=np.array(center_list)
     # Initialize the current position to the center of the first circle
     current_position = centers[visit_sequence[0]]

     # Initialize the path length and plots
     path_length = 0
     circle_plots = []
     center_plots = []
     path_plots = []

     # Define the UAV velocity
     uav_velocity = uav.velocity  # in units per second

     # Initialize service times for all points
     service_times = [5 for _ in range(len(centers))]

     # Define a function to calculate the distance between two points
     def distance(point1, point2):
         return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

     # Define a function to find the nearest point on a circle to a given point
     def find_nearest_point(center, radius, point):
         d = distance(center, point)
         x = center[0] + radius*(point[0]-center[0])/d
         y = center[1] + radius*(point[1]-center[1])/d
         return np.array([x,y])

     # Define a function to calculate the angle between three points
     def angle(point1, center, point2):
         v1 = point1 - center
         v2 = point2 - center
         norm_v1 = np.linalg.norm(v1)
         norm_v2 = np.linalg.norm(v2)
         if norm_v1 > 0 and norm_v2 > 0:  # checking if norms are not zero
             cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
             cos_angle = np.clip(cos_angle, -1, 1)  # ensure the value is within the valid range
             return np.arccos(cos_angle)
         else:
             return np.nan  # or some default value


     # Initialize chord lengths and travel times lists
     chord_lengths = [0 for _ in range(len(centers))]
     travel_times = [0 for _ in range(len(centers))]

     # Iterate through the visit sequence
     for i in range(1, len(visit_sequence)):
         next_circle_index = visit_sequence[i]
         next_center = centers[next_circle_index]

         # Find the next point to visit
         if buffer_status[next_circle_index] == 1:
             next_point = centers[next_circle_index]
             extra_travel_time = 2*radii[next_circle_index] / uav_velocity

             if service_times[next_circle_index] <= extra_travel_time:
                 service_times[next_circle_index] = 0
             else:
                 service_times[next_circle_index] -= extra_travel_time
         else:
             next_point = find_nearest_point(next_center, radii[next_circle_index], current_position)
         
         # Calculate the chord length if the angle is greater than 90 degrees
         angle_rad = angle(current_position, centers[visit_sequence[i-1]], next_center)
         if angle_rad > np.pi / 2:
             start_point = find_nearest_point(centers[visit_sequence[i-1]], radii[visit_sequence[i-1]], current_position)
             end_point = find_nearest_point(centers[visit_sequence[i-1]], radii[visit_sequence[i-1]], next_center)
             chord_length = distance(start_point, end_point)
             chord_lengths[visit_sequence[i-1]] = chord_length
             chord_travel_time = chord_length / uav_velocity
             travel_times[visit_sequence[i-1]] = chord_travel_time

             # Deduct service time for the chord
             if service_times[visit_sequence[i-1]] <= chord_travel_time:
                 service_times[visit_sequence[i-1]] = 0
             else:
                 service_times[visit_sequence[i-1]] -= chord_travel_time

         path_length += distance(current_position, next_point)
         path_plots.append(plt.plot([current_position[0], next_point[0]], [current_position[1], next_point[1]], color='blue'))
         current_position = next_point

         # Plot the current circle and center
         circle_plots.append(plt.gca().add_artist(plt.Circle(centers[next_circle_index], radii[next_circle_index], linestyle='--',color='#838B8B', fill=False)))
         center_plots.append(plt.scatter(centers[next_circle_index][0], centers[next_circle_index][1], color='#FF1493'))

         # Annotate node numbers on the plot
         plt.text(centers[next_circle_index][0], centers[next_circle_index][1], str(Visit_sequence[uav_index][next_circle_index]), color='k', fontsize=11)

     service_times[0]=0

     # Create ordered lists based on visit sequence
     #service_times_ordered = [service_times[i] for i in visit_sequence]
     #chord_lengths_ordered = [round(chord_lengths[i],2) for i in visit_sequence]
     #travel_times_ordered = [round(travel_times[i],2) for i in visit_sequence]

     # What is printed? 
     '''
     print("Path length:", path_length)
     print("Chord lengths (order of visit):", chord_lengths_ordered)
     print("Travel time for each chord (order of visit):", travel_times_ordered)
     print("Service times (order of visit):", service_times_ordered)
     '''
     plt.xlabel('X-Coordinate',fontsize=30)
     plt.ylabel('Y-Coordinate',fontsize=30)
     
     plt.xticks(fontsize=30)
     plt.yticks(fontsize=30)
     plt.axis('square')
     fig.savefig("Adj_Path600.pdf")
     plt.show()
     
     

def rounds(num_of_rounds):
    '''
    dr = xlrd.open_workbook('Data_generation_rate.xls')
    rd=dr.sheet_by_index(0)
    d=rd.row_values(num_of_rounds)
    '''
    i=0
    for node in nodes:
        if node.res_energy>E_th:
            node.isRP=False
            node.RP_id=node.id
            node.visited = False
            node.data_size=200
            node.data_gen_rate=10
            node.deadline=200
            del node.neighbors[:]
            del node.RP_members[:]
        else:
            node.isAlive=False
        i=i+1
    nodes[0].isRP=1 
    
       
    del Visit_sequence[:]
    del visit_sequence[:]
    del center_list[:]
    del buffer_status[:]
    del radii[:]
    
    data['locations']=[]
    data['time_matrix']=[]
    data['time_windows']=[]
    data['id']=[]
    
    for node in nodes:
        if node.isAlive:
            node.neighbors = find_neighbors(node, nodes, R_max)
        else:
            del node.neighbors[:]
    
    # Select RPs
    select_RP(nodes)
    set_data_rate(nodes)
    set_RP_aggregation(nodes)
    set_node_deadline(nodes)
    # Plot the result
    
    # Print x,y coordinates of RPs
    
    count=0
    list_RP=[]
    del list_RP[:]
    for node in nodes:
        if node.isRP and node.isAlive:
            #print(f'Node ID: {node.id}, Coordinates: ({node.x}, {node.y})')
            list_RP.append(node.id) 
            count+=1
    if num_of_rounds==1:
        plot(nodes)
        
    print(count)
    #print(uav.velocity)
    num_RP=count
    time_RP=[[0 for x1 in range(num_RP)] for y1 in range(num_RP)]
    Dist_RP = [[0 for x1 in range(num_RP)] for y1 in range(num_RP)]
    for i in range (0,num_RP):
        for j in range (0,num_RP):
            if (i == j):
                Dist_RP[i][j] = 0
                time_RP[i][j]=int (Dist_RP[i][j]/uav.velocity)
            elif (i < j):
                Dist_RP[i][j] = np.sqrt((nodes[list_RP[i]].x - nodes[list_RP[j]].x)**2 + (nodes[list_RP[i]].y - nodes[list_RP[j]].y)**2)
                time_RP[i][j]=int (Dist_RP[i][j]/uav.velocity)
            else:
                Dist_RP[i][j] = Dist_RP[j][i]
                time_RP[i][j]=int (Dist_RP[i][j]/uav.velocity)
                
    
    
    data['time_matrix']=time_RP
    #data['locations'].append((25,25))
    #data['time_windows'].append((0,1000))
    for node in nodes:
        if node.isRP and node.isAlive:
            data['locations'].append((node.x,node.y))
            data['id'].append(node.id)
            data['time_windows'].append((0,int(node.deadline)))
    data['time_windows'][0]=(0,1000)        
    
    path_formation()
    
    for i in range (0,len(Visit_sequence)): 
        adjust_path2(i)
     
    #Add energy penalty for unvisited RPs

    for i in range (0, len(data['id'])):
        if nodes[data['id'][i]].visited==0:
            for RP_temp in nodes[data['id'][i]].RP_members:
                E_TX=RP_temp.data_size*(E_elec + E_fs*(Dist_matrix[nodes[data['id'][i]].id][RP_temp.id]**2))
                RP_temp.res_energy=RP_temp.res_energy - E_TX
                
                E_RX = E_elec*RP_temp.data_size
                nodes[data['id'][i]].res_energy=nodes[data['id'][i]].res_energy-E_RX
    
    for i in range (0,m):
        item=nodes[i].data_gen_rate
        sheet.write(i,num_of_rounds,int(item))
    file_name="Data_generation_rate.xls"
    wrt.save(file_name)
    
    total_data_missed=0
    total_alive_nodes=0
    total_energy=0
    
    for node in nodes:
        if not node.visited:
            #print (node.id)
            total_data_missed=node.data_size*0.95+total_data_missed
        if node.isAlive:
            total_alive_nodes=total_alive_nodes+1
            total_energy=total_energy+node.res_energy
            
    #print (uav.data_collected, total_data_missed)      
    f_alive.write(str(total_alive_nodes))
    f_alive.write("\n")
    f_data.write(str(uav.data_collected/(uav.data_collected+total_data_missed)))
    f_data.write("\n")
    f_energy.write(str(total_energy/m))
    f_energy.write("\n")
    f_rounds.write(str(num_of_rounds))
    f_rounds.write("\n")
    uav.data_collected=0   
    print("Round number: ")
    print(num_of_rounds)     
    if(num_of_rounds<max_round):
        rounds(num_of_rounds+1)
rounds(1)  

f_alive.close()
f_data.close()  
f_energy.close()       
