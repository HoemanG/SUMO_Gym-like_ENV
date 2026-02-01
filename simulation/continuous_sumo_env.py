# Continuous SUMO gym-like env
import os
import sys
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# All vehicle data are based on Vinfast VF9 model
# (https://static-cms-prod.vinfastauto.us/cms-vinfast-us/Specs/VF-9-Spec.pdf)

# GOAL: TO GET THE EGO CAR TO GO ON A RANDOM ROUT WHICH THE LEAST STEP POSSIBLE AND LEAST CO2 EMISSION AND FUEL CONSUMPTION 
# Discrete action space (for ease of implementation =))).), each action lasts for ~10 simulationstep


# --- BOILERPLATE SETUP ---
# check if SUMO is set up appropriately
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("Please declare environment variable 'SUMO_HOME'")



class SumoEnv(gym.Env):
	def __init__(self, render: bool = True, map_config = ["maps/TestMap/osm.sumocfg"], 
			  VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 5.0,
			  test_mode: bool = False, test_route = "TestMap/test_route.rou.xml",
			  imperfection = 0.5, impatience = 0.5, delay = 20) -> None:
		super().__init__()


		self.VEH_ID = "my_ego_car" # if this is changed, change the .rou.xml too
		self.VTYPE_ID = VTYPE_ID
		self.TRAFFIC_SCALE = TRAFFIC_SCALE
		self.render_mode: bool = render
		self.test_mode = test_mode
		self.test_route = test_route
		self.delay = delay
		self.step_count = 0 # Initialize counter
		self.MAX_EPISODE_STEPS = 200 # Force stop after 2000 steps (approx 30 mins sim time)
		# Get the absolute path to the 'simulation' folder
		curr_dir = os.path.dirname(os.path.abspath(__file__))
		# Go up one level to 'Sumo' folder
		root_dir = os.path.dirname(curr_dir)
		
		# --- CENTRALIZED PHYSICS CONSTANTS ---
		self.MAX_SPEED = 55.6 # m/s
		self.MAX_ACCEL = 4.15 # m/s^2
		self.MAX_DECEL = 6.0  # m/s^2
		self.MAX_ELEC = 120   # Wh/s
		self.MAX_SLOPE = 20   # degrees
		self.MAX_DIST = 100.0 # meters


		# Handle string input and fix path
		self.maps = [map_config] if isinstance(map_config, str) else map_config
		self.imperfection = imperfection
		self.impatience = impatience
		# 2 continuous action box: [steering, throttle]
		self.action_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(2,),
			dtype=np.float32
		)

		# for 1-agent system
		# Ego Physics (4old + 4new): [Speed, Acceleration, Normalized_Lane_Index, 
		# Fuel/Electricity Consumption (both in one as this case is using HEV)]
		# Added: (just for the AI to learn "intent", not real physics, as SUMO handles these very tidy, 
		# so tidy that you don't need to explicitly define the exact value)
			# Lateral Lane Position: Distance from the center of the lane (in meters). -1.5 means left of center, 1.5 means right.
			# Heading Error: Difference between Car Angle and Lane Angle (in degrees or radians).
			# Current Steering Angle: The current angle of the front wheels. The agent needs to know "where are my wheels pointing right now?" to avoid oscillating.
			# Road Slope: The incline of the road

		# Front Radar (2): [Leader_Distance, Leader_Rel_Speed]
		# Side Awareness (8):
			# Front-Left: [Dist, Rel_Speed]
			# Back-Left: [Dist, Rel_Speed]
			# Front-Right: [Dist, Rel_Speed]
			# Back-Right: [Dist, Rel_Speed]
		# Infrastructure (6): [Speed_Limit, Can_Go_Left(0/1), Can_Go_Right(0/1), 
		# TLS_Dist, TLS_State(Red/Green), Dist_To_Turn]
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(24,),
			dtype=np.float32
			)
		
		self.last_known_dist = 0.0
	
	def _get_passenger_edges(self) -> list:
		all_edges = traci.edge.getIDList()
		valid_edges = []
		for edge_id in all_edges:
			if edge_id.startswith(":") or edge_id.startswith("!"):
				continue
			lane_id = f"{edge_id}_0" 
			try:
				allowed = traci.lane.getAllowed(lane_id)
				if not allowed or "passenger" in allowed:
					valid_edges.append(edge_id)
			except:
				continue
		random.shuffle(valid_edges)
		return valid_edges
	
	def _get_surroundings(self):
		# Constants for normalization

		# Default values: [Dist=1.0 (Far), RelSpeed=0.0 (same speed)]
		# Order: [Left-front, Left-back, Right-front, Right-back]
		data = {
			"L_F_Dist": 1.0, "L_F_RelSpeed": 0.0,
			"L_B_Dist": 1.0, "L_B_RelSpeed": 0.0,
			"R_F_Dist": 1.0, "R_F_RelSpeed": 0.0,
			"R_B_Dist": 1.0, "R_B_RelSpeed": 0.0,
		}
		
		my_speed = traci.vehicle.getSpeed(self.VEH_ID)

		# check left lane
		# The '2' is a binary bitset (0b010) meaning "Left Only"
		# for traci.vehicl.getNeighbors: if the return distance is > 0, the car is at front (R_F, L_F)
		# else, the car is at back (R_B, L_B)
		left_cars = traci.vehicle.getNeighbors(self.VEH_ID, 2)

		closest_front = float("inf")
		closest_back = float("inf")

		for n_id, dist in left_cars:
			if dist > 0: # dist > 0 => check front
				if dist < closest_front:
					closest_front = dist
					n_speed = traci.vehicle.getSpeed(n_id)
					data["L_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST # normalization and double check
					data["L_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
			else: # Back (dist <= 0)
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["L_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST # normalization and double check
					data["L_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore

		
		# check right lane
		# The '1' is a binary bitset (0b01) meaning "Right Only"
		right_cars = traci.vehicle.getNeighbors(self.VEH_ID, 1)

		closest_front = float("inf")
		closest_back = float("inf")

		for n_id, dist in right_cars:
			if dist > 0: # front
				if dist < closest_front:
					closest_front = dist
					n_speed = traci.vehicle.getSpeed(n_id)
					data["R_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
					data["R_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
			else: # back
				if abs(dist) < closest_back:
					closest_back = abs(dist)
					n_speed = traci.vehicle.getSpeed(n_id)
					data["R_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
					data["R_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED # type: ignore
		
		output = [stats for key, stats in data.items()]
		return output
	


	def _get_obs(self):
		# safety check: if vehicle is dead -> return zeros
		if self.VEH_ID not in traci.vehicle.getIDList():
			return np.zeros(24, dtype=np.float32)
		
		

		# EGO PHYSICS (4) --------
		velocity = traci.vehicle.getSpeed(self.VEH_ID) # *
		acceleration = traci.vehicle.getAcceleration(self.VEH_ID) # *
		# becareful here, getAcceleration returns the current accelerate (dynamic)
		# getAccel returns the max acceleration of a vehicle type (static)

		try:
			# lane index
			lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			total_lanes = traci.edge.getLaneNumber(road_id)
			norm_lane = lane_idx / max(1, total_lanes - 1) # type: ignore # *

			slope = traci.vehicle.getSlope(self.VEH_ID) / self.MAX_SLOPE # type: ignore # *
			lat_offset = traci.vehicle.getLateralLanePosition(self.VEH_ID) # *

			car_angle = traci.vehicle.getAngle(self.VEH_ID)
			edge_angle = traci.vehicle.getAngle(self.VEH_ID) # getting exact lane angle is kinda complex in traci
			heading_error = ((car_angle - edge_angle) % 360) / 360.0  # type: ignore # *

			steer_angle = traci.vehicle.getAngle(self.VEH_ID) / 360 # type: ignore # *

		except: 
			norm_lane = 0.0
			slope = 0

		# HEV energy: Just take whatever SUMO gives us. 
		# Fuel: Max ~12000/s. Elec: Max ~120 Wh/s		
		# If you want the agent to be smart enough to know "I am currently charging," 
		# you can split the Energy input into two fields. Fuel Burn (0.0 to 1.0), Battery Flow (-1.0 to 1.0) (pos.: draining, neg.: charging)
		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID) / self.MAX_ELEC # type: ignore
	


		# LEADER (2) --------
		try:
			leader = traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST) # leader_id, leader_dist
			if leader:
				l_dist = leader[1] / self.MAX_DIST # *
				l_speed = traci.vehicle.getSpeed(leader[0])
				l_rel_speed = (traci.vehicle.getSpeed(self.VEH_ID) - l_speed) / self.MAX_SPEED  # type: ignore # *
			else:
				l_dist = 1.0 # far away
				l_rel_speed = 0.0
		except:
			l_dist, l_rel_speed = 0.0, 0.0

		# surroundings (8)
		surroundings = self._get_surroundings()

		# infrastructure (6)
		try:
			# the Lane ID (id) is a unique string identifier for a specific lane (e.g., "edge1_0"), 
			# while the Lane Index (index) is its numerical position (starting from 0 for the rightmost lane) within an edge
			lane_id = traci.vehicle.getLaneID(self.VEH_ID)

			if lane_id == "":
				speed_limit = 1.0
				can_left, can_right = 0.0, 0.0
				tls_dist, tls_state, turn_dist = 1.0, 1.0, 1.0
			else:
				# Normal logic
				speed_limit = traci.lane.getMaxSpeed(laneID=lane_id) / self.MAX_SPEED
			
			# lane feasibility
			road_id = traci.vehicle.getRoadID(self.VEH_ID)
			num_lanes = traci.edge.getLaneNumber(road_id)

			can_left = 1.0 if lane_idx < (num_lanes - 1) else 0.0 # type: ignore # *
			can_right = 1.0 if lane_idx > 0 else 0.0 # type: ignore # *

			# traffic light state (tls)
			tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
			if tls_data:
				# tls_data[0] is (tlsID, tlsIndex, dist, state)
				tls_dist = tls_data[0][2] / self.MAX_DIST # *
				
				# Simple logic: 'g'/'G' is Green (1.0), else Red (0.0)
				tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0 # *
			else:
				tls_dist = 1.0
				tls_state = 1.0 # assume green

			# distance to turn
			# distance to end of the lane as a proxy
			lane_len = traci.lane.getLength(lane_id)
			lane_pos = traci.vehicle.getLanePosition(self.VEH_ID)
			turn_dist = min(lane_len - lane_pos, self.MAX_DIST) / self.MAX_DIST # *
		
		except:
			speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0

		# combination
		obs = [velocity, acceleration, elec, norm_lane, slope, lat_offset, heading_error, steer_angle] + \
		surroundings + [l_dist, l_rel_speed] + \
		[speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist] 

		obs = np.array(obs, dtype=np.float32)
		obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

		return obs
	
	def _get_dist_to_destination(self):
		try:
			currrent_edge = traci.vehicle.getRoadID(self.VEH_ID)
			if currrent_edge.startswith(":"): # type: ignore
				if self.last_known_dist:
					return self.last_known_dist
				else:
					return 0

			if hasattr(self, "current_route_edges") and currrent_edge in self.current_route_edges:
				idx = self.current_route_edges.index(current_edge)
				remaining_edges = self.current_route_edges[idx:]
			else:
				return 0
			
			dist = 0.0
			for e in remaining_edges:
				dist += traci.lane.getLength(f"{e}_0")  # type: ignore

			# subtract distance already driven on current edge
			dist -= traci.vehicle.getLanePosition(self.VEH_ID) # type: ignore
			self.last_known_dist = dist
			return dist
		except:
			return 0.0


	# reward function
	def _calculate_reward(self, action):
		# Weights (Tunable)
		W_PROGRESS = 0.1 # (1 meter = 0.1 points)
		W_ENERGY = -0.5  # Negative because energy is bad
		W_COMFORT = -0.5 # Penalty for Jerk/Wiggle


		dist = self._get_dist_to_destination()
		if not hasattr(self, "prev_dist"):
			self.prev_dist = dist

		progress = self.prev_dist - dist
		self.prev_dist = dist

		# energy pen
		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID)
		energy_penalty = elec / 120 # type: ignore

		if np.isnan(energy_penalty) or energy_penalty is None: energy_penalty = 0.0

		if not hasattr(self, "prev_action"):
			self.prev_action = action
		
		# calculate action change in steering/gas
		action_delta = np.abs(action - self.prev_action)
		wiggle_penalty = np.mean(action_delta) # avg change
		self.prev_action = action

		reward = (progress * W_PROGRESS) + (energy_penalty * W_ENERGY) + (wiggle_penalty * W_COMFORT)

		return reward


	def reset(self, seed=None, options=None):

		super().reset(seed=seed,options=options)
		
		# try to close the environment
		try:
			if traci.isLoaded():
				traci.close()
		except Exception:
			pass
		time.sleep(1.0) 
		
		if self.test_mode:
			active_map = self.maps[0]
			route_arg = ["-a", self.test_route]
		else:
			active_map = random.choice(self.maps)
			route_arg = []
		self.step_count = 0
		# set up the launch
		SumoBinary: str = "sumo-gui" if self.render_mode else "sumo"
		SumoCMD: list[str] = [SumoBinary, "-c", active_map] + route_arg + \
				["--start", "--quit-on-end",
				"--device.emissions.probability", "1.0",
				"--scale", str(self.TRAFFIC_SCALE),
				"--delay", str(self.delay),
				"--no-step-log", "true",
				"--no-warnings", "true"]
		

		# launch sumo
		traci.start(SumoCMD)

		# advance a little few steps so that the traffic is ready 
		# (other vehicles are spawned, and not only be in on the edges of the map)
		# Sumo's flow = Static, timestep = random => random traffic
		for _ in range(random.randint(150, 300)): traci.simulationStep()

		# 1. SETUP VEHICLE TYPE
		try:
			existing_types = traci.vehicletype.getIDList()
			source_type = "DEFAULT_VEHTYPE"
			if source_type not in existing_types and len(existing_types) > 0:
				source_type = existing_types[0]

			# Copy base type
			traci.vehicletype.copy(source_type, self.VTYPE_ID)
			traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
			traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0)) 
			traci.vehicletype.setMass(self.VTYPE_ID, 2911)
			traci.vehicletype.setLength(self.VTYPE_ID, 5.1181)
			traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")
			

			# --- EV PARAMETERS START (used to be HEV) ---
			# Enable Emission (Fuel) and Battery (Elec) devices
			#traci.vehicletype.setParameter(self.VTYPE_ID, "has.emissions.device", "true")
			traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device", "true")
			#traci.vehicletype.setParameter(self.VTYPE_ID, "has.elecHybrid.device", "true")
			
			# Set Battery Capacity (e.g., 13600 Wh buffer)
			# According to Vinfast VF9
			# (https://vinfastauto.us/vehicles/vf-9#:~:text=human%2Dlike%20conversations%E2%80%A6-,Power%20meets%20style,-VF%209%20comes)
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity", "123000.00")
			traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel", "123000.00") # Start at full charge
			
			# --- EV PARAMETERS END ---

			# Apply Imperfection to Background Traffic
			for v_type in existing_types:
				traci.vehicletype.setImperfection(v_type, self.imperfection)
				traci.vehicletype.setImpatience(v_type, self.impatience)

		except Exception as e:
			print(f"Error defining vType: {e}")

		# we load a random map every episode so we load drivable roads every episodes
		self.drivable_edges = self._get_passenger_edges()


		# spawn the agent
		spawned = False
		ego_veh_tracked = False # Tracking vehicle: on?
		max_attempts = 50
		if self.test_mode:
			while self.VEH_ID not in traci.vehicle.getIDList():
				traci.simulationStep()
			
			traci.vehicle.setType(self.VEH_ID, self.VTYPE_ID)
			traci.vehicle.setSpeedMode(self.VEH_ID, 0)
			traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

			for _ in range(20):
				traci.simulationStep()
				if self.VEH_ID in traci.vehicle.getIDList():
					break
				# --------------------------
				
				# Check if it actually spawned
				if self.VEH_ID in traci.vehicle.getIDList():
					spawned = True

			if self.VEH_ID in traci.vehicle.getIDList(): # double guard, meaningless
				spawned = True

			if self.render_mode and not ego_veh_tracked:
				traci.gui.trackVehicle("View #0", self.VEH_ID)
				traci.gui.setZoom("View #0", 2000)
		else:
			# --- ROBUST "BEST EFFORT" ROUTE GENERATION ---
			spawned = False
			attempts = 0
			
			# Keep track of the best route we find
			best_route_edges = None
			max_route_length = 0.0

			# 1. Filter edges if not done already (exclude internal edges starting with :)
			if not hasattr(self, 'drivable_edges') or not self.drivable_edges:
				all_edges = traci.edge.getIDList()
				self.drivable_edges = [e for e in all_edges if not e.startswith(":")]

			# Try 20 times to find a route
			while attempts < 20:
				attempts += 1
				try:
					edge_start = random.choice(self.drivable_edges)
					edge_end = random.choice(self.drivable_edges)
					
					if edge_start == edge_end:
						continue

					# Ask SUMO for path
					route = traci.simulation.findRoute(edge_start, edge_end, self.VTYPE_ID)
					
					if not route.edges:
						continue

					# Calculate Length
					current_length = 0.0
					for edge_id in route.edges:
						try:
							# Assume lane 0 exists
							current_length += traci.lane.getLength(f"{edge_id}_0")
						except:
							pass
					
					# LOGIC CHANGE:
					# If it's a "Good Enough" route (>1000m), take it immediately.
					if current_length > 1000.0:
						best_route_edges = route.edges
						max_route_length = current_length
						break # Stop searching, we found a good one
					
					# Otherwise, remember it if it's the best so far
					if current_length > max_route_length:
						best_route_edges = route.edges
						max_route_length = current_length

				except Exception:
					continue
			
			# --- SPAWN USING THE BEST ROUTE FOUND ---
			if best_route_edges and len(best_route_edges) > 0:
				try:
					route_id = f"route_{random.randint(0, 1000000)}"
					traci.route.add(route_id, best_route_edges)
					
					# Store for reward calculation
					self.current_route_edges = best_route_edges 
					
					# Use "free" to prevent stacking
					traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)
					
					# Disable SUMO's internal logic so AI controls it
					traci.vehicle.setSpeedMode(self.VEH_ID, 0)
					traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

					for _ in range(20):
						traci.simulationStep()
						if self.VEH_ID in traci.vehicle.getIDList():
							break
					# --------------------------
					
					# Check if it actually spawned
					if self.VEH_ID in traci.vehicle.getIDList():
						spawned = True

					if self.render_mode and not ego_veh_tracked:
						traci.gui.trackVehicle("View #0", self.VEH_ID)
						traci.gui.setZoom("View #0", 2000)

					traci.simulationStep()
					spawned = True
					# Optional Debug:
					# print(f"Spawned! Length: {max_route_length:.1f}m | Attempts: {attempts}")

				except Exception as e:
					print(f"Spawn Error: {e}")
			
			# If we STILL failed (e.g. map is empty or broken), reload.
			if not spawned:
				print(f"Critial Map Failure. Max len found: {max_route_length}. Reloading...")
				return self.reset(seed=seed, options=options)
			
		if self.render_mode and spawned:
			# Only track if the vehicle ACTUALLY exists
			if self.VEH_ID in traci.vehicle.getIDList():
				traci.gui.trackVehicle("View #0", self.VEH_ID)
				traci.gui.setZoom("View #0", 2000) # Zoom in to the car

		self.stuck_time = 0
		obs = self._get_obs()
		info = {}
		return obs, info		

	
	def step(self, action):
		self.step_count += 1

		# 1. Physics Setup
		steer_cmd = action[0]
		accel_cmd = action[1]
		
		# Longitudinal control
		if accel_cmd >= 0:
			desired_accel = accel_cmd * self.MAX_ACCEL
		else:
			desired_accel = accel_cmd * self.MAX_DECEL

		SIM_STEPS = 10
		delta_time = SIM_STEPS * traci.simulation.getDeltaT()
		
		# Apply smooth speed change
		current_speed = traci.vehicle.getSpeed(self.VEH_ID)
		target_speed = current_speed + (desired_accel * delta_time)
		target_speed = max(0.0, min(target_speed, self.MAX_SPEED))
		
		traci.vehicle.slowDown(vehID=self.VEH_ID, speed=target_speed, duration=delta_time)

		# Lateral control
		LC_THRESHOLD = 0.3
		target_lane_offset = 0 
		if steer_cmd < -LC_THRESHOLD: target_lane_offset = -1 
		elif steer_cmd > LC_THRESHOLD: target_lane_offset = 1 

		if target_lane_offset != 0:
			try:
				current_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
				edge_id = traci.vehicle.getRoadID(self.VEH_ID)
				num_lanes = traci.edge.getLaneNumber(edge_id)
				desired_lane = current_lane + target_lane_offset 
				if 0 <= desired_lane < num_lanes: 
					traci.vehicle.changeLane(self.VEH_ID, desired_lane, 2.0)
			except:
				pass

		# 2. Simulation Loop
		reward = 0.0
		terminated = False
		truncated = False
		
		# Initialize variables BEFORE the loop
		accumulated_energy = 0.0 
		final_real_speed = 0.0

		for _ in range(SIM_STEPS):
			traci.simulationStep()

			# A. Check Existence
			if self.VEH_ID not in traci.vehicle.getIDList():
				arrived_list = traci.simulation.getArrivedIDList()
				if self.VEH_ID in arrived_list:
					terminated = True
					reward += 90.0 # Arrival Bonus
				else:
					terminated = True
					reward -= 90.0 # Teleport/Crash Penalty
				break # Exit loop immediately

			# B. Check Collisions
			collision_list = traci.simulation.getCollisions()
			col_ids = [c.collider for c in collision_list] + [c.victim for c in collision_list]
			if self.VEH_ID in col_ids:
				terminated = True
				reward -= 90.0 
				# print("Vehicle Crashed!!!")
				break # Exit loop immediately
			
			# C. Data Collection (Only happens if car exists)
			final_real_speed = traci.vehicle.getSpeed(self.VEH_ID)
			
			e_consumption = traci.vehicle.getElectricityConsumption(self.VEH_ID)
			if e_consumption is None or np.isnan(e_consumption): e_consumption = 0.0
			accumulated_energy += e_consumption
			
			# D. Step Reward
			reward += self._calculate_reward(action)

		# --- AFTER LOOP (Do not use 'else' block here!) ---

		# 3. Finalize
		if self.step_count >= self.MAX_EPISODE_STEPS:
			truncated = True
		
		# Update Obs
		obs = self._get_obs()
			
		# Stuck Detection Logic
		# (This uses the final_real_speed captured INSIDE the loop)
		if final_real_speed < 0.5:
			self.stuck_time += 1
		else:
			self.stuck_time = 0 
		
		if self.stuck_time >= 50:
			# print(f"Vehicle Stuck! (Speed 0 for 5s). Resetting.")
			terminated = True
			reward -= 50.0 
			info_success = False
		else:
			info_success = 1 if (terminated and reward > 0) else 0

		info = {
			"real_speed": final_real_speed,
			"real_energy": accumulated_energy, 
			"is_success": info_success
		}

		return obs, reward, terminated, truncated, info


	def close(self):
		try:
			traci.close()
		except:
			pass



if __name__ == "__main__":
	# 1. Init
	# To Train (Random):
	env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=False, test_route="TestMap/test_route.rou.xml", delay=100)
	
	# To Test (Fixed XML):
	# env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=True, test_xml="my_route.rou.xml")
	
	# 2. Reset
	obs, info = env.reset()
	print(f"Init Obs Shape: {obs.shape}")
	
	done = False
	total_reward = 0
	
	# 3. Loop (Test for 50 steps)
	print("Starting Loop...")
	for i in range(50):
		# Random action
		action = env.action_space.sample()
		action[1] = np.random.uniform(-1.0, 1.0)
		
		obs, reward, terminated, truncated, info = env.step(action)
		total_reward += reward
		
		print(f"Step {env.step_count} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
		
		if terminated or truncated:
			print("Episode Finished!")
			obs, info = env.reset()
			
	env.close()
	print("Test Complete.")