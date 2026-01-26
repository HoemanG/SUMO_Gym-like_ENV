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
		
		self.step_count = 0
	
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

			speed_limit = traci.lane.getMaxSpeed(laneID=lane_id) / self.MAX_SPEED # type: ignore # *

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
	

	# reward function
	def _calculate_rewards(self, action) -> float:
		# goal: fast, no wiggle driving, safety, energy efficiency
		# step_reward = speed - 0.5*steer - elec_consumption

		current_speed = traci.vehicle.getSpeed(self.VEH_ID)
	
		# Get the limit of the current road
		try:
			lane_id = traci.vehicle.getLaneID(self.VEH_ID)
			limit = traci.lane.getMaxSpeed(lane_id)
		except:
			limit = 50.0

		# REWARD LOGIC:
		# Reward speed up to the limit
		if current_speed <= limit: # type: ignore
			rew_speed = current_speed / limit # type: ignore
		else:
			# Penalize speeding! 
			# If speed > limit, reward drops drastically
			rew_speed = 1.0 - ((current_speed - limit) / limit) # type: ignore

		elec = traci.vehicle.getElectricityConsumption(self.VEH_ID)
		if elec is None or np.isnan(elec):
			elec = 0.0
		rew_elec = max(elec / self.MAX_ELEC, 0) # type: ignore

		steer_cmd = action[0]
		rew_steer = abs(steer_cmd)

		return rew_speed - (0.5 * rew_steer) - rew_elec



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

		# advance a little few steps so that the traffic is ready (other vehicles are spawned)
		for _ in range(5): traci.simulationStep()

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

		if self.test_mode:
			while self.VEH_ID not in traci.vehicle.getIDList():
				traci.simulationStep()
			
			traci.vehicle.setType(self.VEH_ID, self.VTYPE_ID)
			traci.vehicle.setSpeedMode(self.VEH_ID, 0)
			traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

			if self.VEH_ID in traci.vehicle.getIDList(): # double guard, meaningless
				spawned = True

			if self.render_mode and not ego_veh_tracked:
				traci.gui.trackVehicle("View #0", self.VEH_ID)
				traci.gui.setZoom("View #0", 2000)
		else:
			while not spawned: # try to spawn the agent until success
				try:
					# advance 1 step to make sure the env is ready
					traci.simulationStep()

					# pick a random route
					edge_start = random.choice(self.drivable_edges)
					edge_end = random.choice(self.drivable_edges)

					if edge_start != edge_end:
						# ask SUMO for a route
						route = traci.simulation.findRoute(edge_start, edge_end, self.VTYPE_ID)
	
						# Calculate approximate length in meters
						# (We assume lane 0 length approximates edge length)
						total_length = 0
						for edge in route.edges: # type: ignore
							try:
								total_length += traci.lane.getLength(f"{edge}_0") # type: ignore
							except:
								pass

						if route.edges and len(route.edges) > 15 and total_length > 1000: # type: ignore
							route_id = f"route_{random.randint(0, 1000000)}"
							traci.route.add(route_id, route.edges) # type: ignore

							# --- FIX STARTS HERE ---
							# Clean up previous failed attempts
							# If the car is stuck in "pending" state, remove it before adding again.
							try:
								traci.vehicle.remove(self.VEH_ID)
							except:
								pass # If vehicle doesn't exist, just ignore
							# add the car
							traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)

							# disable safety guards and make the world imperfect
							traci.vehicle.setSpeedMode(self.VEH_ID, 0)
							traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)

							# move simulation until we can spawn our ego car
							traci.simulationStep()
							if self.VEH_ID in traci.vehicle.getIDList():
								print(f"Vehicle {self.VEH_ID} has successfully entered the road network!")
								if not ego_veh_tracked and self.render_mode:
									ego_veh_tracked = True
									traci.gui.trackVehicle("View #0", self.VEH_ID)
									traci.gui.setZoom("View #0", 600)
								spawned = True
				except:
					continue # retry if sumo raises error

		obs = self._get_obs()
		info = {}
		return obs, info		

	
	def step(self, action):

		# reset counter
		self.step_count += 1

		# - apply action -
		current_speed = traci.vehicle.getSpeed(self.VEH_ID)
		SIM_STEPS = 10

		# Longitudinal control
		accel_cmd = action[1]
		if accel_cmd >= 0:
			desired_accel = accel_cmd * self.MAX_ACCEL
		else:
			desired_accel = accel_cmd * self.MAX_DECEL

		delta_time = 1.0 * SIM_STEPS
		target_speed = current_speed + desired_accel * delta_time
		target_speed = max(0.0, min(target_speed, self.MAX_SPEED))
		
		# Send Speed
		traci.vehicle.setSpeed(vehID=self.VEH_ID, speed=target_speed)

		# lateral control
		steer_cmd = action[0]
		LC_THRESHOLD = 0.3
		
		target_lane = 0 
		if steer_cmd < -LC_THRESHOLD:
			target_lane = -1 # right
		elif steer_cmd > LC_THRESHOLD:
			target_lane = 1 # left

		if target_lane != 0:
			current_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
			try:
				edge_id = traci.vehicle.getRoadID(self.VEH_ID)
				num_lanes = traci.edge.getLaneNumber(edge_id)
				desired_lane = current_lane + target_lane 
				if 0 <= desired_lane and desired_lane < num_lanes: 
					traci.vehicle.changeLane(self.VEH_ID, desired_lane, 2.0)
			except:
				pass

		# Run simulation
		reward = 0.0
		terminated = False
		truncated = False
		
		# Create accumulators for the 10-step duration
		accumulated_energy = 0.0 
		final_real_speed = 0.0

				# Timeout logic
		if self.step_count >= self.MAX_EPISODE_STEPS:
			truncated = True
		else:
			truncated = False
			
		if terminated:
			truncated = False 


		# If SUMO returned the garbage value (negative huge number), reset to 0
		if final_real_speed < -100: 
			final_real_speed = 0.0
		
		# update states
		obs = self._get_obs()
		
		info = {
			"real_speed": final_real_speed,
			"real_energy": accumulated_energy, 
			"is_success": 1 if (terminated and reward > 0) else 0 
		}


		for _ in range(SIM_STEPS):
			traci.simulationStep()

			# 1. CRITICAL: Check existence BEFORE asking for physics data
			if self.VEH_ID not in traci.vehicle.getIDList():
				# The car is gone. Why?
				
				# Check if it actually Arrived at the destination
				arrived_list = traci.simulation.getArrivedIDList()
				if self.VEH_ID in arrived_list:
					terminated = True
					reward += 90.0 # Real Success
					info['is_success'] = True
				else:
					# It didn't arrive, but it's gone. 
					# Could be a collision that removed it, or a teleport.
					terminated = True
					reward -= 90.0 # Penalty for disappearing without arriving
					info['is_success'] = False
				
				# BREAK THE LOOP IMMEDIATELY so we don't ask for speed later
				break 

			# 2. Check Collisions (if it's still in the list but hit something)
			collision_list = traci.simulation.getCollisions()
			# getCollisions returns object IDs, we need to check if our car is involved
			col_ids = [c.collider for c in collision_list] + [c.victim for c in collision_list]
			if self.VEH_ID in col_ids:
				terminated = True
				reward -= 90.0 # Crash Penalty
				info['is_success'] = False
				break
			
			# get speed 
			info['real_speed'] = traci.vehicle.getSpeed(self.VEH_ID)
			e_consumption = traci.vehicle.getElectricityConsumption(self.VEH_ID)
			if e_consumption is None or np.isnan(e_consumption):
				e_consumption = 0.0
			info['real_energy'] += e_consumption
			# 3. Calculate Step Reward (Only if car exists)
			reward += self._calculate_rewards(action)



		return obs, reward, terminated, truncated, info


	def close(self):
		try:
			traci.close()
		except:
			pass



if __name__ == "__main__":
	# 1. Init
	# To Train (Random):
	env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=True, test_route="TestMap/test_route.rou.xml", delay=100)
	
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
		
		obs, reward, terminated, truncated, info = env.step(action)
		total_reward += reward
		
		print(f"Step {i} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
		
		if terminated or truncated:
			print("Episode Finished!")
			obs, info = env.reset()
			
	env.close()
	print("Test Complete.")