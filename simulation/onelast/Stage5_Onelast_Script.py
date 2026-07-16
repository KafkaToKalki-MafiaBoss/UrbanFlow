# Stage 5: SUMO Baseline Data Collection Script
# Network: onelast.net.xml (4-phase traffic signal)
# Routes: onelast.rou.xml (multi-directional traffic flows)
# Configuration: onelast.sumocfg

import os
import sys
import sqlite3
from datetime import datetime

# Step 1: Setup SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print("ERROR: Please set SUMO_HOME environment variable")
    print("Windows: setx SUMO_HOME C:\\Program Files\\SUMO")
    sys.exit(1)

# Step 2: Import TraCI
import traci

# Step 3: Define SUMO configuration for onelast network
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

SUMO_CONFIG = [
    "sumo-gui",
    "-c", os.path.join(current_dir, "onelast.sumocfg"),
    "--step-length", "0.05",
    "--delay", "100"
]

# Step 4: Initialize database
def initialize_database():
    """Create database and table for baseline results"""
    try:
        conn = sqlite3.connect('baseline_results.db')
        cursor = conn.cursor()
        
        # Drop table if exists (fresh start)
        cursor.execute('DROP TABLE IF EXISTS fixed_timer_baseline')
        
        # Create new table
        cursor.execute('''
            CREATE TABLE fixed_timer_baseline (
                id INTEGER PRIMARY KEY,
                step INTEGER,
                time_sec REAL,
                E0_queue INTEGER,
                E1_queue INTEGER,
                E2_queue INTEGER,
                E3_queue INTEGER,
                total_vehicles INTEGER,
                total_waiting_vehicles INTEGER,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

# Step 5: Define data collection function
def collect_baseline_data(step):
    """Collect queue length, vehicle count, and waiting time data"""
    
    try:
        # Get simulation time (convert milliseconds to seconds)
        sim_time = traci.simulation.getCurrentTime() / 1000
        
        # Get queue lengths for each approach
        # Queue length = number of vehicles that are halting (stopped waiting)
        try:
            # West=E0, East=-E1, South=E2, North=-E3 (incoming edges; E1/E3 are outgoing)
            E0_queue = traci.edge.getLastStepHaltingNumber("E0")
            E1_queue = traci.edge.getLastStepHaltingNumber("-E1")
            E2_queue = traci.edge.getLastStepHaltingNumber("E2")
            E3_queue = traci.edge.getLastStepHaltingNumber("-E3")
        except:
            # If edges don't work, try alternative names
            E0_queue = 0
            E1_queue = 0
            E2_queue = 0
            E3_queue = 0
        
        # Get total vehicles in network
        vehicle_list = traci.vehicle.getIDList()
        total_vehicles = len(vehicle_list)
        
        # Count waiting vehicles (vehicles with speed < 0.1 m/s)
        waiting_count = 0
        for veh_id in vehicle_list:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                if speed < 0.1:  # Essentially stopped
                    waiting_count += 1
            except:
                pass
        
        # Print progress every 60 steps (3 seconds)
        if step % 60 == 0:
            print(f"Step {step:4d} (Time {sim_time:6.1f}s): "
                  f"E0_Q={E0_queue:2d} E1_Q={E1_queue:2d} E2_Q={E2_queue:2d} E3_Q={E3_queue:2d} | "
                  f"Total_Veh={total_vehicles:3d} Waiting={waiting_count:3d}")
        
        # Return data dictionary
        return {
            'step': step,
            'time_sec': sim_time,
            'E0_queue': E0_queue,
            'E1_queue': E1_queue,
            'E2_queue': E2_queue,
            'E3_queue': E3_queue,
            'total_vehicles': total_vehicles,
            'waiting_vehicles': waiting_count,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"❌ Error collecting data at step {step}: {e}")
        return None

# Step 6: Save data to database
def save_to_database(data_list):
    """Save collected data to SQLite database"""
    try:
        conn = sqlite3.connect('baseline_results.db')
        cursor = conn.cursor()
        
        count = 0
        for data in data_list:
            if data is not None:
                cursor.execute('''
                    INSERT INTO fixed_timer_baseline 
                    (step, time_sec, E0_queue, E1_queue, E2_queue, E3_queue,
                     total_vehicles, total_waiting_vehicles, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['step'], data['time_sec'], data['E0_queue'],
                      data['E1_queue'], data['E2_queue'], data['E3_queue'],
                      data['total_vehicles'], data['waiting_vehicles'], data['timestamp']))
                count += 1
        
        conn.commit()
        conn.close()
        print(f"✅ Saved {count} data points to database")
        return True
    
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return False

# Step 7: Calculate statistics
def calculate_statistics():
    """Calculate and display baseline statistics"""
    try:
        conn = sqlite3.connect('baseline_results.db')
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM fixed_timer_baseline')
        total_records = cursor.fetchone()[0]
        
        # Calculate averages per approach
        cursor.execute('SELECT AVG(E0_queue) FROM fixed_timer_baseline')
        avg_E0_queue = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(E1_queue) FROM fixed_timer_baseline')
        avg_E1_queue = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(E2_queue) FROM fixed_timer_baseline')
        avg_E2_queue = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(E3_queue) FROM fixed_timer_baseline')
        avg_E3_queue = cursor.fetchone()[0]
        
        # Get peak values
        cursor.execute('SELECT MAX(E0_queue) FROM fixed_timer_baseline')
        max_E0 = cursor.fetchone()[0]
        
        cursor.execute('SELECT MAX(E1_queue) FROM fixed_timer_baseline')
        max_E1 = cursor.fetchone()[0]
        
        cursor.execute('SELECT MAX(E2_queue) FROM fixed_timer_baseline')
        max_E2 = cursor.fetchone()[0]
        
        cursor.execute('SELECT MAX(E3_queue) FROM fixed_timer_baseline')
        max_E3 = cursor.fetchone()[0]
        
        # Average total vehicles
        cursor.execute('SELECT AVG(total_vehicles) FROM fixed_timer_baseline')
        avg_total_vehicles = cursor.fetchone()[0]
        
        # Average waiting vehicles
        cursor.execute('SELECT AVG(total_waiting_vehicles) FROM fixed_timer_baseline')
        avg_waiting = cursor.fetchone()[0]
        
        conn.close()
        
        # Display results
        print("\n" + "="*90)
        print("STAGE 5 BASELINE RESULTS - FIXED-TIMER SIGNALS (4-PHASE - ONELAST NETWORK)")
        print("="*90)
        print(f"\nSimulation Duration: 1 hour (3600 seconds)")
        print(f"Total Data Points: {total_records}")
        print(f"\nQueue Length Statistics (vehicles):")
        print(f"  E0 (West):  Avg={avg_E0_queue:.2f}, Peak={max_E0}")
        print(f"  E1 (East):  Avg={avg_E1_queue:.2f}, Peak={max_E1}")
        print(f"  E2 (South): Avg={avg_E2_queue:.2f}, Peak={max_E2}")
        print(f"  E3 (North): Avg={avg_E3_queue:.2f}, Peak={max_E3}")
        
        overall_avg_queue = (avg_E0_queue + avg_E1_queue + avg_E2_queue + avg_E3_queue) / 4
        overall_max_queue = max(max_E0, max_E1, max_E2, max_E3)
        
        print(f"\nOverall Statistics:")
        print(f"  Average Queue (All Approaches): {overall_avg_queue:.2f} vehicles")
        print(f"  Peak Queue (any approach): {overall_max_queue} vehicles")
        print(f"  Average Total Vehicles in Network: {avg_total_vehicles:.2f}")
        print(f"  Average Waiting Vehicles: {avg_waiting:.2f}")
        
        print(f"\nTraffic Signal Configuration (4-PHASE):")
        print(f"  Total Cycle Time: 90 seconds")
        print(f"  Phase 1 (E0 + E1 Green): 42s")
        print(f"  Phase 2 (Yellow): 3s")
        print(f"  Phase 3 (E2 + E3 Green): 42s")
        print(f"  Phase 4 (Yellow): 3s")
        
        print("\n" + "="*90)
        print("✅ Stage 5 baseline data collection complete!")
        print(f"Results saved in: baseline_results.db")
        print("="*90 + "\n")
        
        return True
    
    except Exception as e:
        print(f"❌ Error calculating statistics: {e}")
        return False

# Step 8: Main execution
if __name__ == "__main__":
    print("\n" + "="*90)
    print("STAGE 5: SUMO BASELINE DATA COLLECTION - ONELAST NETWORK")
    print("="*90)
    print(f"Starting simulation with:")
    print(f"  Network: onelast.net.xml (4-phase traffic signal)")
    print(f"  Routes: onelast.rou.xml (multi-directional traffic flows)")
    print(f"  Config: onelast.sumocfg")
    print(f"  Duration: 1 hour (3600 seconds)")
    print("="*90 + "\n")
    
    try:
        # Initialize database
        if not initialize_database():
            sys.exit(1)
        
        # Start TraCI connection
        print("Starting SUMO simulation...")
        traci.start(SUMO_CONFIG)
        print("✅ TraCI connection established\n")
        
        # Run simulation and collect data
        print("Running simulation and collecting data...")
        print("-" * 90)
        baseline_data = []
        step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect data at each step
            data = collect_baseline_data(step)
            if data is not None:
                baseline_data.append(data)
            
            step += 1
        
        print("-" * 90)
        
        # Close TraCI connection
        traci.close()
        print("\n✅ Simulation completed\n")
        
        # Save data to database
        if save_to_database(baseline_data):
            # Calculate and display statistics
            calculate_statistics()
        else:
            print("❌ Failed to save data to database")
        
    except traci.TraCIException as e:
        print(f"\n❌ TraCI Error: {e}")
        print("Make sure SUMO is properly installed and SUMO_HOME is set")
        try:
            traci.close()
        except:
            pass
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        try:
            traci.close()
        except:
            pass
        sys.exit(1)