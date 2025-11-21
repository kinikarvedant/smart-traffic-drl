# test_sumo.py
import os
import traci

cfg = os.path.join("sumo_scenario", "simple.sumocfg")
traci.start(["sumo", "-c", cfg])   # use "sumo-gui" if you want the GUI
print("SUMO Connected!")
traci.close()
