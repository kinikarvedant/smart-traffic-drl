# baselines/fixed_time_eval.py
import os, csv
import traci

CFG = os.path.join("sumo_scenario", "simple.sumocfg")
OUT_CSV = os.path.join("results", "baseline_metrics.csv")

def run_fixed(green_time=10):
    traci.start(["sumo-gui", "-c", CFG])
    tls = traci.trafficlight.getIDList()
    tl = tls[0] if tls else None
    step = 0
    total_wait = 0.0
    vehicles = set()
    while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
        phase = (step // green_time) % 2
        if tl:
            traci.trafficlight.setPhase(tl, phase)
        traci.simulationStep()
        for vid in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(vid)
            vehicles.add(vid)
        step += 1
    traci.close()
    avg_wait = total_wait / max(1, len(vehicles))
    return {"avg_wait": avg_wait, "vehicles": len(vehicles)}

def main():
    results = []
    for g in [5,10,15]:
        res = run_fixed(g)
        print("green", g, "->", res)
        results.append({"green": g, **res})
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["green","avg_wait","vehicles"])
        writer.writeheader()
        writer.writerows(results)
    print("Saved baseline metrics to", OUT_CSV)

if __name__ == "__main__":
    main()
