"""
Delivery Optimization — Assessment 3

Approach:
1. Sort deliveries by Priority + Distance (REQUIRED)
2. Assign using ILP (balance workload)
3. Optimize route using TSP (advanced improvement)
4. Batch evaluation + business insights (analyst-level)

"""

import math
import os
import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, PULP_CBC_CMD
)

# ── Configuration ─────────────────────────────────────────────

INPUT_FOLDER = "inputs"     # folder with multiple CSVs
OUTPUT_FOLDER = "outputs"   # generated outputs
NUM_AGENTS = 3

PRIORITY_MAP = {"High": 3, "Medium": 2, "Low": 1}

# ── 1. Load Data ─────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath)

    required_cols = {"Location_ID", "Distance", "Priority"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    df["PriorityScore"] = df["Priority"].map(PRIORITY_MAP)
    return df

# ── 2. Sort ─────────────────────────────────────────────────

def sort_deliveries(df):
    return df.sort_values(
        by=["PriorityScore", "Distance"],
        ascending=[False, True]
    ).reset_index(drop=True)

# ── 3. ILP Assignment ───────────────────────────────────────

def solve_ilp(df, num_agents=NUM_AGENTS):
    n = len(df)
    distances = df["Distance"].tolist()
    agents = [f"Agent_{i+1}" for i in range(num_agents)]

    max_per_agent = math.ceil(n / num_agents)

    prob = LpProblem("Delivery_Assignment", LpMinimize)

    x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary)
          for j in range(num_agents)]
         for i in range(n)]

    max_dist = LpVariable("max_dist", lowBound=0)
    min_dist = LpVariable("min_dist", lowBound=0)

    prob += max_dist - min_dist

    for i in range(n):
        prob += lpSum(x[i][j] for j in range(num_agents)) == 1

    for j in range(num_agents):
        total = lpSum(x[i][j] * distances[i] for i in range(n))
        prob += total <= max_dist
        prob += total >= min_dist
        prob += lpSum(x[i][j] for i in range(n)) <= max_per_agent

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.05))

    assignment = {a: [] for a in agents}

    for i in range(n):
        for j in range(num_agents):
            if value(x[i][j]) > 0.5:
                assignment[agents[j]].append(i)
                break

    return assignment

# ── 4. TSP ──────────────────────────────────────────────────

def tsp_nearest_neighbor(distances):
    n = len(distances)
    if n == 0:
        return []

    visited = [False] * n
    route = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = route[-1]
        next_node = None
        min_dist = float("inf")

        for j in range(n):
            if not visited[j]:
                dist = abs(distances[last] - distances[j])
                if dist < min_dist:
                    min_dist = dist
                    next_node = j

        route.append(next_node)
        visited[next_node] = True

    return route

# ── 5. Build Output ─────────────────────────────────────────

def build_output(df, assignment):
    rows = []

    for agent, indices in assignment.items():
        agent_df = df.iloc[indices].copy()

        agent_df = agent_df.sort_values(
            by=["PriorityScore"],
            ascending=[False]
        ).reset_index(drop=True)

        optimized_groups = []

        for priority in ["High", "Medium", "Low"]:
            group = agent_df[agent_df["Priority"] == priority].reset_index(drop=True)

            if len(group) > 1:
                order = tsp_nearest_neighbor(group["Distance"].tolist())
                group = group.iloc[order]

            optimized_groups.append(group)

        agent_df = pd.concat(optimized_groups).reset_index(drop=True)

        cumulative = 0
        for seq, (_, row) in enumerate(agent_df.iterrows(), start=1):
            cumulative += row["Distance"]
            rows.append({
                "Agent": agent,
                "Location_ID": row["Location_ID"],
                "Priority": row["Priority"],
                "Distance_km": row["Distance"],
                "Sequence": seq,
                "Cumulative_km": cumulative
            })

    return pd.DataFrame(rows)

# ── 6. Metrics ─────────────────────────────────────────────

def compute_imbalance(output_df):
    totals = output_df.groupby("Agent")["Distance_km"].sum()
    return (totals.max() - totals.min()) / totals.mean() * 100

def compute_priority_compliance(output_df):
    high_in_top = 0
    total_high = 0

    for agent, group in output_df.groupby("Agent"):
        n = len(group)
        top_k = max(1, -(-n // 3))

        high = group[group["Priority"] == "High"]
        total_high += len(high)
        high_in_top += (high["Sequence"] <= top_k).sum()

    return (high_in_top / total_high) * 100 if total_high > 0 else 100

# ── 7. Reporting (NEW) ─────────────────────────────────────

def print_report(file, output_df, imbalance, compliance):
    print("\n" + "=" * 60)
    print(f"  DELIVERY REPORT -> {file}")
    print("=" * 60)

    agent_totals = {}

    for agent, group in output_df.groupby("Agent"):
        total = group["Distance_km"].sum()
        agent_totals[agent] = total

        print(f"\n{agent}")
        print(f"  Locations : {group['Location_ID'].tolist()}")
        print(f"  Total     : {total} km")
        print(f"  Deliveries: {len(group)}")

    totals = list(agent_totals.values())

    print("\n" + "-" * 60)
    print("QUALITY METRICS")
    print("-" * 60)

    print(f"\nImbalance Score     : {imbalance:.2f}%")
    print(f"Status              : {'PASS' if imbalance < 5 else 'REVIEW'}")

    print(f"\nPriority Compliance : {compliance:.2f}%")

    print(f"\nMax Distance        : {max(totals)} km")
    print(f"Min Distance        : {min(totals)} km")
    print(f"Mean Distance       : {np.mean(totals):.2f} km")

    print("\n" + "-" * 60)
    print("BUSINESS INSIGHTS")
    print("-" * 60)

    if imbalance < 5:
        print(" Balanced workload -> efficient agent utilization")
    else:
        print(" Imbalance detected -> inefficient distribution")

    if compliance >= 90:
        print(" Strong priority handling -> high SLA performance")
    elif compliance >= 75:
        print(" Moderate priority handling -> possible delays")
    else:
        print(" Poor priority handling -> SLA risk")

    print(" Optimized routing -> reduced travel cost & time")

    print("=" * 60)

# ── 8. Batch Runner (NEW) ───────────────────────────────────

def batch_run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    summary = []

    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".csv"):
            input_path = os.path.join(INPUT_FOLDER, file)
            output_path = os.path.join(OUTPUT_FOLDER, file.replace("input", "output"))

            df = load_data(input_path)
            df = sort_deliveries(df)

            assignment = solve_ilp(df)
            output_df = build_output(df, assignment)

            output_df.to_csv(output_path, index=False)

            imbalance = compute_imbalance(output_df)
            compliance = compute_priority_compliance(output_df)

            print_report(file, output_df, imbalance, compliance)

            summary.append((file, imbalance, compliance))

    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for file, imb, comp in summary:
        print(f"{file} -> Imbalance: {imb:.2f}% | Compliance: {comp:.2f}%")

    print("=" * 60)

# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    batch_run()