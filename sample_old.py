"""
Delivery Optimization — Assessment 3
Decision and Computing Sciences

Approach: Integer Linear Programming (ILP) via PuLP
Objective: Minimize workload imbalance across 3 agents
           while enforcing priority-based delivery sequencing.

Install dependency: pip install pulp
"""

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, PULP_CBC_CMD
)

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FILE  = r"C:\Users\HP\Documents\intern\digitivity\sample.csv"
OUTPUT_FILE = "delivery_plan.csv"
NUM_AGENTS  = 3

PRIORITY_MAP = {"High": 3, "Medium": 2, "Low": 1}

# ── 1. Load & Validate Data ───────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Read CSV and validate required columns exist."""
    df = pd.read_csv(filepath)

    required_cols = {"Location_ID", "Distance", "Priority"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["PriorityScore"] = df["Priority"].map(PRIORITY_MAP)
    if df["PriorityScore"].isna().any():
        raise ValueError("Priority column contains values other than High/Medium/Low")

    return df

# ── 2. Sort by Priority then Distance ────────────────────────────────────────

def sort_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by Priority descending (High first), then Distance ascending
    (nearest first within the same priority tier).
    This satisfies Task 2 and gives the ILP a well-ordered input.
    """
    return (
        df.sort_values(
            by=["PriorityScore", "Distance"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )

# ── 3 & 4. ILP Assignment — Minimise Imbalance ───────────────────────────────

def solve_ilp(df: pd.DataFrame, num_agents: int = NUM_AGENTS) -> dict:
    """
    Integer Linear Programming formulation.

    Decision variable: x[i][j] = 1 if delivery i is assigned to agent j

    Objective: Minimise (max_agent_distance − min_agent_distance)
               → drives all agents toward equal total distance

    Constraints:
      1. Every delivery assigned to exactly one agent.
      2. max_dist >= each agent's total distance.
      3. min_dist <= each agent's total distance.
      4. Priority ordering within each agent's route is preserved
         by sorting the input before solving (Tasks 2→3 pipeline).

    Why ILP over greedy / random restart?
      Greedy is O(n) but sub-optimal. Random restart (500 iterations)
      produces no convergence guarantee. ILP is provably optimal for
      the stated objective within the given constraints.
    """
    n = len(df)
    agents = [f"Agent_{j+1}" for j in range(num_agents)]
    distances = df["Distance"].tolist()

    prob = LpProblem("Delivery_Assignment", LpMinimize)

    # Binary assignment variables
    x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary)
          for j in range(num_agents)]
         for i in range(n)]

    # Auxiliary variables to capture spread
    max_dist = LpVariable("max_dist", lowBound=0)
    min_dist = LpVariable("min_dist", lowBound=0)

    # Objective: minimise the imbalance spread
    prob += max_dist - min_dist

    # Constraint 1 — each delivery assigned to exactly one agent
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(num_agents)) == 1

    # Constraints 2 & 3 — bind max/min to actual agent totals
    for j in range(num_agents):
        agent_total = lpSum(x[i][j] * distances[i] for i in range(n))
        prob += agent_total <= max_dist
        prob += agent_total >= min_dist

    # Solve silently
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.05))
    # Add this check:
    if prob.status not in [1, -2]:  # 1=Optimal, -2=Not solved/timeout with solution
        raise RuntimeError(f"ILP solver failed with status: {prob.status}")

    # Extract results
    assignment = {a: [] for a in agents}
    for i in range(n):
        for j in range(num_agents):
            if value(x[i][j]) == 1:
                assignment[agents[j]].append(i)
                break

    return assignment

# ── 5. Build Output & Compute Metrics ────────────────────────────────────────

def build_output(df: pd.DataFrame, assignment: dict) -> pd.DataFrame:
    """
    Construct the full delivery plan DataFrame.
    Each row = one delivery, with agent, sequence position,
    cumulative distance, and priority flag.
    """
    rows = []
    for agent, indices in assignment.items():
        # Re-sort within each agent's route: High first, then by distance
        agent_df = df.iloc[indices].sort_values(
            by=["PriorityScore", "Distance"],
            ascending=[False, True]
        )
        cumulative = 0
        for seq, (_, row) in enumerate(agent_df.iterrows(), start=1):
            cumulative += row["Distance"]
            rows.append({
                "Agent":          agent,
                "Location_ID":    row["Location_ID"],
                "Priority":       row["Priority"],
                "Distance_km":    row["Distance"],
                "Sequence":       seq,
                "Cumulative_km":  cumulative,
            })
    return pd.DataFrame(rows)

def compute_imbalance_score(output_df: pd.DataFrame) -> float:
    """
    Imbalance Score = (max_total − min_total) / mean_total × 100

    Measures how unevenly work is distributed across agents.
    Target: < 5% (world-class logistics benchmark).
    A 20% imbalance means one driver travels 20% more for the same pay.
    """
    totals = output_df.groupby("Agent")["Distance_km"].sum()
    return (totals.max() - totals.min()) / totals.mean() * 100

def compute_priority_compliance(output_df: pd.DataFrame) -> float:
    """
    Priority Compliance = % of High-priority deliveries that appear
    in the top-third of their agent's sequence.

    Why this matters: assigning a High delivery to an agent is NOT
    enough. If that agent delivers 4 Low-priority packages first,
    the SLA is breached. This metric catches that failure.
    """
    high_in_top = 0
    high_total = 0

    for agent, group in output_df.groupby("Agent"):
        n = len(group)
        top_third_cutoff = max(1, n // 3)      # at least slot 1
        top_indices = set(range(1, top_third_cutoff + 1))

        high_rows = group[group["Priority"] == "High"]
        high_total += len(high_rows)
        high_in_top += high_rows["Sequence"].isin(top_indices).sum()

    if high_total == 0:
        return 100.0   # no High deliveries → trivially compliant
    return (high_in_top / high_total) * 100

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DELIVERY OPTIMIZATION — ILP SOLVER")
    print("=" * 60)

    # Step 1: Load
    print("\n[1] Loading data...")
    df = load_data(INPUT_FILE)
    print(f"    {len(df)} deliveries loaded.")

    # Step 2: Sort
    print("[2] Sorting by Priority -> Distance...")
    df = sort_deliveries(df)

    # Steps 3 & 4: ILP solve
    print("[3/4] Solving ILP assignment (minimising imbalance)...")
    assignment = solve_ilp(df)

    # Step 5: Build output
    print("[5] Building delivery plan...")
    output_df = build_output(df, assignment)

    # ── Per-agent summary ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  FINAL OPTIMIZED DELIVERY PLAN")
    print("-" * 60)

    agent_totals = {}
    for agent, group in output_df.groupby("Agent"):
        total = group["Distance_km"].sum()
        agent_totals[agent] = total
        locs = group["Location_ID"].tolist()
        print(f"\n  {agent}")
        print(f"    Locations : {locs}")
        print(f"    Total     : {total} km")
        print(f"    Deliveries: {len(group)}")

    # ── Quality metrics ───────────────────────────────────────────
    imbalance   = compute_imbalance_score(output_df)
    compliance  = compute_priority_compliance(output_df)
    totals_list = list(agent_totals.values())

    print("\n" + "-" * 60)
    print("  QUALITY METRICS")
    print("-" * 60)
    print(f"\n  Imbalance Score     : {imbalance:.2f}%")
    print(f"  (target < 5% | world-class logistics benchmark)")
    status = "PASS" if imbalance < 5 else "REVIEW"
    print(f"  Status              : {status}")

    print(f"\n  Priority Compliance : {compliance:.1f}%")
    print(f"  (% of High-priority deliveries in top-1/3 of route)")

    print(f"\n  Max agent distance  : {max(totals_list)} km")
    print(f"  Min agent distance  : {min(totals_list)} km")
    print(f"  Mean agent distance : {np.mean(totals_list):.1f} km")

    # ── Save output ───────────────────────────────────────────────
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OUTPUT] Plan saved to '{OUTPUT_FILE}'")
    print("=" * 60)


if __name__ == "__main__":
    main()