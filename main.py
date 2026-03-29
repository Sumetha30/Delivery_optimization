import os
import math
import pandas as pd
import numpy as np
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    LpBinary,
    lpSum,
    value,
    PULP_CBC_CMD
)

INPUT_FOLDER= "inputs"
OUTPUT_FOLDER = "outputs"
NUM_AGENTS= 3
PRIORITY_MAP= {
    "High": 3,
    "Medium": 2,
    "Low": 1
}

# ---------------------Load and Validate the Data-------------------------------
def load_data(filepath):
    df =pd.read_csv(filepath)

    required_cols ={"Location_ID", "Distance", "Priority"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    df["Priority"] =df["Priority"].str.strip().str.capitalize()
    df["PriorityScore"]= df["Priority"].map(PRIORITY_MAP)
    if df["PriorityScore"].isna().any():
        bad = df[df["PriorityScore"].isna()]["Priority"].unique().tolist()
        raise ValueError(f"Unrecognised Priority values: {bad}. Expected High / Medium / Low.")
    return df

# ---------------------Sorting-------------------------------
def sort_deliveries(df):
    return df.sort_values(
        by=["PriorityScore", "Distance", "Location_ID"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

# -----------------------Greedy (Fallback appraoch- used when ILP gets failed----------------
def greedy_fallback(df, num_agents=NUM_AGENTS):
    agents= [f"Agent_{i+1}" for i in range(num_agents)]
    totals ={a: 0 for a in agents}
    assignment= {a: [] for a in agents}

    for i, row in df.iterrows():
        lightest= min(totals, key=totals.get)
        assignment[lightest].append(i)
        totals[lightest] += row["Distance"]
    return assignment

# -----------------------ILP Assignment to agents-------------
def solve_ilp(df, num_agents=NUM_AGENTS):
    n =len(df)
    distances = df["Distance"].tolist()
    agents= [f"Agent_{i+1}" for i in range(num_agents)]
    if n <= num_agents:
        assignment= {a: [] for a in agents}
        for idx, agent in enumerate(agents[:n]):
            assignment[agent].append(idx)
        return assignment

    prob = LpProblem("Delivery_Assignment", LpMinimize)

    x= [
        [LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(num_agents)]
        for i in range(n)
    ]

    max_dist= LpVariable("max_dist", lowBound=0)
    min_dist =LpVariable("min_dist", lowBound=0)
    max_per= math.ceil(n / num_agents)
    prob+= max_dist - min_dist

    for i in range(n):
        prob +=lpSum(x[i][j] for j in range(num_agents)) == 1
    for j in range(num_agents):
        total_distance = lpSum(x[i][j] * distances[i] for i in range(n))
        prob += total_distance <= max_dist
        prob += total_distance >= min_dist
        prob += lpSum(x[i][j] for i in range(n)) <= max_per
    high_indices= [i for i in range(n) if df.iloc[i]["Priority"] == "High"]
    n_high = len(high_indices)

    if n_high > 0:
        max_high_per_agent = math.ceil(n_high / num_agents)
        for j in range(num_agents):
            prob += lpSum(x[i][j] for i in high_indices) <= max_high_per_agent

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10, gapRel=0.05))

    if prob.status not in [1, -2]:
        print(f"  WARNING: ILP solver failed (status {prob.status}). Using greedy fallback.")
        return greedy_fallback(df, num_agents)
    assignment = {a: [] for a in agents}

    for i in range(n):
        for j in range(num_agents):
            if value(x[i][j]) is not None and value(x[i][j]) > 0.5:
                assignment[agents[j]].append(i)
                break

    assigned= {idx for indices in assignment.values() for idx in indices}
    unassigned = [i for i in range(n) if i not in assigned]

    if unassigned:
        print(f"  WARNING: {len(unassigned)} deliveries unassigned by ILP. Assigning via fallback.")
        agent_totals = {
            a: sum(df.iloc[idx]["Distance"] for idx in assignment[a])
            for a in agents
        }
        for idx in unassigned:
            lightest = min(agent_totals, key=agent_totals.get)
            assignment[lightest].append(idx)
            agent_totals[lightest] += df.iloc[idx]["Distance"]
    return assignment
# ---------------Route Sequencing (TSP Nearest Neighbour)--------------
def tsp_nn(distances):

    if len(distances)== 0:
        return []
    n = len(distances)
    visited = [False] * n
    start = distances.index(min(distances))
    route= [start]
    visited[start] =True

    for _ in range(n - 1):
        last = route[-1]
        next_node= min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: abs(distances[last] - distances[j])
        )
        route.append(next_node)
        visited[next_node] = True
    return route
# ---------------------output-------------------------------
def build_output(df, assignment):
    rows = []
    for agent, indices in assignment.items():
        agent_df = df.iloc[indices].copy()
        agent_df = agent_df.sort_values(
            by="PriorityScore",
            ascending=False
        ).reset_index(drop=True)
        groups = []
        for priority in ["High", "Medium", "Low"]:
            grp = agent_df[agent_df["Priority"] == priority].reset_index(drop=True)
            if len(grp) > 1:
                order = tsp_nn(grp["Distance"].tolist())
                grp = grp.iloc[order]
            groups.append(grp)
        cumulative = 0
        combined = pd.concat(groups).reset_index(drop=True)

        for seq, (_, row) in enumerate(combined.iterrows(), start=1):
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

#----------------------- metrics---------------------------------
def compute_imbalance(df):
    totals = df.groupby("Agent")["Distance_km"].sum()
    return (totals.max() - totals.min()) / totals.mean() * 100

def compute_priority_compliance(df):
    if len(set(df["Priority"])) == 1:
        return 100.0
    high_top = 0
    total_high = 0
    for _, group in df.groupby("Agent"):
        top_k = max(1, -(-len(group) // 3))
        high = group[group["Priority"] == "High"]
        total_high+= len(high)
        high_top += (high["Sequence"] <= top_k).sum()
    return (high_top / total_high * 100) if total_high else 100.0

# ---------------------outlier Detection-------------------------------
def flag_outliers(df):
    mean_dist = df["Distance"].mean()
    std_dist  = df["Distance"].std()

    if std_dist == 0:
        return [], 0                               
    threshold = mean_dist + 2 * std_dist
    outliers  = df[df["Distance"] > threshold]["Location_ID"].tolist()
    return outliers, threshold

# ---------------------Reporting-------------------------------
def print_report(file, df, imbalance, compliance, source_df):

    edge_case = len(df)<= NUM_AGENTS
    print(f"DELIVERY REPORT -> {file}")
    print(f"{'='*60}")

    totals ={}
    for agent, group in df.groupby("Agent"):
        total = group["Distance_km"].sum()
        totals[agent] = total
        print(f"\n{agent}")
        print(f"  Locations : {group['Location_ID'].tolist()}")
        print(f"  Total     : {total} km")
        print(f"  Deliveries: {len(group)}")
    values =list(totals.values())

    print("\nQUALITY METRICS")
    print(f"Imbalance Score     : {imbalance:.2f}%")
    print(f"Priority Compliance : {compliance:.2f}%")
    print(f"Max / Min / Mean    : {max(values)} / {min(values)} / {np.mean(values):.2f}")

    outlier_result = flag_outliers(source_df)
    if outlier_result:
        outliers, threshold = outlier_result
        if outliers:
            print(f"\n  OUTLIER ALERT: {outliers} exceed {threshold:.0f}km threshold.")
            print(f"  Consider scheduling separately or assigning a dedicated agent.")

    print("\nBUSINESS INSIGHTS")

    if edge_case:
        print(" Low volume -> consolidate routes")
    elif imbalance < 5:
        print(" Balanced workload -> efficient utilization")
    else:
        print(" Imbalance detected -> review distribution")

    if compliance >= 90:
        print(" Strong priority handling -> high SLA")
    elif compliance >= 75:
        print(" Moderate priority handling -> check SLA")
    else:
        print(" Poor priority handling -> SLA risk")
    print("=" * 60)

# ---------------------Batch run-------------------------------
def batch_run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv"))
    if not files:
        print("No CSV files found.")
        return
    summary = []
    for file in files:
        path = os.path.join(INPUT_FOLDER, file)
        df = load_data(path)
        df = sort_deliveries(df)
        assignment = solve_ilp(df)
        output     = build_output(df, assignment)
        output_path = os.path.join(OUTPUT_FOLDER, file.replace("input", "output"))
        output.to_csv(output_path, index=False)
        imbalance  = compute_imbalance(output)
        compliance = compute_priority_compliance(output)
        print_report(file, output, imbalance, compliance, source_df=df)
        summary.append((file, imbalance, compliance, len(output)))

    print("\nFINAL SUMMARY")
    for f, i, c, n in summary:
        status = "N/A" if n <= NUM_AGENTS else ("PASS" if i < 5 else "REVIEW")
        print(f"{f} | {i:.2f}% | {c:.2f}% | {status}")

if __name__ == "__main__":
    batch_run()