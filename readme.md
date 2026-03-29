# Logistics/ Delivery Optimization problem

## Overview

The objective was not just to assign deliveries, but to build a system that:
- Balances workload across agents  
- Handles delivery priorities effectively  
- Produces efficient execution plans  
- Remains reliable under real-world constraints  

## Problem Statement

Given a CSV file containing:
- Location ID  
- Distance from warehouse  
- Delivery Priority (High / Medium / Low)  

The system must:

1. Read CSV  
2. Sort deliveries by priority and distance  
3. Assign deliveries to 3 agents  
4. Ensure nearly equal workload  
5. Generate delivery plans with execution order  

## Project Evolution (Approachs i have done)

I initially started with a simple heuristic based approach and gradually refined the system through multiple iterations.

- Early versions focused on basic sorting and greedy assignment  
- Later versions introduced constraints and optimization logic  
- ILP was incorporated to achieve balanced workload distribution  
- Routing and sequencing were added to improve execution order  
- Finally, robustness features like fallback handling, validation, and outlier detection were introduced  

This iterative process helped transform the solution from a basic model into a more structured and reliable system.

## Final System's Pipeline

The system follows a structured pipeline:

1. Load and validate input data  
2. Clean and normalize priority values  
3. Sort deliveries (Priority → Distance → Location_ID[used in tie breaker situation eg: 2 locations have same priority and same distance])  
4. Assign deliveries using Integer Linear Programming (ILP)  
5. Apply fallback strategy if optimization fails  
6. Sequence deliveries within each agent  
7. Detect outlier deliveries  
8. Compute performance metrics  
9. Generate reports and insights  

## Current Approach

### 1. Sorting Strategy

Deliveries are sorted using:
- Priority (High → Low)  
- Distance (Low → High)  
- Location_ID (tie-breaker for consistency)  

This ensures urgent deliveries are considered first.


### 2. Assignment using ILP

The assignment problem is modeled using **Integer Linear Programming (ILP)**.

#Objective:
Minimize workload imbalance:
```
minimize (max_distance − min_distance)
```
Constraints:
- Each delivery is assigned exactly once  
- Each agent has limited deliveries  
- Total distance is balanced across agents  
- High-priority deliveries are distributed across agents  

This ensures fairness while avoiding clustering of critical deliveries.

### 3. Fallback Strategy

If the ILP solver fails or returns an incomplete solution:

- A greedy assignment is used  
- Deliveries are assigned to the least-loaded agent  

This guarantees the system always produces a valid output.

### 4. Route Sequencing
Within each agent:

- Deliveries are grouped by priority  
- High → Medium → Low order is enforced  
- A nearest-neighbour heuristic is used for ordering  

This ensures priority-aware execution with reasonable efficiency.

### 5. Outlier Detection

Unusually long-distance deliveries are detected using:
threshold = mean + 2 × standard deviation

These are flagged because they:
- Distort workload balancing  
- May require separate scheduling  

### 6. Metrics & Insights

The system evaluates performance using:

- **Imbalance Score** → workload fairness  
- **Priority Compliance** → SLA adherence  

It also generates business insights such as:
- Efficiency of resource utilization  
- Potential SLA risks  
- Detection of abnormal deliveries  

## Sample Output

| Agent   | Location_ID | Priority | Distance_km | Sequence | Cumulative_km |
|---------|------------|----------|-------------|----------|---------------|
| Agent_1 | L1         | High     | 10          | 1        | 10            |


## Key Design Decisions

- Distance is used as a proxy for delivery effort due to limited input data  
- Priority is enforced during sorting and sequencing instead of the ILP objective  
- High-priority deliveries are explicitly distributed across agents  
- A fallback mechanism ensures system reliability  


## Limitations

- Distance is radial (from warehouse) and does not represent actual travel path  
- Route sequencing is heuristic due to lack of geographic coordinates  
- Number of stops per agent is not considered in workload balancing  
- Optimization is distance-based, not time-based  


## How to Run

1. Place input CSV files in the `inputs/` folder  
2. Run:
python main.py

3. Output files will be generated in the `outputs/` folder  

## Technologies Used

- Python  
- Pandas  
- NumPy  
- PuLP (Integer Linear Programming)  

## Final Note

This project reflects my approach to problem-solving —> starting simple, identifying limitations, and progressively improving the system to handle both technical and practical challenges.

The final solution balances:
-> efficiency  
-> fairness  
-> reliability  
-> and business relevance  
