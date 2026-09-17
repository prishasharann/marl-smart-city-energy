# Multi-Objective Reinforcement Learning for Smart City Energy Management

Research project for the **WSU Online Research Internship Program 2026**,
advised by Dr. Surender Reddy Salkuti, Woosong University.

## Overview
Proposes and evaluates the **Adaptive Multi-Objective Reward Management (AMORM)**
framework for cooperative multi-agent energy management in cyber-physical smart cities.
Four cooperative agents (Building A, Building B, Battery, Grid Interface) are trained
using MAPPO under the CTDE paradigm to simultaneously optimise electricity cost,
carbon emissions, and grid stability.

## Dataset
Real building energy data from the
[Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2).

## Notebooks
| Notebook | Description |
|---|---|
| 01_data_exploration | Data loading, cleaning, and visualisation |
| 02_environment | Custom PettingZoo microgrid environment |
| 03_training | MAPPO training across 4 reward configurations |
| 04_evaluation | Test episode evaluation and metrics |
| 05_human_override | Human-in-the-loop operator preference test |

## Key Results
- Config D (AMORM) achieved the highest Hypervolume Indicator (4.809)
- ~20% lower Reward Signal Variance vs discrete adaptive baseline
- Interpretable domain-informed weight adaptation without learned preference models

## Stack
Python · PyTorch · PettingZoo · NumPy · Pandas · Jupyter Notebook
