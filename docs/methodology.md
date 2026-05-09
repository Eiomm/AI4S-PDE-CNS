# Methodology Draft

## Agent Architecture

The Agent follows an observe-plan-act-record loop. It observes project files, race notes, experiment outputs, and logs. It asks an API-hosted LLM to plan the next step, executes only approved tools, and records every LLM response and action.

## LLM Logging

Every LLM call is written as one JSON line in `task{N}_logs.log`. Each line contains `timestamp`, `elapsed_seconds`, `provider`, `model`, `messages`, and `response`.

## Tool Use

The first version allows file reads, file writes inside approved roots, and allowlisted shell commands. Every write and shell command is recorded in the run manifest.

## PDE Modeling Plan

Task 1 will start from FNO through `neuraloperator` or the official PDEBench checkpoint. Task 2 will be trained from scratch and kept isolated from Task 1 data and checkpoint artifacts.
