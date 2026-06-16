# When `iint` (active integer) is needed

## Current status

The 3-compartment cardiovascular model uses only:
- `aadc.iif(condition, val_true, val_false)` — conditional selection
- `aadc.iand(a, b)` — logical AND for compound conditions

This covers all valve logic, activation thresholds, and clamping.
`iint` is **not needed** for current Auckland CellML models.

## Use cases where `iint` would be needed

### 1. Lookup tables indexed by active variable

If the model uses a precomputed table with integer indexing:

```python
# Pressure-volume relationship from experimental data
pv_table = [0.0, 12.5, 45.0, 98.0, 160.0, ...]  # measured values
index = int(volume / dV)   # integer index from continuous variable
pressure = pv_table[index]  # lookup
```

With plain `int()`, the index becomes passive (gradient = 0).
With `iint`, the index stays active, and interpolation between
table entries can be differentiated.

### 2. Discrete state machines

If the model has discrete states (e.g., cardiac phase as integer):

```python
phase = 0  # 0=diastole, 1=isovolumic contraction, 2=ejection, 3=isovolumic relaxation
if pressure_lv > pressure_aortic:
    phase = 2
```

With `iint`, phase transitions can be tracked on the tape
and the adjoint knows which phase was active at each time.

### 3. Variable-length loops

If the number of iterations depends on an active variable:

```python
n_segments = int(vessel_length / segment_size)  # depends on parameter
for i in range(n_segments):
    compute_segment(i)
```

With `iint`, the loop count stays active. Without it, changing
`vessel_length` doesn't change the number of segments in replay.

### 4. Multi-scale model coupling

If a macro-model selects which micro-model to run:

```python
cell_type = classify(gene_expression)  # returns integer 0, 1, or 2
if cell_type == 0:
    run_cardiomyocyte_model()
elif cell_type == 1:
    run_fibroblast_model()
```

### 5. Branching vascular networks

If the model topology depends on parameters:

```python
n_branches = int(flow_rate / threshold)  # number of open branches
for b in range(n_branches):
    compute_branch_flow(b)
```

### 6. Adaptive mesh / time stepping inside the model

If the model internally adapts resolution:

```python
n_substeps = int(max_rate * dt / safety_factor)
for sub in range(n_substeps):
    micro_step(dt / n_substeps)
```

## When `iif` is sufficient (no `iint` needed)

- Valve open/close: `iif(u_ra >= u_rv, opening_rate, closing_rate)` ✓
- Activation threshold: `iif(chi >= 0.5, active_value, 0.0)` ✓
- Clamping: `iif(x >= 0.0, x, 0.0)` ✓
- Compound conditions: `iand(iif(a>=b, 1, 0), iif(c>=d, 1, 0))` ✓
- Piecewise functions with known breakpoints ✓

## Summary

| Feature | Current models | Future models |
|---|---|---|
| `iif` | ✅ needed, used | ✅ always needed |
| `iand`/`ior` | ✅ needed for compound conditions | ✅ |
| `iint` | ❌ not needed | Maybe — lookup tables, state machines |
| `ibool` | ❌ not needed | Maybe — complex logic |
