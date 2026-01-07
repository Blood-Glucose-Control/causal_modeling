# Synthetic Glucose Data Generation

## Time Series Parameters
- Sampling frequency: 5 min
- Time range: [t₀, t₀ + n days]
- Base glucose (G₀): 100 mg/dL

## Core Dynamics

### Insulin Activity
```
I(t) = D * [0.8 * exp(-((t/tp - 1)²) * 4) + 0.2 * exp(-t/td)] * 1{t < td}
where:
- D = insulin dose (units)
- tp = 75 min (peak time)
- td = 300 min (duration)
```

### Carbohydrate Impact
```
C(t) = G * (t/tp) * exp(1 - t/tp) * 1{t < td}
where:
- G = carb grams
- tp = 45 min (peak time)
- td = 180 min (duration)
```

### Dawn Effect
```
D(h) = 20 * sin(π * (h-4)/(10-4)) * 1{4 ≤ h ≤ 10}
where h = hour of day
```

## Daily Events Distribution

### Meals
- Breakfast: N(8h, 0.5h), U(40g, 60g)
- Lunch: N(13h, 0.5h), U(50g, 80g)
- Dinner: N(19h, 0.5h), U(45g, 70g)

### Insulin
- Timing: meal_time + N(-15min, 10min)
- Dose: counted_carbs/10, where counted_carbs = true_carbs * N(1, 0.1)

### Exercise
- P(exercise) = 0.7/day
- Time: U(14h, 20h)
- Duration: 45min
- Effect: insulin_sensitivity * 1.2

### Stress
- P(stress) = 0.4/day
- Time: U(9h, 17h)
- Duration: 120min
- Intensity: N(0.7, 0.2)
- Effect: intensity * 30 mg/dL

## Glucose Evolution
```
G(t) = 0.9G(t-1) + 0.1[G₀ + ΣC(t) - ΣI(t)E(t) + S(t) + D(t) + N(0, 2)]
where:
- G(t) = glucose at time t
- ΣC(t) = sum of active carb effects
- ΣI(t) = sum of active insulin effects
- E(t) = exercise multiplier (1 or 1.2)
- S(t) = stress effect
- D(t) = dawn effect
```

## Constraints
- G(t) ∈ [40, 400] mg/dL
- carbs, insulin ≥ 0
- exercise ∈ {0, 1}
- stress ∈ [0, 1]
- meal_insulin_delay ~ N(-15, 10) min

## Visualization
The generated data includes an interactive plot showing:
- Continuous glucose curve
- Meal markers (size proportional to carbs)
- Insulin doses (size proportional to units)
- Exercise periods
- Stress events
- Target range guidelines (70-180 mg/dL) 