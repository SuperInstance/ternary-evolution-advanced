# ternary-evolution-advanced

Advanced evolutionary algorithms for ternary optimization — differential evolution, CMA-ES-like adaptation, NSGA-II multi-objective optimization, speciation, niching, and crowding distance.

## Why This Exists

The basic `ternary-ga` crate handles standard genetic algorithms. But real-world ternary optimization often requires more sophisticated approaches: multi-objective problems with Pareto fronts, continuous-to-ternary adaptation strategies, or population diversity management. This crate provides advanced evolutionary algorithms specifically designed for the ternary search space `{-1, 0, +1}`.

## Core Concepts

- **DifferentialEvolution** — DE/rand/1/bin variant adapted for ternary genomes via clamp-to-ternary mutation
- **TernaryCMAES** — CMA-ES-like distribution adaptation maintaining per-gene ternary probabilities
- **NSGA2** — Multi-objective optimizer with non-dominated sorting and crowding distance
- **Speciation** — Group individuals by Hamming distance to preserve diversity
- **Fitness Sharing** — Niching to prevent premature convergence

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-evolution-advanced = "0.1"
```

```rust
use ternary_evolution_advanced::*;

// --- Single-objective: Differential Evolution ---
let fitness: FitnessFn = |genome: &TernaryGenome| {
    vec![genome.iter().map(|&v| v as f64).sum()]  // maximize sum
};

let mut de = DifferentialEvolution::new(30, 10, 0.9, 0.8, fitness, 42);
let mut seed = 123u64;
de.run(100, &mut seed);
println!("DE best fitness: {:.1}", de.best().fitness[0]);

// --- Single-objective: CMA-ES-like adaptation ---
let mut cma = TernaryCMAES::new(10, 30, fitness);
cma.run(50, &mut seed);
let best = cma.best.as_ref().unwrap();
println!("CMA best fitness: {:.1}", best.fitness[0]);
println!("Probabilities for gene 0: {:?}", cma.probabilities[0]);

// --- Multi-objective: NSGA-II ---
let multi_fitness: FitnessFn = |genome: &TernaryGenome| {
    let sum: f64 = genome.iter().map(|&v| v as f64).sum();
    let ones = genome.iter().filter(|&&v| v == 1).count() as f64;
    vec![sum, ones]  // two objectives
};

let mut nsga = NSGA2::new(50, 10, multi_fitness, 42);
nsga.run(30, &mut seed);

let pareto = nsga.pareto_front();
println!("Pareto front size: {}", pareto.len());
for ind in pareto {
    println!("  objectives: {:?}", ind.fitness);
}

// --- Diversity management ---
let mut pop: Vec<Individual> = (0..20)
    .map(|i| Individual::new(vec![if i % 2 == 0 { 1 } else { -1 }; 5]))
    .collect();

let n_species = speciate(&mut pop, 2);
println!("Species found: {}", n_species);

// Fitness sharing to reduce dominance
pop[0].fitness = vec![10.0];
pop[1].fitness = vec![10.0];
fitness_sharing(&mut pop, 3);
```

## API Overview

| Type / Function | Description |
|---|---|
| `TernaryGenome` | Type alias for `Vec<i8>` (−1, 0, +1 values) |
| `Individual` | Genome with fitness vector, rank, crowding distance, species |
| `DifferentialEvolution` | DE optimizer with `step`, `run`, `best` |
| `TernaryCMAES` | Distribution-based optimizer with adaptive probabilities |
| `NSGA2` | Multi-objective optimizer with `pareto_front` |
| `non_dominated_sort` | Pareto front extraction |
| `crowding_distance` | Diversity metric for NSGA-II |
| `speciate` | Group population by Hamming distance |
| `fitness_sharing` | Niche-based fitness reduction |

## How It Works

**Differential Evolution**: For each individual, three random parents are selected. A trial vector is formed as `a + F × (b − c)`, then clamped to ternary via rounding. If the trial's fitness exceeds the current individual's, it replaces it.

**TernaryCMAES**: Maintains per-gene probability distributions `[p_neg, p_zero, p_pos]`. Each generation samples a population, evaluates fitness, selects the elite, and adapts probabilities toward values favored by high-fitness individuals. The mean and evolution path track the search center.

**NSGA-II**: Combines parent and offspring populations, performs non-dominated sorting into Pareto fronts, assigns crowding distances, and selects the top `pop_size` individuals. Tournament selection uses rank then crowding distance as tiebreakers.

**Speciation**: Assigns individuals to species based on Hamming distance to species representatives. The first individual starts a species; others join if close enough to an existing representative.

**Fitness Sharing**: Reduces fitness of individuals in crowded niches: `shared_fitness = raw_fitness / niche_count`, where niche count sums overlap with nearby individuals.

## Use Cases

1. **Multi-objective architecture search** — Simultaneously optimize accuracy, latency, and model size of ternary neural networks
2. **Portfolio optimization with ternary signals** — Buy/sell/hold strategies with multiple risk-return objectives
3. **Game AI behavior tuning** — Evolve diverse strategies via speciation to prevent exploitable monocultures
4. **Constraint satisfaction** — Use CMA-ES adaptation to efficiently explore ternary constraint spaces

## Ecosystem

Part of the **SuperInstance** ternary computing crate family:

- `ternary-compression-v2` — Multi-algorithm ternary compression
- `ternary-hash` — Hashing and fingerprinting for ternary data
- `ternary-pca` — Principal component analysis on ternary values
- `ternary-ga` — Basic genetic algorithms with ternary genomes
- `ternary-matrix` — Compact ternary matrix operations
- `ternary-reservoir` — Echo state networks with ternary nodes
- `ternary-geometry` — Geometric algorithms in ternary space
- `ternary-causality` — Causal inference for ternary systems
- `ternary-consensus` — Distributed consensus for ternary agents

## License

MIT

## See Also
- **ternary-ga** — related
- **ternary-fitness** — related
- **ternary-genome** — related
- **ternary-popgen** — related
- **ternary-swarm** — related

