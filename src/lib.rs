#![forbid(unsafe_code)]

//! Advanced evolutionary algorithms for ternary optimization:
//! differential evolution, CMA-ES-like adaptation, NSGA-II multi-objective,
//! speciation, niching, and crowding distance.

/// Ternary genome: a vector of values in {-1, 0, +1}.
pub type TernaryGenome = Vec<i8>;

/// Create a random ternary genome using a simple LCG PRNG.
pub fn random_genome(len: usize, seed: &mut u64) -> TernaryGenome {
    let next = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    };
    (0..len).map(|_| (next(seed) % 3) as i8 - 1).collect()
}

/// Clamp a value to ternary range {-1, 0, +1} by rounding.
pub fn clamp_ternary(v: f64) -> i8 {
    if v > 0.5 {
        1
    } else if v < -0.5 {
        -1
    } else {
        0
    }
}

/// An individual in the population.
#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: TernaryGenome,
    pub fitness: Vec<f64>,    // multi-objective: multiple fitness values
    pub rank: usize,          // Pareto rank
    pub crowding_distance: f64,
    pub species_id: usize,
}

impl Individual {
    pub fn new(genome: TernaryGenome) -> Self {
        Individual {
            genome,
            fitness: vec![],
            rank: 0,
            crowding_distance: 0.0,
            species_id: 0,
        }
    }

    /// Hamming distance to another individual.
    pub fn hamming_distance(&self, other: &Individual) -> usize {
        self.genome
            .iter()
            .zip(other.genome.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Euclidean distance (treating ternary as real).
    pub fn euclidean_distance(&self, other: &Individual) -> f64 {
        self.genome
            .iter()
            .zip(other.genome.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// A fitness function type: takes a ternary genome and returns fitness values.
pub type FitnessFn = fn(&TernaryGenome) -> Vec<f64>;

/// Differential evolution adapted for ternary genomes.
pub struct DifferentialEvolution {
    pub pop_size: usize,
    pub genome_len: usize,
    pub cr: f64,  // crossover rate
    pub f: f64,   // differential weight
    pub population: Vec<Individual>,
    pub fitness_fn: FitnessFn,
}

impl DifferentialEvolution {
    pub fn new(pop_size: usize, genome_len: usize, cr: f64, f: f64, fitness_fn: FitnessFn, seed: u64) -> Self {
        let mut rng = seed;
        let population: Vec<Individual> = (0..pop_size)
            .map(|_| {
                let genome = random_genome(genome_len, &mut rng);
                let mut ind = Individual::new(genome);
                ind.fitness = fitness_fn(&ind.genome);
                ind
            })
            .collect();

        DifferentialEvolution {
            pop_size,
            genome_len,
            cr,
            f,
            population,
            fitness_fn,
        }
    }

    /// Run one generation of DE (DE/rand/1/bin variant).
    pub fn step(&mut self, seed: &mut u64) {
        let next = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        };

        let old_pop = self.population.clone();

        for i in 0..self.pop_size {
            // Pick 3 distinct random individuals
            let a = (next(seed) as usize) % self.pop_size;
            let b = (next(seed) as usize) % self.pop_size;
            let c = (next(seed) as usize) % self.pop_size;

            let mut trial = self.population[i].genome.clone();

            let r = (next(seed) as usize) % self.genome_len;

            for j in 0..self.genome_len {
                let ri = (next(seed) as u64) as f64 / u64::MAX as f64;
                if ri < self.cr || j == r {
                    // DE mutation: v = a + F * (b - c), then clamp to ternary
                    let va = old_pop[a].genome[j] as f64;
                    let vb = old_pop[b].genome[j] as f64;
                    let vc = old_pop[c].genome[j] as f64;
                    let mutated = va + self.f * (vb - vc);
                    trial[j] = clamp_ternary(mutated);
                }
            }

            let trial_fitness = (self.fitness_fn)(&trial);
            // Single-objective: use first fitness value
            let trial_f = trial_fitness[0];
            let current_f = self.population[i].fitness[0];

            if trial_f >= current_f {
                self.population[i].genome = trial;
                self.population[i].fitness = trial_fitness;
            }
        }
    }

    /// Run for multiple generations.
    pub fn run(&mut self, generations: usize, seed: &mut u64) {
        for _ in 0..generations {
            self.step(seed);
        }
    }

    /// Get the best individual.
    pub fn best(&self) -> &Individual {
        self.population
            .iter()
            .max_by(|a, b| a.fitness[0].partial_cmp(&b.fitness[0]).unwrap())
            .unwrap()
    }
}

/// CMA-ES-like adaptation for ternary optimization.
/// Maintains a probability distribution over ternary values and adapts it.
pub struct TernaryCMAES {
    pub genome_len: usize,
    pub pop_size: usize,
    /// Per-gene probabilities: [p_neg, p_zero, p_pos]
    pub probabilities: Vec<[f64; 3]>,
    /// Step size (adaptation rate)
    pub sigma: f64,
    /// Mean of the distribution
    pub mean: Vec<f64>,
    /// Evolution path for cumulation
    pub evolution_path: Vec<f64>,
    pub fitness_fn: FitnessFn,
    pub best: Option<Individual>,
    pub generation: usize,
}

impl TernaryCMAES {
    pub fn new(genome_len: usize, pop_size: usize, fitness_fn: FitnessFn) -> Self {
        let probabilities = vec![[1.0 / 3.0; 3]; genome_len];
        let mean = vec![0.0; genome_len];
        let evolution_path = vec![0.0; genome_len];

        TernaryCMAES {
            genome_len,
            pop_size,
            probabilities,
            sigma: 1.0,
            mean,
            evolution_path,
            fitness_fn,
            best: None,
            generation: 0,
        }
    }

    /// Sample a genome from current distribution.
    pub fn sample(&self, seed: &mut u64) -> TernaryGenome {
        let next = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        };

        (0..self.genome_len)
            .map(|i| {
                let r = (next(seed) as f64) / u64::MAX as f64;
                let probs = self.probabilities[i];
                if r < probs[0] {
                    -1
                } else if r < probs[0] + probs[1] {
                    0
                } else {
                    1
                }
            })
            .collect()
    }

    /// Run one generation: sample, evaluate, select, adapt.
    pub fn step(&mut self, seed: &mut u64) {
        // Sample population
        let mut pop: Vec<Individual> = (0..self.pop_size)
            .map(|_| {
                let genome = self.sample(seed);
                let mut ind = Individual::new(genome);
                ind.fitness = (self.fitness_fn)(&ind.genome);
                ind
            })
            .collect();

        // Sort by fitness (single objective, maximize)
        pop.sort_by(|a, b| b.fitness[0].partial_cmp(&a.fitness[0]).unwrap());

        // Update best
        if self.best.is_none() || pop[0].fitness[0] > self.best.as_ref().unwrap().fitness[0] {
            self.best = Some(pop[0].clone());
        }

        // Select top half for adaptation
        let elite_count = self.pop_size / 2;
        let elite = &pop[..elite_count];

        // Adapt probabilities
        let learning_rate = 0.2;
        for i in 0..self.genome_len {
            let mut counts = [0.0; 3];
            for ind in elite {
                let idx = (ind.genome[i] + 1) as usize;
                counts[idx] += 1.0 / elite_count as f64;
            }
            for k in 0..3 {
                self.probabilities[i][k] = (1.0 - learning_rate) * self.probabilities[i][k]
                    + learning_rate * counts[k];
            }
        }

        // Update mean
        for i in 0..self.genome_len {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for (idx, ind) in elite.iter().enumerate() {
                let w = (elite_count - idx) as f64;
                sum += w * ind.genome[i] as f64;
                weight_sum += w;
            }
            if weight_sum > 0.0 {
                self.mean[i] = sum / weight_sum;
            }
        }

        self.generation += 1;
    }

    /// Run for multiple generations.
    pub fn run(&mut self, generations: usize, seed: &mut u64) {
        for _ in 0..generations {
            self.step(seed);
        }
    }
}

/// Non-dominated sorting for NSGA-II.
pub fn non_dominated_sort(population: &[Individual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let n_objectives = if population.is_empty() { 0 } else { population[0].fitness.len() };

    if n_objectives == 0 {
        return (0..n).map(|i| vec![i]).collect();
    }

    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
    let mut fronts: Vec<Vec<usize>> = vec![];
    let mut assigned = vec![false; n];

    // Compute domination relations
    for i in 0..n {
        for j in (i + 1)..n {
            let dom = dominates(&population[i].fitness, &population[j].fitness);
            let dom_rev = dominates(&population[j].fitness, &population[i].fitness);
            if dom {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if dom_rev {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // Extract fronts
    loop {
        let front: Vec<usize> = (0..n)
            .filter(|&i| !assigned[i] && domination_count[i] == 0)
            .collect();
        if front.is_empty() {
            break;
        }
        for &i in &front {
            assigned[i] = true;
            for &j in &dominated_by[i] {
                domination_count[j] = domination_count[j].saturating_sub(1);
            }
        }
        fronts.push(front);
    }

    fronts
}

/// Check if fitness_a dominates fitness_b (maximization).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;
    for (va, vb) in a.iter().zip(b.iter()) {
        if va < vb {
            return false;
        }
        if va > vb {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

/// Compute crowding distance for individuals in a front.
pub fn crowding_distance(population: &[Individual], front: &[usize]) -> Vec<f64> {
    if front.len() <= 2 {
        return vec![f64::INFINITY; front.len()];
    }

    let n_objectives = population[front[0]].fitness.len();
    let mut distances = vec![0.0; front.len()];

    for obj in 0..n_objectives {
        // Sort front by objective
        let mut sorted: Vec<usize> = (0..front.len()).collect();
        sorted.sort_by(|&a, &b| {
            population[front[a]].fitness[obj]
                .partial_cmp(&population[front[b]].fitness[obj])
                .unwrap()
        });

        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[sorted.len() - 1]] = f64::INFINITY;

        let f_min = population[front[sorted[0]]].fitness[obj];
        let f_max = population[front[sorted[sorted.len() - 1]]].fitness[obj];
        let range = f_max - f_min;

        if range.abs() < 1e-15 {
            continue;
        }

        for k in 1..sorted.len() - 1 {
            let next_val = population[front[sorted[k + 1]]].fitness[obj];
            let prev_val = population[front[sorted[k - 1]]].fitness[obj];
            distances[sorted[k]] += (next_val - prev_val) / range;
        }
    }

    distances
}

/// Speciation: assign individuals to species based on distance threshold.
pub fn speciate(population: &mut [Individual], threshold: usize) -> usize {
    if population.is_empty() {
        return 0;
    }

    let mut species_representatives: Vec<usize> = vec![0];
    population[0].species_id = 0;

    for i in 1..population.len() {
        let mut assigned = false;
        for &rep_idx in &species_representatives {
            let dist = population[i].hamming_distance(&population[rep_idx]);
            if dist <= threshold {
                population[i].species_id = population[rep_idx].species_id;
                assigned = true;
                break;
            }
        }
        if !assigned {
            let new_species = species_representatives.len();
            population[i].species_id = new_species;
            species_representatives.push(i);
        }
    }

    species_representatives.len()
}

/// Niching: fitness sharing based on hamming distance.
pub fn fitness_sharing(population: &mut [Individual], sigma_share: usize) {
    let n = population.len();
    let mut shared_fitness = vec![0.0; n];

    for i in 0..n {
        let mut niche_count = 0.0;
        for j in 0..n {
            let dist = population[i].hamming_distance(&population[j]);
            if dist <= sigma_share {
                niche_count += 1.0 - (dist as f64 / sigma_share as f64);
            }
        }
        if niche_count > 0.0 {
            shared_fitness[i] = population[i].fitness[0] / niche_count;
        } else {
            shared_fitness[i] = population[i].fitness[0];
        }
    }

    for i in 0..n {
        population[i].fitness[0] = shared_fitness[i];
    }
}

/// NSGA-II style multi-objective optimizer.
pub struct NSGA2 {
    pub pop_size: usize,
    pub genome_len: usize,
    pub population: Vec<Individual>,
    pub fitness_fn: FitnessFn,
    pub generation: usize,
}

impl NSGA2 {
    pub fn new(pop_size: usize, genome_len: usize, fitness_fn: FitnessFn, seed: u64) -> Self {
        let mut rng = seed;
        let population: Vec<Individual> = (0..pop_size)
            .map(|_| {
                let genome = random_genome(genome_len, &mut rng);
                let mut ind = Individual::new(genome);
                ind.fitness = fitness_fn(&ind.genome);
                ind
            })
            .collect();

        NSGA2 {
            pop_size,
            genome_len,
            population,
            fitness_fn,
            generation: 0,
        }
    }

    /// Run one generation.
    pub fn step(&mut self, seed: &mut u64) {
        let next = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        };

        // Create offspring via tournament selection + mutation
        let mut offspring = vec![];
        for _ in 0..self.pop_size {
            // Tournament selection
            let a = (next(seed) as usize) % self.pop_size;
            let b = (next(seed) as usize) % self.pop_size;
            let parent_idx = if self.population[a].rank <= self.population[b].rank { a } else { b };

            let mut child_genome = self.population[parent_idx].genome.clone();

            // Mutate: flip random gene
            let pos = (next(seed) as usize) % self.genome_len;
            child_genome[pos] = (next(seed) as i8 % 3) - 1;

            let mut child = Individual::new(child_genome);
            child.fitness = (self.fitness_fn)(&child.genome);
            offspring.push(child);
        }

        // Combine parent + offspring
        let mut combined = self.population.clone();
        combined.extend(offspring);

        // Non-dominated sorting
        let fronts = non_dominated_sort(&combined);

        // Assign ranks and crowding distances
        for (rank, front) in fronts.iter().enumerate() {
            let distances = crowding_distance(&combined, front);
            for (idx, &ind_idx) in front.iter().enumerate() {
                combined[ind_idx].rank = rank;
                combined[ind_idx].crowding_distance = distances[idx];
            }
        }

        // Select top pop_size using rank and crowding distance
        let mut selected = vec![];
        for front in &fronts {
            if selected.len() + front.len() <= self.pop_size {
                for &idx in front {
                    selected.push(combined[idx].clone());
                }
            } else {
                let remaining = self.pop_size - selected.len();
                let mut front_inds: Vec<usize> = front.clone();
                front_inds.sort_by(|&a, &b| {
                    combined[b].crowding_distance
                        .partial_cmp(&combined[a].crowding_distance)
                        .unwrap()
                });
                for &idx in front_inds.iter().take(remaining) {
                    selected.push(combined[idx].clone());
                }
                break;
            }
            if selected.len() >= self.pop_size {
                break;
            }
        }

        self.population = selected;
        self.generation += 1;
    }

    /// Run for multiple generations.
    pub fn run(&mut self, generations: usize, seed: &mut u64) {
        for _ in 0..generations {
            self.step(seed);
        }
    }

    /// Get the Pareto front (rank 0 individuals).
    pub fn pareto_front(&self) -> Vec<&Individual> {
        self.population.iter().filter(|i| i.rank == 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple fitness: sum of genome values (maximize).
    fn sum_fitness(genome: &TernaryGenome) -> Vec<f64> {
        vec![genome.iter().map(|&v| v as f64).sum()]
    }

    /// Two-objective: maximize sum and maximize count of 1s.
    fn multi_fitness(genome: &TernaryGenome) -> Vec<f64> {
        let sum: f64 = genome.iter().map(|&v| v as f64).sum();
        let ones = genome.iter().filter(|&&v| v == 1).count() as f64;
        vec![sum, ones]
    }

    #[test]
    fn test_random_genome() {
        let mut seed = 42u64;
        let g = random_genome(10, &mut seed);
        assert_eq!(g.len(), 10);
        for &v in &g {
            assert!(v == -1 || v == 0 || v == 1);
        }
    }

    #[test]
    fn test_clamp_ternary() {
        assert_eq!(clamp_ternary(0.8), 1);
        assert_eq!(clamp_ternary(-0.7), -1);
        assert_eq!(clamp_ternary(0.2), 0);
    }

    #[test]
    fn test_individual_hamming() {
        let a = Individual::new(vec![1, 0, -1]);
        let b = Individual::new(vec![1, 1, -1]);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_individual_euclidean() {
        let a = Individual::new(vec![1, 0, -1]);
        let b = Individual::new(vec![-1, 0, 1]);
        let dist = a.euclidean_distance(&b);
        assert!((dist - (8f64).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_de_creation() {
        let de = DifferentialEvolution::new(20, 10, 0.9, 0.8, sum_fitness, 42);
        assert_eq!(de.population.len(), 20);
    }

    #[test]
    fn test_de_step() {
        let mut de = DifferentialEvolution::new(20, 10, 0.9, 0.8, sum_fitness, 42);
        let mut seed = 123u64;
        de.step(&mut seed);
        assert_eq!(de.population.len(), 20);
    }

    #[test]
    fn test_de_convergence() {
        let mut de = DifferentialEvolution::new(30, 5, 0.9, 0.8, sum_fitness, 42);
        let mut seed = 123u64;
        de.run(100, &mut seed);
        let best = de.best();
        // Should find all-ones or close
        assert!(best.fitness[0] > 3.0, "best fitness: {}", best.fitness[0]);
    }

    #[test]
    fn test_cmaes_creation() {
        let cma = TernaryCMAES::new(10, 20, sum_fitness);
        assert_eq!(cma.probabilities.len(), 10);
        assert_eq!(cma.pop_size, 20);
    }

    #[test]
    fn test_cmaes_sample() {
        let cma = TernaryCMAES::new(5, 10, sum_fitness);
        let mut seed = 42u64;
        let g = cma.sample(&mut seed);
        assert_eq!(g.len(), 5);
        for &v in &g {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn test_cmaes_step() {
        let mut cma = TernaryCMAES::new(10, 20, sum_fitness);
        let mut seed = 42u64;
        cma.step(&mut seed);
        assert!(cma.best.is_some());
    }

    #[test]
    fn test_cmaes_convergence() {
        let mut cma = TernaryCMAES::new(5, 30, sum_fitness);
        let mut seed = 42u64;
        cma.run(50, &mut seed);
        let best = cma.best.as_ref().unwrap();
        assert!(best.fitness[0] > 3.0, "best fitness: {}", best.fitness[0]);
    }

    #[test]
    fn test_non_dominated_sort() {
        let mut pop = vec![
            Individual::new(vec![1]),
            Individual::new(vec![-1]),
            Individual::new(vec![0]),
        ];
        pop[0].fitness = vec![3.0, 2.0];
        pop[1].fitness = vec![1.0, 1.0];
        pop[2].fitness = vec![2.0, 3.0];

        let fronts = non_dominated_sort(&pop);
        assert!(!fronts.is_empty());
        // pop[0] and pop[2] should be in front 0 (non-dominated)
        assert!(fronts[0].contains(&0) || fronts[0].contains(&2));
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let mut pop = vec![
            Individual::new(vec![1]),
            Individual::new(vec![0]),
        ];
        pop[0].fitness = vec![1.0, 2.0];
        pop[1].fitness = vec![2.0, 1.0];
        let front = vec![0, 1];
        let dists = crowding_distance(&pop, &front);
        assert!(dists.iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn test_speciate() {
        let mut pop = vec![
            Individual::new(vec![1, 1, 1]),
            Individual::new(vec![1, 1, 0]),
            Individual::new(vec![-1, -1, -1]),
        ];
        let n_species = speciate(&mut pop, 2);
        assert!(n_species >= 1);
    }

    #[test]
    fn test_niching() {
        let mut pop = vec![
            Individual::new(vec![1]),
            Individual::new(vec![1]),
        ];
        pop[0].fitness = vec![10.0];
        pop[1].fitness = vec![10.0];
        fitness_sharing(&mut pop, 2);
        // Shared fitness should be lower than original
        assert!(pop[0].fitness[0] < 10.0);
    }

    #[test]
    fn test_nsga2_creation() {
        let nsga = NSGA2::new(20, 5, multi_fitness, 42);
        assert_eq!(nsga.population.len(), 20);
    }

    #[test]
    fn test_nsga2_step() {
        let mut nsga = NSGA2::new(20, 5, multi_fitness, 42);
        let mut seed = 123u64;
        nsga.step(&mut seed);
        assert_eq!(nsga.population.len(), 20);
    }

    #[test]
    fn test_nsga2_pareto_front() {
        let mut nsga = NSGA2::new(30, 5, multi_fitness, 42);
        let mut seed = 123u64;
        nsga.run(20, &mut seed);
        let pf = nsga.pareto_front();
        assert!(!pf.is_empty());
    }

    #[test]
    fn test_dominates() {
        assert!(dominates(&[3.0, 2.0], &[1.0, 1.0]));
        assert!(!dominates(&[3.0, 1.0], &[1.0, 2.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_de_best() {
        let de = DifferentialEvolution::new(20, 10, 0.9, 0.8, sum_fitness, 42);
        let best = de.best();
        assert!(best.fitness[0] >= -10.0); // just verify it works
    }

    #[test]
    fn test_cmaes_probabilities_normalize() {
        let cma = TernaryCMAES::new(5, 10, sum_fitness);
        for probs in &cma.probabilities {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_crowding_distance_three_points() {
        let mut pop = vec![
            Individual::new(vec![1]),
            Individual::new(vec![0]),
            Individual::new(vec![-1]),
        ];
        pop[0].fitness = vec![1.0];
        pop[1].fitness = vec![2.0];
        pop[2].fitness = vec![3.0];
        let front = vec![0, 1, 2];
        let dists = crowding_distance(&pop, &front);
        // Boundary points should have infinite distance
        assert!(dists[0].is_infinite() || dists.iter().any(|d| d.is_infinite()));
    }
}
