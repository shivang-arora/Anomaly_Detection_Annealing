//! Modul welches einen oder mehrere Sampler anbietet
//! Definiert weiterhin eine Schnittstelle für Sampler und Modelle

//imports
pub use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArray2};
pub use numpy::ndarray::prelude::*;
use numpy::pyo3::prelude::*;
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

type Oarray1 = Option<Array1<f32>>;
type Oarray2 = Option<Array2<f32>>;
type Osize = Option<usize>;

/// Ein Boltzmann Sampler, kann aus mehreren private oder public fields
/// bestehen
#[derive(Debug)]
#[pyclass(frozen)]
pub struct BoltzmannSampler {
    seed: u64,
    qubo: Array2<f32>,
    temperature_schedule: Array1<f32>,
    length: usize,
    num_samples: usize,
    num_total_anneals: usize,
    samples_per_anneal: usize,
}

// Rust-only-methods
impl BoltzmannSampler {

    pub fn new(
        qubo_matrix: &Array2<f32>,
        num_samples: usize,
        temperatures: Oarray1,
        num_total_anneals_per_sampling: Osize,
        seed: u64,
    ) -> Self {
        assert_eq!(qubo_matrix.nrows(), qubo_matrix.ncols());
        if let Some(ref temperature_schedule) = temperatures {
            assert!(temperature_schedule.is_empty())
        };
        let temperature_schedule: Array1<f32> = match temperatures {
            Some(temperature_schedule) => temperature_schedule,
            // annealing schedule according to paper, collecting statistics (i.e. drawing actual samples) is then done at the last temp of the schedule
            None => array![20.0, 20.0, 15.0, 15.0, 12.0, 12.0, 10.0, 10.0, 10.0, 10.0],
        };

        let num_total_anneals: usize = num_total_anneals_per_sampling.unwrap_or(1);
        assert!(num_total_anneals <= num_samples);
        assert!(num_total_anneals > 0);

        let samples_per_anneal =
            ((num_samples as f32) / (num_total_anneals as f32)).ceil() as usize;

        Self {
            seed,
            qubo: qubo_matrix.clone(),
            temperature_schedule,
            length: qubo_matrix.nrows(),
            num_samples,
            num_total_anneals,
            samples_per_anneal,
        }
    }

    // the main annealing function
    fn annealing(
        &self,
        mut current_state: Array1<f32>,
        list_of_lists_of_random_tuples: Array2<(usize, f32)>,
    ) -> Array1<f32> {
        let temp_schedule = self.temperature_schedule.clone();
        for (i, temperature) in temp_schedule.iter().enumerate() {
            current_state = self.one_step_annealing(
                &current_state,
                *temperature,
                &list_of_lists_of_random_tuples.slice(s![i, ..]),
            );
        }
        current_state
    }

    // Generate an initial state array of given length with random 1s and 0s
    fn generate_initial_state(&self, rng: &mut ChaCha8Rng) -> Array1<f32> {
        let mut state = Array1::zeros(self.length);
        state.mapv_inplace(|_x: f32| {
            if rng.gen_range(0.0..1.0) >= 0.5 {
                1.0
            } else {
                0.0
            }
        });
        state
    }

    fn one_step_annealing(
        &self,
        state: &Array1<f32>,
        temperature: f32,
        random_tuples_list: &ArrayView1<(usize, f32)>,
    ) -> Array1<f32> {
        let mut current_state = state.clone();
        // Theoretically, another parallelization might be possible here
        for i in 0..self.length {
            self.perturb_state(&mut current_state, temperature, random_tuples_list[i]);
        }
        current_state
    }

    // Define a function to generate a new state by perturbing the current state
    fn perturb_state(
        &self,
        current_state: &mut Array1<f32>,
        temperature: f32,
        random_num_tuple: (usize, f32),
    ) {
        let (rand_index, random_value) = random_num_tuple;
        // delta energy is the signed value of how much energy the unit contributes the the total system state if it's 1
        // -> will usually be negative
        let delta_energy = self.get_energy_delta(current_state, rand_index);
        let p_on = 1.0 / (1.0 + (delta_energy / temperature).exp());
        if random_value < p_on {
            current_state[rand_index] = 1.0;
        } else {
            current_state[rand_index] = 0.0;
        }
    }

    fn get_energy_delta(&self, state: &Array1<f32>, index: usize) -> f32 {
        let mut delta_energy = self.qubo.column(index).dot(state) + self.qubo.row(index).dot(state);
        if state[index] == 1.0 {
            delta_energy -= self.qubo[[index, index]];
        } else {
            delta_energy += self.qubo[[index, index]];
        }
        delta_energy
    }

    // annealing run at temperature 10.0 to "collect statistics"
    fn quick_sample(
        &self,
        annealing_result: &Array1<f32>,
        random_nums: ArrayView1<(usize, f32)>,
    ) -> Array1<f32> {
        // -> ArrayView<f32, Ix1> {
        let temperature = self.temperature_schedule[self.temperature_schedule.len() - 1];
        self.one_step_annealing(annealing_result, temperature, &random_nums)
    }

    fn get_list_of_random_tuples_lists(
        &self,
        rng: &mut ChaCha8Rng,
        list_length: usize,
    ) -> Array2<(usize, f32)> {
        Array2::from_shape_fn((list_length, self.length), |_| {
            (rng.gen_range(0..self.length), rng.gen())
        })
    }

}

// methods to be accessed by Rust and Python
#[pymethods]
impl BoltzmannSampler {

    #[new]
    pub fn python_new(
        qubo_matrix: &PyArray2<f32>,
        num_samples: usize,
        seed: u64,
        num_total_anneals_per_sampling: Option<usize>,
        temperatures: Option<&PyArray1<f32>>,
    ) -> Self {
        let qubo: Array2<f32> = qubo_matrix.to_owned_array();
        let length: usize = qubo.nrows();
        assert_eq!(qubo.nrows(), qubo.ncols());
        if let Some(ref temperature_schedule) = temperatures {
            assert!(temperature_schedule.is_empty())
        };
        let temperature_schedule: Array1<f32> = match temperatures {
            Some(temperature_schedule) => temperature_schedule.to_owned_array(),
            // annealing schedule according to paper, collecting statistics (i.e. drawing actual samples) is then done at the last temp of the schedule
            // None => array![20.0, 15.0, 10.0, 7.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            None => array![20.0, 20.0, 15.0, 15.0, 12.0, 12.0, 10.0, 10.0, 10.0, 10.0],
        };

        let num_total_anneals: usize = num_total_anneals_per_sampling.unwrap_or(1);
        assert!(num_total_anneals <= num_samples);
        assert!(num_total_anneals > 0);

        let samples_per_anneal =
            ((num_samples as f32) / (num_total_anneals as f32)).ceil() as usize;

        Self {
            seed,
            qubo,
            temperature_schedule,
            length,
            num_samples,
            num_total_anneals,
            samples_per_anneal,
        }
    }

    pub fn draw_samples<'py>(&self, py: Python<'py>, qubo: Option<&PyArray2<f32>>) -> (Self, &'py PyArray2<f32>) {
        let qubo_arg: Oarray2 = match qubo {
            Some(qubo_matrix) => {let binding = Some(qubo_matrix.to_owned_array()); binding},
            None => None,
        };

        let (sampler_new, result_array) = Sampler::sample(self, &qubo_arg);
        (sampler_new, result_array.into_pyarray(py))
    }
}

/// Definiert welche Operationen ein beliebiger sampler nach aussen hin anbietet
/// und was diese zurückliefern
pub trait Sampler {
    /// sampled mit Anzahl samples
    fn sample(&self, qubo: &Oarray2) -> (Self, Array2<f32>)
    where
        Self: Sized;
}

/// Implementierung von Sampler für den BoltzmannSampler
impl Sampler for BoltzmannSampler {
    // returns ´self.num_samples´ samples, annealing self.num_total_anneals in total (unless less samples are required)
    // Annealing only different amounts of times for the clamped case and the unclamped case as in Paper is possible
    // (Also, we don't want to use noisy sampling and fixed weight steps, right?)
    fn sample(&self, qubo: &Oarray2) -> (Self, Array2<f32>) {
        // set new qubo if necessary
        let sampler = if let Some(qubo) = qubo {
            Self {
                seed: self.seed,
                qubo: qubo.clone(),
                temperature_schedule: self.temperature_schedule.clone(),
                length: self.length,
                num_samples: self.num_samples,
                num_total_anneals: self.num_total_anneals,
                samples_per_anneal: self.samples_per_anneal,
            }
        } else {
            Self {
                seed: self.seed,
                qubo: self.qubo.clone(),
                temperature_schedule: self.temperature_schedule.clone(),
                length: self.length,
                num_samples: self.num_samples,
                num_total_anneals: self.num_total_anneals,
                samples_per_anneal: self.samples_per_anneal,
            }
        };
        // create random number generator
        let mut rng = ChaCha8Rng::seed_from_u64(sampler.seed);
        // create list to store the samples that are to be drawn and returned later
        let mut sample_list: Array2<f32> = Array2::zeros((sampler.num_samples, sampler.length));

        // run all anneals but the last one (last one might have less samples)
        for anneal in 0..(sampler.num_total_anneals - 1) {
            let initial_state: Array1<f32> = sampler.generate_initial_state(&mut rng);

            // get random numbers for annealing
            let list_of_lists_of_random_tuples_anneal: Array2<(usize, f32)> = sampler
                .get_list_of_random_tuples_lists(&mut rng, sampler.temperature_schedule.len());

            // anneal to "equilibrium"
            let annealing_result_state: Array1<f32> =
                sampler.annealing(initial_state, list_of_lists_of_random_tuples_anneal);

            // get random numbers for drawing samples
            let list_of_lists_of_random_tuples_samples: Array2<(usize, f32)> =
                sampler.get_list_of_random_tuples_lists(&mut rng, sampler.samples_per_anneal);

            // draw samples based on annealing_result (in parallel)
            let samples_for_this_anneal_nested: Array1<Array1<f32>> = ndarray::Zip::from(
                list_of_lists_of_random_tuples_samples.rows(),
            )
            .par_map_collect(|random_tuples_list| {
                (sampler.quick_sample(&annealing_result_state, random_tuples_list)).to_owned()
            });
            let samples_for_this_anneal_flat: Array1<f32> = samples_for_this_anneal_nested
                .into_iter()
                .flatten()
                .collect();
            let samples_for_this_anneal: Array2<f32> = samples_for_this_anneal_flat
                .into_shape((sampler.samples_per_anneal, sampler.length))
                .unwrap();

            // enter samples for this anneal into sample_list
            let sample_num_start: usize = anneal * sampler.samples_per_anneal;
            let sample_num_end: usize = sample_num_start + sampler.samples_per_anneal;
            let mut row_slice: ArrayViewMut2<f32> =
                sample_list.slice_mut(s![sample_num_start..sample_num_end, ..]);
            row_slice.assign(&samples_for_this_anneal);
        }

        // last anneal (might be the only one if sampler.num_total_anneals == 1,
        // might be less samples than sampler.samples_per_anneal otherwise, namely if sampler.num_samples % sampler.samples per anneal != 0)
        let initial_state: Array1<f32> = sampler.generate_initial_state(&mut rng);
        // get random numbers for annealing
        let list_of_lists_of_random_tuples_anneal: Array2<(usize, f32)> =
            sampler.get_list_of_random_tuples_lists(&mut rng, sampler.temperature_schedule.len());

        // anneal to "equilibrium"
        let annealing_result_state: Array1<f32> =
            sampler.annealing(initial_state, list_of_lists_of_random_tuples_anneal);
        let num_samples_modulo = sampler.num_samples % sampler.samples_per_anneal;
        let num_samples_rest: usize = if num_samples_modulo != 0 {
            num_samples_modulo
        } else {
            sampler.samples_per_anneal
        };
        let num_samples_done: usize = sampler.num_samples - num_samples_rest;

        // get random numbers for drawing samples
        let list_of_lists_of_random_tuples_samples: Array2<(usize, f32)> =
            sampler.get_list_of_random_tuples_lists(&mut rng, num_samples_rest);

        // draw last samples based on annealing_result (in parallel)
        let samples_for_this_anneal_nested: Array1<Array1<f32>> = ndarray::Zip::from(
            list_of_lists_of_random_tuples_samples.rows(),
        )
        .par_map_collect(|random_tuples_list| {
            (sampler.quick_sample(&annealing_result_state, random_tuples_list)).to_owned()
        });
        let samples_for_this_anneal_flat: Array1<f32> = samples_for_this_anneal_nested
            .into_iter()
            .flatten()
            .collect();
        let samples_for_this_anneal: Array2<f32> = samples_for_this_anneal_flat
            .into_shape((num_samples_rest, sampler.length))
            .unwrap();

        // enter last samples into sample_list
        let mut row_slice: ArrayViewMut2<f32> =
            sample_list.slice_mut(s![num_samples_done..sampler.num_samples, ..]);
        row_slice.assign(&samples_for_this_anneal);

        // return Sampler and list with samples
        (sampler, sample_list)
    }
}

#[pymodule]
fn boltzmann_sampler(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<BoltzmannSampler>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    // tests methods
    #[test]
    fn test_sample_generate_initial_state() {
        let qubo: Array2<f32> = array![[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]];
        let seed = 5;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let sampler = BoltzmannSampler::new(&qubo, 5, None, None, seed);
        let state = sampler.generate_initial_state(&mut rng);
        let desired_result: Array1<f32> = array![0.0, 1.0, 1.0];
        assert_eq!(state, desired_result);
        println!("Initial state: {state}.");
    }

    #[test]
    fn test_sample_get_energy_delta() {
        let qubo: Array2<f32> = array![[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]];
        let seed = 5;
        let sampler = BoltzmannSampler::new(&qubo, 5, None, None, seed);
        let state: Array1<f32> = array![1.0, 0.0, 1.0];
        let index0: usize = 1;
        let index1: usize = 2;
        let energy0 = sampler.get_energy_delta(&state, index0);
        let energy1 = sampler.get_energy_delta(&state, index1);
        // (1 * 0 + 0 * 4 + 1 * 5) + (1 * 2 + 0 * 4 + 1 * 0) + 4 = 11
        assert_eq!(energy0, 11.0);
        // (1 * 0 + 0 * 0 + 1 * 6) + (1 * 3 + 0 * 5 + 1 * 6) - 6 = 9
        assert_eq!(energy1, 9.0);
    }

    // tests sampler
    #[test]
    fn test_sample_anneal() {
        let now = Instant::now();
        let qubo: Array2<f32> = array![[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]];
        let sampler = BoltzmannSampler::new(&qubo, 5, None, Some(5), 5);
        let (sampler, result) = sampler.sample(&None);
        assert!(result.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result.len(), 15);
        println!("{}", result);
        let qubo: Array2<f32> = array![[7., 8., 9.], [0., 10., 11.], [0., 0., 12.]];
        let (_sampler, result2) = sampler.sample(&Some(qubo));
        assert!(result2.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result2.len(), 15);
        println!("{}", result2);
        let end = now.elapsed().as_nanos();
        println!("test_sample_anneal took {end} nanos.");
    }

    #[test]
    fn test_sample_quick() {
        let now = Instant::now();
        let qubo: Array2<f32> = array![[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]];
        let sampler = BoltzmannSampler::new(&qubo, 5, None, None, 5);
        let (sampler, result) = sampler.sample(&None);
        assert!(result.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result.len(), 15);
        println!("{}", result);
        let qubo: Array2<f32> = array![[7., 8., 9.], [0., 10., 11.], [0., 0., 12.]];
        let (_sampler, result2) = sampler.sample(&Some(qubo));
        assert!(result2.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result2.len(), 15);
        println!("{}", result2);
        let end = now.elapsed().as_nanos();
        println!("test_sample_quick took {end} nanos.");
    }

    #[test]
    fn test_sample_semi_quick() {
        let now = Instant::now();
        let qubo: Array2<f32> = array![[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]];
        let sampler = BoltzmannSampler::new(&qubo, 5, None, Some(2), 5);
        let (sampler, result) = sampler.sample(&None);
        assert!(result.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result.len(), 15);
        println!("{}", result);
        let qubo: Array2<f32> = array![[7., 8., 9.], [0., 10., 11.], [0., 0., 12.]];
        let (_sampler, result2) = sampler.sample(&Some(qubo));
        assert!(result2.iter().all(|x| (*x == 1.0 || *x == 0.0)));
        assert_eq!(result2.len(), 15);
        println!("{}", result2);
        let end = now.elapsed().as_nanos();
        println!("test_sample_semi_quick took {end} nanos.");
    }
}

// Einfach nur zum veranschaulichen von private
// #[derive(Debug)]
// pub struct MultiSampler {
//     mult: u8,
// }
//
// impl MultiSampler {
//     // erzeugt einen neuen Multisampler, bietet nutzern die Möglichkeit
//     // die private variable bei erzeugung zu setzen
//     pub fn new(mult: u8) -> Self {
//         MultiSampler { mult }
//     }
// }
//
// // Diese Typkonversion ist allgemein besser
// impl Sampler for MultiSampler {
//     fn sample(&self, samples: usize) -> f32 {
//         3.14 * f32::from(self.mult * samples)
//     }
// }
