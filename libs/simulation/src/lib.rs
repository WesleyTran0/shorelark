use self::animal_individual::*;
pub use self::{animal::*, brain::*, eye::*, food::*, world::*};

mod animal;
mod animal_individual;
mod brain;
mod eye;
mod food;
mod world;

use lib_genetic_algorithm as ga;
use lib_neural_network as nn;
use nalgebra as na;
use rand::{Rng, RngCore};
use std::f32::consts::FRAC_PI_2;

/// How many steps have to occur before we run a generation algorithm, thus wiping the birds and
/// evolving
const GENERATION_LENGTH: usize = 2500;

/// Minimum speed of a bird to keep them moving
const SPEED_MIN: f32 = 0.001;
/// Maximum speed of a bird to prevent speeds reaching infinity
const SPEED_MAX: f32 = 0.005;
/// Determines how much the brain can affect speed in a single step
const SPEED_ACCEL: f32 = 0.2;
/// Determines how much the brain can affect rotation in a single step
const ROTATION_ACCEL: f32 = FRAC_PI_2;

pub struct Simulation {
    world: World,
    ga: ga::GeneticAlgorithm<ga::RouletteWheelSelection>,
    age: usize,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let ga = ga::GeneticAlgorithm::new(
            ga::RouletteWheelSelection,
            ga::UniformCrossover,
            ga::GaussianMutation::new(0.01, 0.3),
        );

        Self {
            world: World::random(rng),
            ga,
            age: 0,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Performs a single step - a single second of our simulation
    pub fn step(&mut self, rng: &mut dyn RngCore) -> Option<ga::Statistics> {
        self.process_collisions(rng);
        self.process_brains();
        self.process_movements();
        self.age += 1;

        if self.age > GENERATION_LENGTH {
            Some(self.evolve(rng))
        } else {
            None
        }
    }

    /// Fast Forwards to the next generation
    pub fn train(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        loop {
            if let Some(summary) = self.step(rng) {
                return summary;
            }
        }
    }

    pub fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.foods {
                let distance = na::distance(&animal.position, &food.position);

                if distance <= 0.013 {
                    food.position = rng.random();
                    animal.satiation += 1;
                }
            }
        }
    }

    pub fn process_brains(&mut self) {
        for animal in &mut self.world.animals {
            let vision =
                animal
                    .eye
                    .process_vision(animal.position, animal.rotation, &self.world.foods);

            let response = animal.brain.nn.propagate(vision);

            let speed = response[0].clamp(-SPEED_ACCEL, SPEED_ACCEL);
            let rotation = response[1].clamp(-ROTATION_ACCEL, ROTATION_ACCEL);

            animal.speed = (animal.speed + speed).clamp(SPEED_MIN, SPEED_MAX);
            animal.rotation = na::Rotation2::new(animal.rotation.angle() + rotation);
        }
    }

    pub fn process_movements(&mut self) {
        for animal in &mut self.world.animals {
            animal.position += animal.rotation * na::Vector2::new(0.0, animal.speed);

            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    pub fn evolve(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        self.age = 0;

        let current_population: Vec<_> = self
            .world
            .animals
            .iter()
            .map(AnimalIndividual::from_animal)
            .collect();

        let (evolved_population, stats) = self.ga.evolve(rng, &current_population);

        self.world.animals = evolved_population
            .into_iter()
            .map(|individual| individual.into_animal(rng))
            .collect();

        // Restarting food, mostly for visual purposes to spot when evolution happens
        for food in &mut self.world.foods {
            food.position = rng.random();
        }

        stats
    }
}
