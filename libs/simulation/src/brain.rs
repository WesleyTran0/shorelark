use crate::*;

#[derive(Debug)]
pub struct Brain {
    pub(crate) nn: nn::Network,
}

impl Brain {
    pub fn random(rng: &mut dyn RngCore, eye: &Eye) -> Self {
        Self {
            nn: nn::Network::random(rng, &Self::topology(eye)),
        }
    }

    pub(crate) fn from_chromosome(chromosome: ga::Chromosome, eye: &Eye) -> Self {
        Self {
            nn: nn::Network::from_weights(&Self::topology(eye), chromosome),
        }
    }

    pub(crate) fn as_chromosome(&self) -> ga::Chromosome {
        self.nn.weights().into_iter().collect()
    }

    fn topology(eye: &Eye) -> [nn::LayerTopology; 3] {
        // The input Layer that takes in what the eyes see
        [
            nn::LayerTopology {
                neurons: eye.cells(),
            },
            // The hidden Layer(s) that mutate. Start with one with more nodes than input
            nn::LayerTopology {
                neurons: 2 * eye.cells(),
            },
            // The Output Layer; since the brain will control the bird's speed and rotation, we
            // need two numbers = two neurons
            nn::LayerTopology { neurons: 2 },
        ]
    }
}
