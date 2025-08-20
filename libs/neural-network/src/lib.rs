use std::iter::once;

use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers_pair| Layer::random(rng, layers_pair[0].neurons, layers_pair[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn weights(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .copied()
            .collect()
    }

    pub fn from_weights(layers: &[LayerTopology], weights: impl IntoIterator<Item = f32>) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| Layer::from_weights(layers[0].neurons, layers[1].neurons, &mut weights))
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();

        Self { neurons }
    }

    fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }
}

#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(&input, &weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }

    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let bias = rng.random_range(-1.0..=1.0);

        let weights = (0..input_size)
            .map(|_| rng.random_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }

    fn from_weights(input_size: usize, weights: &mut (dyn Iterator<Item = f32>)) -> Neuron {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..input_size)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Neuron tests
    #[test]
    fn random_neuron() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);

        assert_relative_eq!(neuron.bias, -0.6255188);
        assert_relative_eq!(
            neuron.weights.as_slice(),
            [0.67383933, 0.81812596, 0.26284885, 0.5238805].as_ref()
        );
    }

    #[test]
    fn propagate_neuron() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

        assert_relative_eq!(
            neuron.propagate(&[0.5, 1.0]),
            (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
        );
    }

    // Layer tests
    #[test]
    fn random_layer() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let layer = Layer::random(&mut rng, 3, 2);

        // test all elements of first neuron
        assert_relative_eq!(layer.neurons[0].bias, -0.6255188);
        assert_relative_eq!(
            layer.neurons[0].weights.as_slice(),
            [0.67383933, 0.81812596, 0.26284885].as_ref()
        );

        assert_relative_eq!(layer.neurons[1].bias, 0.5238805);
        assert_relative_eq!(
            layer.neurons[1].weights.as_slice(),
            [-0.5351684, 0.0693696, -0.7648182].as_ref()
        )
    }

    #[test]
    fn propagate_layer() {
        let layer = Layer {
            neurons: vec![
                Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8],
                },
                Neuron {
                    bias: 0.2,
                    weights: vec![0.2, -0.4],
                },
            ],
        };

        assert_relative_eq!(
            layer.propagate(vec!(-10.0, -10.0)).as_slice(),
            [0.0, 2.2].as_ref()
        );
        assert_relative_eq!(
            layer.propagate(vec!(0.7, 0.8)).as_slice(),
            [
                (-0.3 * 0.7) + (0.8 * 0.8) + 0.5,
                (0.2 * 0.7) + (-0.4 * 0.8) + 0.2
            ]
            .as_ref()
        );
    }

    // Tests for Network
    #[test]
    fn random_network() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let network = Network::random(
            &mut rng,
            &[
                LayerTopology { neurons: 4 },
                LayerTopology { neurons: 3 },
                LayerTopology { neurons: 2 },
            ],
        );

        println!("{:?}", network);
        assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
        assert_relative_eq!(network.layers[0].neurons[1].bias, -0.5351684);
        assert_relative_eq!(network.layers[0].neurons[2].bias, -0.19277143);

        assert_relative_eq!(network.layers[1].neurons[0].bias, -0.4766221);
        assert_relative_eq!(network.layers[1].neurons[1].bias, 0.35662675);

        assert_relative_eq!(
            network.layers[0].neurons[0].weights.as_slice(),
            [0.67383933, 0.81812596, 0.26284885, 0.5238805].as_ref()
        );
        assert_relative_eq!(
            network.layers[0].neurons[1].weights.as_slice(),
            [0.069369555, -0.7648182, -0.102499485, -0.48879623].as_ref()
        );
        assert_relative_eq!(
            network.layers[0].neurons[2].weights.as_slice(),
            [-0.8020501, 0.27546048, -0.98680043, 0.4452355].as_ref()
        );

        assert_relative_eq!(
            network.layers[1].neurons[0].weights.as_slice(),
            [-0.89078736, -0.36127806, -0.14956546].as_ref()
        );
        assert_relative_eq!(
            network.layers[1].neurons[1].weights.as_slice(),
            [-0.8566594, 0.3330984, 0.11767411].as_ref(),
        );
    }

    #[test]
    fn propagate_network() {
        let network = Network {
            layers: vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: 0.5,
                            weights: vec![-0.3, 0.8],
                        },
                        Neuron {
                            bias: 0.2,
                            weights: vec![0.2, -0.4],
                        },
                    ],
                },
                Layer {
                    neurons: vec![Neuron {
                        bias: 0.4,
                        weights: vec![-0.7, 0.9],
                    }],
                },
            ],
        };

        assert_relative_eq!(
            network.propagate(vec!(10.0, 10.0)).as_slice(),
            [0.0].as_ref()
        );
        assert_relative_eq!(
            network.propagate(vec!(-0.7, -0.8)).as_slice(),
            [(((-0.3 * -0.7) + (0.8 * -0.8) + 0.5) * -0.7)
                + (((0.2 * -0.7) + (-0.4 * -0.8) + 0.2) * 0.9)
                + 0.4]
            .as_ref()
        );
    }

    #[test]
    fn weights() {
        let network = Network {
            layers: vec![
                Layer {
                    neurons: vec![Neuron {
                        bias: 0.1,
                        weights: vec![0.2, 0.3, 0.4],
                    }],
                },
                Layer {
                    neurons: vec![Neuron {
                        bias: 0.5,
                        weights: vec![0.6, 0.7, 0.8],
                    }],
                },
            ],
        };

        let actual = network.weights();
        let expected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn from_weights() {
        let layers = &[LayerTopology { neurons: 3 }, LayerTopology { neurons: 2 }];

        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let network = Network::from_weights(layers, weights.clone());
        let actual: Vec<_> = network.weights().into_iter().collect();

        assert_relative_eq!(actual.as_slice(), weights.as_slice());
    }
}
