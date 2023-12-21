use ndarray::prelude::*;
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::weighted_alias::AliasableWeight;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use neural_net::csv_reader;
use neural_net::csv_reader::read_csv_to_vec;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;

struct NN {
    data: Array2<f32>,
    X_train: Array2<f32>,
    Y_train: Array1<f32>,
    train_shape: (usize, usize), // m,n
    model: model,
}

impl NN {
    fn new() -> NN {
        NN {
            data: Array2::default((0, 0)),
            X_train: Array2::default((0, 0)),
            Y_train: Array1::default(0),
            train_shape: (0, 0),
            model: model::new(),
        }
    }

    fn fill_train_data(&mut self, path: &str) {
        let mut data = read_csv_to_vec(path).unwrap();

        data.shuffle(&mut thread_rng());

        let inner_length = data[0].len();
        assert!(data.iter().all(|v| v.len() == inner_length));

        let flat: Vec<f32> = data.into_par_iter().flatten().collect();

        let cols = flat.len() / inner_length;

        self.data = Array2::from_shape_vec((cols, inner_length), flat)
            .unwrap()
            .t()
            .to_owned();

        let shape = self.data.shape();

        self.train_shape = (shape[0], shape[1]);
    }

    fn fill_XY_train(&mut self) {
        self.X_train = self.data.slice(s![1.., ..]).to_owned() / 255.0;
        self.Y_train = self.data.row(0).to_owned();
    }

    fn train_model(&mut self, alpha: f32, iterations: usize) {
        self.model.gradient_descent(
            &self.X_train,
            &self.Y_train,
            alpha,
            iterations,
            self.train_shape.0,
        )
    }
}

struct model {
    W1: Array2<f32>,
    b1: Array1<f32>,
    W2: Array2<f32>,
    b2: Array1<f32>,
}

impl model {
    fn new() -> model {
        model {
            W1: Array2::random((10, 784), Uniform::new(-0.5, 0.5)),
            b1: Array1::random(10, Uniform::new(-0.5, 0.5)),
            W2: Array2::random((10, 10), Uniform::new(-0.5, 0.5)),
            b2: Array1::random(10, Uniform::new(-0.5, 0.5)),
        }
    }
    fn relu(&self, Z: &mut Array2<f32>) {
        Z.par_mapv_inplace(|a| a.max(0.0));
    }

    fn softmax(&self, Z: &mut Array2<f32>) {
        Z.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut row| {
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.mapv(|a| ((a - max_val).exp())).sum();
                row.par_mapv_inplace(|a| ((a - max_val).exp()) / sum);
            });
    }

    fn forward_prop(
        &self,
        X: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let Z1 = self.W1.dot(X) + &self.b1.view().insert_axis(Axis(1));
        let mut A1 = Z1.clone();
        self.relu(&mut A1);
        let Z2 = self.W2.dot(&A1) + &self.b2.view().insert_axis(Axis(1));
        let mut A2 = Z2.clone();
        self.softmax(&mut A2);

        (Z1, A1, Z2, A2)
    }
    fn relu_deriv(&self, Z: &mut Array2<f32>) {
        Z.par_mapv_inplace(|a| if a > 0.0 { 1.0 } else { 0.0 });
    }

    fn one_hot(&self, Y: Array1<f32>) -> Array2<f32> {
        let num_classes = 10;
        let num_samples = Y.len();
        if num_samples == 0 {
            panic!("Y array is empty.");
        }
        let mut one_hot_y = Array2::zeros((num_samples, num_classes));
        for (i, &label) in Y.iter().enumerate() {
            let label_idx = label as usize;
            if label_idx < num_classes {
                one_hot_y[(i, label_idx)] = 1.0;
            } else {
                panic!("Label out of bounds: {}", label_idx);
            }
        }
        one_hot_y.reversed_axes()
    }

    fn backward_prop(
        &self,
        mut Z1: Array2<f32>,
        A1: &Array2<f32>,
        Z2: &Array2<f32>,
        A2: &Array2<f32>,
        X: &Array2<f32>,
        Y: &Array1<f32>,
        m: usize,
    ) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
        let one_hot_y = self.one_hot(Y.clone());
        let dZ2 = A2 - one_hot_y;
        let dW2 = (1.0 / m as f32) * dZ2.dot(&A1.t());
        let db2 = (1.0 / m as f32) * dZ2.sum_axis(ndarray::Axis(1));

        self.relu_deriv(&mut Z1);

        let dZ1 = &self.W2.t().dot(&dZ2) * Z1;
        let dW1 = (1.0 / m as f32) * dZ1.dot(&X.t());
        let db1 = (1.0 / m as f32) * dZ1.sum_axis(ndarray::Axis(1));

        (dW1, db1, dW2, db2)
    }

    fn update_params(
        &mut self,
        dW1: Array2<f32>,
        db1: Array1<f32>,
        dW2: Array2<f32>,
        db2: Array1<f32>,
        alpha: f32,
    ) {
        self.W1 = &self.W1 - alpha * dW1;
        self.b1 = &self.b1 - alpha * db1;
        self.W2 = &self.W2 - alpha * dW2;
        self.b2 = &self.b2 - alpha * db2;
    }

    fn get_predictions(&self, A2: &Array2<f32>) -> Array1<f32> {
        A2.axis_iter(Axis(1))
            .map(|column| {
                let (max_index, _) = column
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                max_index as f32
            })
            .collect::<Array1<f32>>()
    }

    fn get_accuracy(&self, predictions: &Array1<f32>, Y: &Array1<f32>) -> f32 {
        let correct_count = predictions
            .iter()
            .zip(Y.iter())
            .filter(|(&p, &y)| (p - y).abs() < f32::EPSILON)
            .count();
        correct_count as f32 / Y.len() as f32
    }

    fn gradient_descent(
        &mut self,
        X: &Array2<f32>,
        Y: &Array1<f32>,
        alpha: f32,
        iterations: usize,
        m: usize,
    ) {
        for i in 0..iterations {
            let (Z1, A1, Z2, A2) = self.forward_prop(X);
            let (dW1, db1, dW2, db2) = self.backward_prop(Z1, &A1, &Z2, &A2, &X, &Y, m);

            self.update_params(dW1, db1, dW2, db2, alpha);

            if i % 10 == 0 {
                println!("Iteration: {}", i);
                let predictions = self.get_predictions(&A2);
                let accuracy = self.get_accuracy(&predictions, &Y);
                println!("Accuracy: {}", accuracy);
            }
        }
    }
}
fn main() {
    let time = Instant::now();

    let mut NN = NN::new();
    NN.fill_train_data("src/train.csv");
    NN.fill_XY_train();
    NN.train_model(0.01, 800);

    println!("{:?}", time.elapsed());
}
