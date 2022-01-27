use itertools::Itertools;

//#[derive(Debug, Clone, PartialEq)]
//pub struct BinGrid {
//    mu_bins: Vec<MuBin>,
//    v_bin_size: f64,
//    empty_mu_bin: MuBin,
//}
//
//impl BinGrid {
//    pub fn new(mu_bins: Vec<MuBin>, v_bin_size: f64) -> Self {
//        Self {
//            mu_bins,
//            v_bin_size,
//            empty_mu_bin: MuBin { v_bins: Vec::new() },
//        }
//    }
//
//    pub fn mu_bins(&self) -> &Vec<MuBin> {
//        &self.mu_bins
//    }
//
//    pub fn grad_mu(&self) -> Self {
//        let mut new_mu_bins = Vec::with_capacity(self.mu_bins.len());
//        for (i_mu, _mu, _bin) in self.iter_mu() {
//            let prev_bin = self.get_mu_bin(i_mu - 1);
//            let next_bin = self.get_mu_bin(i_mu + 1);
//            let capacity = usize::max(prev_bin.v_bins.len(), next_bin.v_bins.len()) + 1;
//            let mut mu_bin = MuBin { v_bins: Vec::with_capacity(capacity) };
//            for v_bins_pair in prev_bin
//                .v_bins
//                .iter()
//                .copied()
//                .zip_longest(next_bin.v_bins.iter().copied())
//            {
//                let prev_v_bin = v_bins_pair
//                    .clone()
//                    .left()
//                    .unwrap_or_else(|| VBin { f_value: 0.0 });
//                let next_v_bin = v_bins_pair
//                    .clone()
//                    .right()
//                    .unwrap_or_else(|| VBin { f_value: 0.0 });
//                let d_mu = 2.0 * self.v_bin_size;
//                let grad = next_v_bin.f_value - prev_v_bin.f_value / d_mu;
//                mu_bin.v_bins.push(VBin { f_value: grad });
//            }
//            new_mu_bins.push(mu_bin);
//        }
//        Self::new(new_mu_bins, self.v_bin_size)
//    }
//
//    pub fn grad_v(&self) -> Self {
//        let mut new_mu_bins = Vec::with_capacity(self.mu_bins.len());
//        for (i_mu, _mu, _bin) in self.iter_mu() {
//            let prev_bin = self.get_mu_bin(i_mu - 1);
//            let next_bin = self.get_mu_bin(i_mu + 1);
//            let capacity = usize::max(prev_bin.v_bins.len(), next_bin.v_bins.len()) + 1;
//            let mut mu_bin = MuBin { v_bins: Vec::with_capacity(capacity) };
//            for v_bins_pair in prev_bin
//                .v_bins
//                .iter()
//                .copied()
//                .zip_longest(next_bin.v_bins.iter().copied())
//            {
//                let prev_v_bin = v_bins_pair
//                    .clone()
//                    .left()
//                    .unwrap_or_else(|| VBin { f_value: 0.0 });
//                let next_v_bin = v_bins_pair
//                    .clone()
//                    .right()
//                    .unwrap_or_else(|| VBin { f_value: 0.0 });
//                let d_mu = 2.0 * self.v_bin_size;
//                let grad = next_v_bin.f_value - prev_v_bin.f_value / d_mu;
//                mu_bin.v_bins.push(VBin { f_value: grad });
//            }
//            new_mu_bins.push(mu_bin);
//        }
//        Self::new(new_mu_bins, self.v_bin_size)
//    }
//
//    pub fn iter_mu(&self) -> impl Iterator<Item = (isize, f64, &MuBin)> {
//        self.mu_bins.iter().enumerate().map(|(i_mu, bin)| {
//            let i_mu = i_mu as isize;
//            let mu = self.i_to_mu(i_mu);
//            (i_mu, mu, bin)
//        })
//    }
//
//    pub fn get_mu_bin(&self, index: isize) -> &MuBin {
//        if index < 0 {
//            return &self.empty_mu_bin;
//        }
//        self.mu_bins
//            .get(index as usize)
//            .unwrap_or(&self.empty_mu_bin)
//    }
//
//    pub fn i_to_mu(&self, index: isize) -> f64 {
//        (index as f64 + 0.5) / self.mu_bins.len() as f64
//    }
//}
//
//#[derive(Debug, Clone, PartialEq)]
//pub struct MuBin {
//    pub v_bins: Vec<VBin>,
//}
//
//#[derive(Debug, Copy, Clone, PartialEq)]
//pub struct VBin {
//    pub f_value: f64,
//}

fn to_index(x: f64, min: f64, max: f64, num_bins: usize) -> Option<usize> {
    let offset = x - min;
    let range = max - min;
    if num_bins == 0 || offset < 0.0 || offset >= range {
        return None;
    }
    let proportion = range / offset;
    Some((proportion * num_bins as f64).floor() as usize)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinGrid {
    pub bins: Vec<Vec<f64>>,
    pub min_mu: f64,
    pub max_mu: f64,
    pub min_v: f64,
    pub max_v: f64,
}

impl BinGrid {
    pub fn get(&self, mu: f64, v: f64) -> Option<f64> {
        let mu_bin = to_index(mu, self.min_mu, self.max_mu, self.bins.len())?;
        let v_bin = to_index(v, self.min_v, self.max_v, self.bins[0].len())?;
        self.iget(mu_bin, v_bin)
    }

    pub fn iget(&self, mu_bin: usize, v_bin: usize) -> Option<f64> {
        Some(*self.bins.get(mu_bin)?.get(v_bin)?)
    }

    pub fn iget_mean(&self, mu_bin: usize, v_bin: usize, mu_size: usize, v_size: usize) -> f64 {
        let mut sum = 0.0;
        for off_mu in 0..mu_size {
            for off_v in 0..v_size {
                sum += self.iget(mu_bin + off_mu, v_bin + off_v).unwrap_or(0.0);
            }
        }
        sum / (mu_size * v_size) as f64
    }

    pub fn mu_from_index(&self, mu_bin: usize) -> f64 {
        let num_mu_bins = self.bins.len();
        let quotient = (mu_bin as f64 + 0.5) / num_mu_bins as f64;
        quotient * (self.max_mu - self.min_mu) + self.min_mu
    }

    pub fn v_from_index(&self, v_bin: usize) -> f64 {
        let num_v_bins = self.bins[0].len();
        let quotient = (v_bin as f64 + 0.5) / num_v_bins as f64;
        quotient * (self.max_v - self.min_v) + self.min_v
    }

    pub fn grad_mu(&self) -> Self {
        let mut new_self = self.clone();
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                let prev_f = self.iget(mu_bin - 1, v_bin).unwrap_or(0.0);
                let next_f = self.iget(mu_bin + 1, v_bin).unwrap_or(0.0);
                let d_mu = (self.max_mu - self.min_mu) / num_mu_bins as f64 * 2.0;
                let grad = (next_f - prev_f) / d_mu;
                new_self.bins[mu_bin][v_bin] = grad;
            }
        }
        new_self
    }

    pub fn grad_v(&self) -> Self {
        let mut new_self = self.clone();
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                let prev_f = self.iget(mu_bin, v_bin - 1).unwrap_or(0.0);
                let next_f = self.iget(mu_bin, v_bin + 1).unwrap_or(0.0);
                let d_v = (self.max_v - self.min_v) / num_v_bins as f64 * 2.0;
                let grad = (next_f - prev_f) / d_v;
                new_self.bins[mu_bin][v_bin] = grad;
            }
        }
        new_self
    }

    pub fn map_in_place<F: FnMut(f64, f64) -> f64>(&mut self, func: &mut F) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            let mu = self.mu_from_index(mu_bin);
            for v_bin in 0..num_v_bins {
                let v = self.v_from_index(v_bin);
                self.bins[mu_bin][v_bin] = func(mu, v);
            }
        }
    }

    pub fn map1_in_place<F: FnMut(f64) -> f64>(&mut self, func: &mut F) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                let old = self.bins[mu_bin][v_bin];
                self.bins[mu_bin][v_bin] = func(old);
            }
        }
    }

    pub fn map1<F: FnMut(f64) -> f64>(&self, func: &mut F) -> Self {
        let mut new_self = self.clone();
        new_self.map1_in_place(func);
        new_self
    }

    pub fn times_v_in_place(&mut self) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                self.bins[mu_bin][v_bin] *= self.v_from_index(v_bin);
            }
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.bins.iter().map(|x| x.iter().copied()).flatten()
    }
}

impl std::fmt::Display for BinGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut max = f64::MIN;
        let mut min = f64::MAX;

        for mu_bin in self.bins.iter() {
            for value in mu_bin.iter().copied() {
                max = max.max(value);
                min = min.min(value);
            }
        }

        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        let scale = 4;
        for mu_bin in (0..num_mu_bins).step_by(scale) {
            for v_bin in (0..num_v_bins).step_by(scale) {
                let value = self.iget_mean(mu_bin, v_bin, scale, scale);
                let value_std = value / (max - min) - min;
                let value_256 = (value_std * 255.0).floor() as u8;
                write!(f, "\x1b[48;2;{};{};{}m  ", value_256, value_256, value_256)?;
            }
            writeln!(f, "\x1b[0m")?;
        }
        writeln!(f, "max {:.5}, min {:.5}", max, min)
    }
}

impl std::ops::AddAssign<&BinGrid> for BinGrid {
    fn add_assign(&mut self, rhs: &Self) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                self.bins[mu_bin][v_bin] += rhs.bins[mu_bin][v_bin];
            }
        }
    }
}

impl std::ops::SubAssign<&BinGrid> for BinGrid {
    fn sub_assign(&mut self, rhs: &Self) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                self.bins[mu_bin][v_bin] -= rhs.bins[mu_bin][v_bin];
            }
        }
    }
}

impl std::ops::MulAssign<f64> for BinGrid {
    fn mul_assign(&mut self, rhs: f64) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                self.bins[mu_bin][v_bin] *= rhs;
            }
        }
    }
}

impl std::ops::MulAssign<&BinGrid> for BinGrid {
    fn mul_assign(&mut self, rhs: &Self) {
        let num_mu_bins = self.bins.len();
        let num_v_bins = self.bins[0].len();
        for mu_bin in 0..num_mu_bins {
            for v_bin in 0..num_v_bins {
                self.bins[mu_bin][v_bin] *= rhs.bins[mu_bin][v_bin];
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Simulation<Q> {
    bins: BinGrid,
    prev_bins: BinGrid,
    t: f64,
    prev_t: f64,
    q: Q,
}

impl<Q> Simulation<Q>
where
    Q: FnMut(f64) -> f64,
{
    pub fn new<F>(
        min_mu: f64,
        max_mu: f64,
        min_v: f64,
        max_v: f64,
        num_mu_bins: usize,
        num_v_bins: usize,
        f0: &mut F,
        q: Q,
    ) -> Self
    where
        F: FnMut(f64, f64) -> f64,
    {
        let t = 0.0;
        let mut bins = BinGrid {
            bins: vec![vec![0.0; num_v_bins]; num_mu_bins],
            min_mu,
            max_mu,
            min_v,
            max_v,
        };
        bins.map_in_place(&mut |mu, v| f0(mu, v));
        let prev_bins = bins.clone();
        Self { bins, t, q, prev_t: 0.0, prev_bins }
    }

    pub fn step(&mut self, dt: f64) {
        self.prev_bins = self.bins.clone();
        self.prev_t = self.t;
        let mut drift_term = self.bins.grad_mu();
        drift_term.times_v_in_place();
        drift_term *= -dt;
        let diffusion_term_rhs = self.bins.grad_v().grad_v();
        let mut diffusion_term = self.bins.map1(&mut self.q);
        diffusion_term *= &diffusion_term_rhs;
        diffusion_term *= dt;
        let mut df = drift_term;
        df += &diffusion_term;
        //println!("{}", diffusion_term);
        self.bins += &df;
        self.bins.map1_in_place(&mut |old| old.max(0.0));
        self.t += dt;
    }

    pub fn f(&self) -> &BinGrid {
        &self.bins
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn conservation_laws(&mut self) -> Option<(Stats, Stats, Stats)> {
        let dt = self.t - self.prev_t;
        if dt == 0.0 {
            return None;
        }

        let mut grad_t_term = self.bins.clone();
        grad_t_term -= &self.prev_bins;
        grad_t_term *= dt.recip();

        let mut grad_mu_term = self.bins.grad_mu();
        grad_mu_term.times_v_in_place();

        let q = self.bins.map1(&mut self.q);

        let mut grad_v_term = self.bins.grad_v().grad_v();
        grad_v_term *= &q;
        grad_v_term *= dt;

        let mut fancy_e = grad_t_term;
        fancy_e += &grad_mu_term;
        fancy_e += &grad_v_term;
        fancy_e *= dt;

        let q_recip = q.map1(&mut f64::recip);

        let mut law_1 = q_recip.clone();
        law_1 *= &fancy_e;
        
        let mut law_2 = q_recip.clone();
        law_2.times_v_in_place();
        law_2 *= &fancy_e;

        let mut law_3 = q_recip.clone();
        let mut law_3_num = law_3.clone();
        law_3_num.map_in_place(&mut |mu, v| self.t * v - mu);
        law_3 *= &law_3_num;
        law_3 *= &fancy_e;

        let stats_1 = iter_stats(law_1.iter());
        let stats_2 = iter_stats(law_2.iter());
        let stats_3 = iter_stats(law_3.iter());
        Some((stats_1, stats_2, stats_3))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Stats {
    mean: f64,
    stddev: f64,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.10} (std.dev = {:.10})", self.mean, self.stddev)
    }
}

fn iter_stats(it: impl Iterator<Item = f64>) -> Stats {
    let items: Vec<_> = it.map(|x| x.powi(2)).collect();
    let sum: f64 = items.iter().copied().sum();
    let mean = sum / items.len() as f64;
    let ess: f64 = items.iter().copied().map(|x| (x - mean).powi(2)).sum();
    let variance = ess / items.len() as f64;
    let stddev = variance.sqrt();
    Stats { mean, stddev }
}

fn main() {
    let mut f0 = |mu: f64, v: f64| {
        let dist1 = ((v - 1.0).powi(2) + 9.0 * (mu - 0.25).powi(2)).sqrt();
        let dist2 = ((v - 2.0).powi(2) + 9.0 * (mu - 0.2).powi(2)).sqrt();
        (-dist1.powi(2) * 30.0).exp() + (-dist2.powi(2) * 30.0).exp()
    };
    let q = |f: f64| (f - 1.0).exp();
    let mut sim = Simulation::new(0.0, 1.0, 0.0, 3.0, 128, 128, &mut f0, q);
    print!("\x1b[2J\x1b[3J");
    loop {
        print!("\x1b[H");
        println!("{}", sim.f());
        println!("t = {:.5}", sim.t());
        if let Some((stats_1, stats_2, stats_3)) = sim.conservation_laws() {
            println!("-> Conservation law 1: MSE = {}", stats_1);
            println!("-> Conservation law 2: MSE = {}", stats_2);
            println!("-> Conservation law 3: MSE = {}", stats_3);
        }
        //std::io::stdin().read_line(&mut String::new()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        sim.step(0.001);
    }
}
