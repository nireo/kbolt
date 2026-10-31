// Fixed-seed SplitMix64: reproducible inputs without an extra random dependency.
pub fn random_dna(len: usize) -> Vec<u8> {
    let mut state = 42_u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            value ^= value >> 31;
            b"ACGT"[(value & 3) as usize]
        })
        .collect()
}
