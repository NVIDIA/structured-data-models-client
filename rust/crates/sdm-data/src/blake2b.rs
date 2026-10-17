// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BLAKE2b (RFC 7693), unkeyed, of a digest length the caller chooses: the
//! hash the reference's Postgres sampler maps a seed with
//! (`hashlib.blake2b(text, digest_size=8)`). The digest length is a
//! parameter of the hash, not a truncation of a longer digest, so the
//! eight-byte digest is computed as such. Ported from the RFC's reference
//! code (its Appendix C) rather than taken as a crate, since the shipped
//! tree needs this one use (`phase-1/dependencies.md`'s rule; T2.3).

/// The initialization vector, SHA-512's.
const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

/// The message schedule of each of the twelve rounds (rounds 10 and 11
/// repeat 0 and 1).
const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

const BLOCK: usize = 128;

fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// The compression of one block, `bytes` the message length so far
/// (counting this block), `last` whether it is the final block.
fn compress(h: &mut [u64; 8], block: &[u8; BLOCK], bytes: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        let mut le = [0u8; 8];
        le.copy_from_slice(&block[i * 8..i * 8 + 8]);
        *word = u64::from_le_bytes(le);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the counter's low and high words, as the RFC splits it"
    )]
    {
        v[12] ^= bytes as u64;
        v[13] ^= (bytes >> 64) as u64;
    }
    if last {
        v[14] = !v[14];
    }
    for round in 0..12 {
        let s = &SIGMA[round % 10];
        mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// The unkeyed BLAKE2b digest of `message`, `length` bytes long (1 to 64).
fn digest(message: &[u8], length: usize) -> Vec<u8> {
    debug_assert!((1..=64).contains(&length));
    let mut h = IV;
    // The parameter block's first word: the digest length, no key, fanout
    // and depth 1.
    h[0] ^= 0x0101_0000 ^ length as u64;
    let mut done: u128 = 0;
    let mut rest = message;
    while rest.len() > BLOCK {
        let (block, tail) = rest.split_at(BLOCK);
        done += BLOCK as u128;
        let mut full = [0u8; BLOCK];
        full.copy_from_slice(block);
        compress(&mut h, &full, done, false);
        rest = tail;
    }
    let mut last = [0u8; BLOCK];
    last[..rest.len()].copy_from_slice(rest);
    done += rest.len() as u128;
    compress(&mut h, &last, done, true);
    h.iter()
        .flat_map(|word| word.to_le_bytes())
        .take(length)
        .collect()
}

/// The eight-byte BLAKE2b digest of `message`.
pub(crate) fn digest8(message: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest(message, 8));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn rfc_7693_appendix_a() {
        assert_eq!(
            hex(&digest(b"abc", 64)),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
    }

    #[test]
    fn python_hashlib_at_eight_bytes() {
        // hashlib.blake2b(b'42', digest_size=8).hexdigest()
        assert_eq!(hex(&digest8(b"42")), "57b43cf02666687a");
        // The empty message, exactly one block, and one past a block,
        // which takes two compressions: hashlib.blake2b(m, digest_size=8)
        // of b'', b'x' * 128 and b'x' * 200.
        assert_eq!(hex(&digest8(b"")), "e4a6a0577479b2b4");
        assert_eq!(hex(&digest8(&[b'x'; 128])), "901ce374e8e12402");
        assert_eq!(hex(&digest8(&[b'x'; 200])), "56304c0e1b76a18d");
    }
}
