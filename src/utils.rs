#[inline(always)]
pub(super) fn get_qubits_number(size: usize) -> u32 {
    size.ilog2()
}

#[inline(always)]
pub(super) fn pos_fortran2c(qubits_number: u32, pos: usize) -> usize {
    qubits_number as usize - pos - 1
}

#[inline(always)]
pub(super) fn get_left_mask(pos: usize) -> usize {
    usize::MAX << pos
}

#[inline(always)]
pub(super) fn get_left_masks(positions: &[usize]) -> Vec<usize> {
    positions.iter().map(|pos| get_left_mask(*pos)).collect()
}

#[inline(always)]
pub(super) fn get_right_mask(pos: usize) -> usize {
    !get_left_mask(pos)
}

#[inline(always)]
pub(super) fn get_right_masks(positions: &[usize]) -> Vec<usize> {
    positions.iter().map(|pos| get_right_mask(*pos)).collect()
}

#[inline(always)]
pub(super) fn insert_zero_bit(number: usize, left_mask: usize, right_mask: usize) -> usize {
    ((number & left_mask) << 1) | (number & right_mask)
}

#[inline(always)]
pub(super) fn insert_zero_bits(
    mut number: usize,
    left_masks: &[usize],
    right_masks: &[usize],
) -> usize {
    for (left_mask, right_mask) in left_masks.iter().zip(right_masks) {
        number = insert_zero_bit(number, *left_mask, *right_mask);
    }
    number
}

#[inline(always)]
pub(super) fn get_stride(pos: usize) -> usize {
    1 << pos
}

#[inline(always)]
pub(super) fn get_strides(positions: &[usize]) -> Vec<usize> {
    positions.iter().map(|pos| get_stride(*pos)).collect()
}

#[inline(always)]
pub(super) fn get_task_size(batch_size: usize, threads_number: usize) -> usize {
    if batch_size % threads_number == 0 {
        batch_size / threads_number
    } else {
        batch_size / threads_number + 1
    }
}

#[inline(always)]
pub(super) fn dens_idx2state_idx(mut dens_idx: usize, strides: &[usize]) -> usize {
    let mut state_idx = 0;
    for stride in strides {
        state_idx += (dens_idx & 1) * stride;
        dens_idx >>= 1;
    }
    state_idx
}
