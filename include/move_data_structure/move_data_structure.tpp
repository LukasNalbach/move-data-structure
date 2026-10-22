/**
 * part of LukasNalbach/move-data-structure
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_TPP

#include <move_data_structure/move_data_structure.hpp>

// ############################# INTERNAL METHODS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
std::array<uint8_t, move_data_structure<enc, mode>::num_rows> move_data_structure<enc, mode>::interleaved_widths(const row_widths_t& row_widths) const
{
    std::array<uint8_t, num_rows> w {};
    w[0] = omega_offs;
    w[1] = omega_idx;
    w[2] = omega_offs;
    for (uint8_t j = 0; j < row_widths.size(); j++) w[num_core_rows + j] = row_widths[j];
    return w;
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::build(mds_pairs& I, uint64_t n, mds_params params, row_widths_t row_widths,
    row_widths_t row_widths_inv, no_init_vector<uint40_t>* pi_out)
{
    if (!has_inverse && !row_widths_inv.empty()) throw std::invalid_argument("a move data structure without an inverse has no rows of output intervals");
    if (row_widths_inv.size() > max_extra_rows) throw std::invalid_argument("a move data structure has at most " + std::to_string(max_extra_rows) + " extra rows of output intervals");
    num_extra_rows = uint8_t(row_widths.size());
    header_value_width = params.header_value_width;
    construction(*this, I, n, params, row_widths, row_widths_inv, pi_out);
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::compute_layout(uint64_t l_max, const std::array<uint8_t, max_extra_rows>& extra_widths, uint64_t l_max_inv,
    uint8_t omega_offs_inv, uint8_t num_extra_rows_inv, const std::array<uint8_t, max_extra_rows>& extra_widths_inv)
{
    uint8_t omega_slot_fwd = std::bit_width(k_);
    uint8_t omega_slot_inv = has_inverse ? omega_slot_fwd : 0;

    for (;;) {
        // with an inverse, the header value of a block of the input intervals is a slot of an output interval
        input_blocks.compute_layout(n, k_, omega_p, omega_offs, omega_slot_fwd, has_inverse ? omega_slot_inv : header_value_width, l_max,
            num_extra_rows, extra_widths, records_store_lengths);
        uint8_t needed_fwd = std::bit_width(input_blocks.max_slot());
        uint8_t needed_inv = 0;

        if constexpr (has_inverse) {
            output_blocks.compute_layout(n, k_, omega_p, omega_offs_inv, omega_slot_inv, omega_slot_fwd, l_max_inv, num_extra_rows_inv, extra_widths_inv);
            needed_inv = std::bit_width(output_blocks.max_slot());
        }

        if (needed_fwd <= omega_slot_fwd && needed_inv <= omega_slot_inv) return;
        if (omega_slot_fwd >= 60 || omega_slot_inv >= 60) throw std::logic_error("move data structure: the slot widths do not converge");
        omega_slot_fwd = std::max(omega_slot_fwd, needed_fwd);
        omega_slot_inv = std::max(omega_slot_inv, needed_inv);
    }
}

// ############################# MISC PUBLIC METHODS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
uint64_t move_data_structure<enc, mode>::size_in_bytes() const
{
    uint64_t size = sizeof(*this);
    if constexpr (enc == POS) size += input_blocks.blocks_size_in_bytes();
    else size += data.size_in_bytes() + high_part_increments.size_in_bytes();
    if constexpr (has_inverse) size += output_blocks.blocks_size_in_bytes();
    return size;
}

template <move_pos_encoding_t enc, move_mode_t mode>
uint64_t move_data_structure<enc, mode>::blocks_size_with_rows(uint8_t bits, uint8_t header_value_width, uint64_t l_max) const
    requires(enc == POS && mode == UNI)
{
    move_data_structure probe;
    probe.n = n;
    probe.k_ = k_;
    probe.omega_p = omega_p;
    probe.omega_offs = omega_offs;
    probe.num_extra_rows = num_extra_rows + 1;
    probe.header_value_width = header_value_width;
    probe.records_store_lengths = true;
    std::array<uint8_t, max_extra_rows> extra_widths = input_blocks.row_widths();
    extra_widths[num_extra_rows] = bits;
    probe.compute_layout(l_max, extra_widths, 0, 0, 0, {});
    return (probe.input_blocks.data_blocks() + 1) * move_interval_blocks::bits_per_block / 8;
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::log_data_structures() const
{
    aligned_table log;
    log.add_row("p:", k_, [&](uint64_t i) { return p(i); });
    log.add_row("q:", k_, [&](uint64_t i) { return q(i); });
    log.add_row("idx:", k_, [&](uint64_t i) { return idx(i); });

    if constexpr (has_inverse) {
        log.add_row("output_interval_of_input:", k_, [&](uint64_t x) { return output_interval_of_input(x); });
        log.add_row("p_inv:", k_, [&](uint64_t y) { return p_inv(y); });
        log.add_row("idx_inv:", k_, [&](uint64_t y) { return idx_inv(y); });
        log.add_row("pi:", k_, [&](uint64_t y) { return pi(y); });
        log.add_row("input_interval_of_output:", k_, [&](uint64_t y) { return input_interval_of_output(y); });
    }

    log.print();
}

// ############################# ACCESS METHODS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
inline uint64_t move_data_structure<enc, mode>::p(uint64_t x) const
{
    if constexpr (enc == POS) return input_blocks.p(x);
    else return (high_part_increments.rank(x) << omega_offs) | low(x);
}

// ############################# CONSTRUCTION METHODS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
template <uint8_t J>
void move_data_structure<enc, mode>::set_row_width(uint8_t width, uint64_t num_threads) requires(mode == UNI)
{
    if constexpr (enc == EF) {
        data.template set_width<num_core_rows + J>(width, num_threads);
    } else {
        if (k_ == 0 || input_blocks.row_width(J) == width) return;
        move_input_interval_blocks old = std::move(input_blocks);
        input_blocks = move_input_interval_blocks();
        uint64_t l_max = 0;

        #pragma omp parallel for num_threads(num_threads) reduction(max : l_max)
        for (uint64_t x = 0; x < k_; x++) l_max = std::max(l_max, old.len(x));

        std::array<uint8_t, max_extra_rows> extra_widths = old.row_widths();
        extra_widths[J] = width;
        compute_layout(l_max, extra_widths, 0, 0, 0, {});
        uint64_t mask = low_bits_mask(width);

        // copy the blocks into the new layout, truncating the values of row J to the new width
        input_blocks.fill(num_threads,
            [&](uint64_t x) { return old.p(x); },
            [&](uint64_t x) { return old.idx(x); },
            [&](uint64_t x) { return old.offs(x); },
            [&](uint64_t x, uint8_t j) { return j == J ? old.row(x, j) & mask : old.row(x, j); },
            [](uint64_t) { return uint64_t { 0 }; }, 0);
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
template <typename row_fnc_t>
void move_data_structure<enc, mode>::add_rows(const row_widths_t& widths, row_fnc_t row_of, uint8_t header_value_width,
    uint64_t l_max, uint64_t num_threads) requires(enc == POS && mode == UNI)
{
    if (num_extra_rows + widths.size() > max_extra_rows) throw std::invalid_argument("a move data structure has at most " + std::to_string(max_extra_rows) + " extra rows");
    move_input_interval_blocks old = std::move(input_blocks);
    input_blocks = move_input_interval_blocks();
    const uint8_t num_old = num_extra_rows;
    std::array<uint8_t, max_extra_rows> extra_widths = old.row_widths();
    for (uint8_t J = 0; J < widths.size(); J++) extra_widths[num_old + J] = widths[J];
    num_extra_rows += uint8_t(widths.size());
    this->header_value_width = header_value_width;
    records_store_lengths = true;
    compute_layout(l_max, extra_widths, 0, 0, 0, {});

    // copy the blocks into the new layout and append the added rows
    input_blocks.fill(num_threads,
        [&](uint64_t x) { return old.p(x); },
        [&](uint64_t x) { return old.idx(x); },
        [&](uint64_t x) { return old.offs(x); },
        [&](uint64_t x, uint8_t J) { return J < num_old ? old.row(x, J) : x < k_ ? row_of(x, J - num_old) : 0; },
        [](uint64_t) { return uint64_t { 0 }; }, 0);
}

// ############################# SERIALIZATION METHODS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::serialize(std::ostream& out) const
{
    out.write((char*) &n, sizeof(uint64_t));
    out.write((char*) &k, sizeof(uint64_t));
    out.write((char*) &k_, sizeof(uint64_t));

    // a is stored with 16 bits
    uint16_t a_ = a;
    out.write((char*) &a_, sizeof(uint16_t));

    if constexpr (enc == POS) {
        input_blocks.serialize_blocks(out);
        if constexpr (has_inverse) output_blocks.serialize_blocks(out);
    } else {
        data.serialize(out);
        high_part_increments.serialize(out);
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::load(std::istream& in, uint8_t num_extra_rows)
{
    this->num_extra_rows = num_extra_rows;
    in.read((char*) &n, sizeof(uint64_t));
    in.read((char*) &k, sizeof(uint64_t));
    in.read((char*) &k_, sizeof(uint64_t));
    uint16_t a_;
    in.read((char*) &a_, sizeof(uint16_t));
    a = a_;
    omega_p = std::bit_width(uint64_t(n));
    omega_idx = std::bit_width(uint64_t(k_));

    if constexpr (enc == POS) {
        input_blocks.load_blocks(in);
        if constexpr (has_inverse) output_blocks.load_blocks(in);
        omega_offs = input_blocks.width_offs();
        this->num_extra_rows = input_blocks.num_extra_rows();
        records_store_lengths = input_blocks.records_store_lengths();
    } else {
        data.load(in);
        omega_offs = data.template width<2>();
        high_part_increments.load(in);
    }
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_TPP
