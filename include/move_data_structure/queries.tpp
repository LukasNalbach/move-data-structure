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

/**
 * @file queries.tpp
 * @brief the queries of move_data_structure: move, inverse move and input intervals
 */

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_QUERIES_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_QUERIES_TPP

#include <move_data_structure/move_data_structure.hpp>

template <move_pos_encoding_t enc, move_mode_t mode>
inline uint64_t move_data_structure<enc, mode>::advance_to_interval(uint64_t& o, uint64_t& x) const requires(enc == ELIAS_FANO)
{
    const uint64_t mask = low_mask();
    uint64_t low_x = low(x);

    for (;;) {
        uint64_t low_next = low(x + 1);
        uint64_t l = (low_next - low_x) & mask;
        if (o < l) return low_x;
        o -= l;
        x++;
        low_x = low_next;
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline uint64_t move_data_structure<enc, mode>::distance(uint64_t x1, uint64_t o1, uint64_t x2, uint64_t o2) const
{
    if constexpr (enc == ELIAS_FANO) {
        // add the lengths of x1, ..., x2 - 1, if the entries of D_rows of x1, ..., x2 occupy at most one cache line
        if ((x2 - x1 + 1) * D_rows.bits_per_entry() <= 8 * cache_line_bytes) {
            const uint64_t mask = low_mask();
            uint64_t sum = o2;
            for (uint64_t x = x1; x < x2; x++) sum += (low(x + 1) - low(x)) & mask;
            return sum - o1;
        }
    }

    return (p(x2) + o2) - (p(x1) + o1);
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline uint64_t move_data_structure<enc, mode>::input_interval_at(uint64_t i) const requires(enc == ELIAS_FANO)
{
    const uint64_t high = i >> omega_offs;
    const uint64_t i_low = i & low_mask();
    uint64_t first = 0; // the first input interval with the high part of i

    if (high != 0) {
        // the last line whose rank is at most the rank of the set bit at which the high part increases to the one of i
        const uint64_t rank = high - 1;
        const uint64_t j = rank >> D_high_select_shift;
        uint64_t l = D_high_select.template get<0, uint64_t>(j);
        uint64_t num = D_high_select.template get<0, uint64_t>(j + 1) - l + 1;

        while (num > 1) {
            const uint64_t half = num / 2;
            l += D_high.line_rank(l + half) <= rank ? half : 0;
            num -= half;
        }

        first = D_high.select_in_line(l, rank) + 1;
    }

    if (low(first) > i_low) return first - 1;
    // the last input interval with the high part of i whose D_low is at most the low part of i
    uint64_t x = first;
    uint64_t num = std::min<uint64_t>(D_high.next_one(first), k_) - first + 1;

    while (num > 1) {
        const uint64_t half = num / 2;
        x += low(x + half) <= i_low ? half : 0;
        num -= half;
    }

    return x;
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::move(uint64_t& i, uint64_t& x) const
{
    if constexpr (enc == CACHE_LINES) {
        input_lines.move(i, x);
    } else {
        // move the offset o = i - p(x) as in move_offset, then add p(x) of the resulting interval x
        uint64_t o = i - p(x);
        auto [idx_x, offs_x] = D_rows.template get_multiple<2, 1, uint64_t>(x);
        o += offs_x;
        x = idx_x;
        // prefetch the line of D_high that contains bit idx(x)
        __builtin_prefetch(D_high.line(x));
        uint64_t low_x = advance_to_interval(o, x);
        i = ((D_high.rank(x) << omega_offs) | low_x) + o;
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::move_offset(uint64_t& o, uint64_t& x) const requires(enc == ELIAS_FANO)
{
    // read idx(x) and offs(x) with one access
    auto [idx_x, offs_x] = D_rows.template get_multiple<2, 1, uint64_t>(x);
    o += offs_x;
    x = idx_x;
    advance_to_interval(o, x);
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline typename move_data_structure<enc, mode>::query_pair_t move_data_structure<enc, mode>::interval_and_offset(uint64_t i) const
{
    uint64_t x = bin_search_max_leq(i, uint64_t(0), k_ - 1, [&](uint64_t y) { return p(y); });
    return { x, i - p(x) };
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::move_inv(uint64_t& i, uint64_t& y) const requires(has_inverse)
{
    output_lines.move(i, y);
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::move_inv(uint64_t& i, uint64_t& x, uint64_t& y) const requires(has_inverse)
{
    output_lines.move(i, y);
    x = input_interval_of(i, y);
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_QUERIES_TPP
