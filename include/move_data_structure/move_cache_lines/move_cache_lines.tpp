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
 * @file move_cache_lines.tpp
 * @brief the methods of move_cache_lines: fields, moves and filling the lines
 */

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_TPP

#include <move_data_structure/move_cache_lines/move_cache_lines.hpp>

inline void move_cache_lines::write_field(uint64_t b, uint64_t bit, uint8_t width, uint64_t value)
{
    if (width == 0) [[unlikely]] return;
    uint64_t pos = bits_per_line * b + bit;
    uint64_t* word = words.data() + (pos >> 6);
    uint8_t shift = uint8_t(pos & 63);
    value &= low_bits_mask(width);
    word[0] |= value << shift;
    if (shift + width > 64) word[1] |= value >> (64 - shift);
}

inline void move_cache_lines::overwrite_field_parallel(uint64_t b, uint64_t bit, uint8_t width, uint64_t value)
{
    uint64_t pos = bits_per_line * b + bit;
    uint64_t word = pos >> 6;
    uint64_t shift = pos & 63;
    uint64_t mask = low_bits_mask(width);
    value &= mask;
    atomic_store_bits(words[word], mask << shift, value << shift);
    if (shift + width > 64) atomic_store_bits(words[word + 1], mask >> (64 - shift), value >> (64 - shift));
}

inline void move_input_cache_lines::move(uint64_t& i, uint64_t& x) const
{
    uint64_t b = line_of(x);
    uint64_t j = x - b * records_per_line;
    const uint64_t* B = line(b);
    record_fields_t fields = read_record(B, j);
    uint64_t p_x = (B[0] & mask_line_start) + fields.rel_p;

    // set i to i' = q(x) + (i - p(x)) and x to idx(x)
    b = slot_line(fields.idx_slot);
    j = slot_record(fields.idx_slot);
    B = line(b);
    uint64_t start = B[0] & mask_line_start;
    i = start + rel_p(B, j) + fields.offs + (i - p_x);
    x = b * records_per_line + j;

    // find the input interval containing i'
    for (;;) {
        if (++j == records_per_line) {
            b++;
            B = line(b);
            start = B[0] & mask_line_start;
            j = 0;
        }

        if (i < start + rel_p(B, j)) return;
        x++;
    }
}

inline uint64_t move_input_cache_lines::record_reader::start_move(uint64_t o, uint64_t record, uint64_t& slot) const
{
    slot = idx_slot(record);
    const uint64_t* B = line(slot >> slot_record_bits);
    __builtin_prefetch(B);
    __builtin_prefetch(B + words_per_line);
    return o + ((record >> offs_bit) & mask_offs);
}

inline uint64_t move_input_cache_lines::record_reader::finish_move(
    uint64_t slot, uint64_t offset, const uint64_t*& B, uint64_t& j, uint64_t& record) const
{
    B = line(slot >> slot_record_bits);
    j = slot & ((uint64_t { 1 } << slot_record_bits) - 1);
    record = this->record(B, j);

    // the input interval containing the image is the first one whose length exceeds the remaining offset
    while (offset >= len(record)) {
        offset -= len(record);
        next(B, j);
        record = this->record(B, j);
    }

    return offset;
}

inline uint64_t move_input_cache_lines::scan_from(uint64_t slot, uint64_t i) const
{
    uint64_t b = slot_line(slot);
    uint64_t j = slot_record(slot);
    const uint64_t* B = line(b);
    uint64_t start = B[0] & mask_line_start;
    uint64_t x = b * records_per_line + j;

    for (;;) {
        if (++j == records_per_line) {
            b++;
            B = line(b);
            start = B[0] & mask_line_start;
            j = 0;
        }

        if (i < start + rel_p(B, j)) return x;
        x++;
    }
}

inline uint64_t move_input_cache_lines::interval_near(uint64_t i, uint64_t x) const
{
    uint64_t b = line_of(x);
    uint64_t j = x - b * records_per_line;
    const uint64_t* B = line(b);
    uint64_t start = B[0] & mask_line_start;

    if (i >= start + rel_p(B, j)) {
        // check the x-th input interval
        uint64_t end = j + 1 < records_per_line ? start + rel_p(B, j + 1) : line_start(b + 1);
        if (i < end) [[likely]] return x;

        // search the following records of the line
        for (uint64_t t = j + 1; t + 1 < records_per_line; t++) {
            if (i < start + rel_p(B, t + 1)) return b * records_per_line + t;
        }

        if (j + 1 < records_per_line && i < line_start(b + 1)) return b * records_per_line + records_per_line - 1;

        // search the following lines
        uint64_t last = std::min<uint64_t>(b + near_lines, num_lines - 1);

        do {
            if (b == last) return k_;
            b++;
        } while (i >= line_start(b + 1));

        return interval_in_line(b, i);
    }

    // search the preceding records of the line
    if (i >= start) {
        for (uint64_t t = 0; t < j; t++) {
            if (i < start + rel_p(B, t + 1)) return b * records_per_line + t;
        }
    }

    // search the preceding lines
    uint64_t first = b > near_lines ? b - near_lines : 0;

    do {
        if (b == first) return k_;
        b--;
    } while (i < line_start(b));

    return interval_in_line(b, i);
}

template <typename p_fnc_t, typename idx_fnc_t, typename offs_fnc_t, typename extra_fnc_t, typename header_value_fnc_t>
void move_input_cache_lines::fill(uint64_t num_threads, p_fnc_t p_of, idx_fnc_t idx_of, offs_fnc_t offs_of, extra_fnc_t extra_of,
    header_value_fnc_t header_value_of, uint64_t sentinel_header_value)
{
    allocate(num_threads);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t b = 0; b < num_lines; b++) {
        uint64_t first = b * records_per_line;
        uint64_t start = p_of(std::min(first, k_));
        write_field(b, 0, omega_line_start, start);
        write_checked_field(b, omega_line_start, omega_header_value, first >= k_ ? sentinel_header_value : header_value_of(first), "header value");

        for (uint64_t t = 0; t < records_per_line; t++) {
            uint64_t x = first + t;
            uint64_t bit = first_record_bit + t * bits_per_record;
            if (stores_lengths) write_checked_field(b, bit, omega_rel_p, x < k_ ? p_of(x + 1) - p_of(x) : 0, "length");
            else if (t != 0) write_checked_field(b, bit, omega_rel_p, (x <= k_ ? p_of(x) : n) - start, "relative p");
            write_checked_field(b, bit + idx_bit, omega_slot, slot_of(x <= k_ ? idx_of(x) : k_), "idx slot");
            write_checked_field(b, bit + offs_bit, omega_offs, x <= k_ ? offs_of(x) : 0, "offs");
            for (uint8_t J = 0; J < num_extra; J++) write_field(b, bit + extra_bit[J], omega_extra[J], x <= k_ ? extra_of(x, J) : 0);
        }
    }

    write_sentinel(sentinel_header_value);
}

inline void move_input_cache_lines::write_idx_offs_parallel(uint64_t x, uint64_t idx, uint64_t offs)
{
    uint64_t b = line_of(x);
    uint64_t bit = first_record_bit + (x - b * records_per_line) * bits_per_record;
    uint64_t slot = slot_of(idx);
    if (slot > mask_slot) [[unlikely]] throw_field_too_wide(omega_slot, slot, "idx slot");
    if (offs > mask_offs) [[unlikely]] throw_field_too_wide(omega_offs, offs, "offs");

    // idx and offs are adjacent; they are written as one value if they fit into 64 bits together
    if (omega_slot + omega_offs <= 64) {
        or_field_parallel(b, bit + idx_bit, omega_slot + omega_offs, slot | (offs << omega_slot));
    } else {
        or_field_parallel(b, bit + idx_bit, omega_slot, slot);
        or_field_parallel(b, bit + offs_bit, omega_offs, offs);
    }
}

inline void move_input_cache_lines::write_idx_offs_row_parallel(uint64_t x, uint64_t idx, uint64_t offs, uint8_t J, uint64_t value)
{
    uint64_t b = line_of(x);
    uint64_t bit = first_record_bit + (x - b * records_per_line) * bits_per_record;
    uint64_t slot = slot_of(idx);
    if (slot > mask_slot) [[unlikely]] throw_field_too_wide(omega_slot, slot, "idx slot");
    if (offs > mask_offs) [[unlikely]] throw_field_too_wide(omega_offs, offs, "offs");
    if (value > mask_extra[J]) [[unlikely]] throw_field_too_wide(omega_extra[J], value, "extra row");

    // the first extra row follows offs; the three fields are written as one value if they fit into 64 bits together
    if (J == 0 && omega_slot + omega_offs + omega_extra[0] <= 64) {
        or_field_parallel(b, bit + idx_bit, omega_slot + omega_offs + omega_extra[0],
            slot | (offs << omega_slot) | (value << (omega_slot + omega_offs)));
    } else {
        write_idx_offs_parallel(x, idx, offs);
        or_field_parallel(b, bit + extra_bit[J], omega_extra[J], value);
    }
}

inline move_input_cache_lines::record_fields_t move_input_cache_lines::read_record(const uint64_t* B, uint64_t j) const
{
    if (records_are_words) return unpack_record(B[1 + j]);
    uint64_t bit = first_record_bit + j * bits_per_record;

    record_fields_t fields = records_are_narrow ? unpack_record(read_field(B, bit, ~uint64_t { 0 }))
        : record_fields_t { read_field(B, bit, mask_rel_p), read_field(B, bit + idx_bit, mask_slot), read_field(B, bit + offs_bit, mask_offs) };

    if (stores_lengths) fields.rel_p = sum_of_lengths(B, j);
    else if (j == 0) fields.rel_p = 0;
    return fields;
}

inline uint64_t move_input_cache_lines::interval_in_line(uint64_t b, uint64_t i) const
{
    const uint64_t* B = line(b);
    uint64_t start = B[0] & mask_line_start;
    uint64_t j = 0;
    while (j + 1 < records_per_line && i >= start + rel_p(B, j + 1)) j++;
    return b * records_per_line + j;
}

inline uint64_t move_input_cache_lines::rel_p(const uint64_t* B, uint64_t j) const
{
    if (stores_lengths) [[unlikely]] return sum_of_lengths(B, j);
    // if the records are words, record 0 stores 0 as its rel p; else, the bits read for it can be header bits
    if (records_are_words) return B[1 + j] & mask_rel_p;
    uint64_t rel = read_field(B, first_record_bit + j * bits_per_record, mask_rel_p);
    return j == 0 ? 0 : rel;
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_TPP
