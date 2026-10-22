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

#include <cassert>

#include <move_data_structure/move_interval_blocks/move_interval_blocks.hpp>

// ############################# INTERVAL BLOCKS #############################

void move_interval_blocks::throw_field_too_wide(uint8_t width, uint64_t value, const char* name)
{
    throw std::logic_error(std::string("interval blocks: the ") + name + " " + std::to_string(value) +
        " does not fit " + std::to_string(width) + " bits");
}

void move_interval_blocks::set_layout(uint64_t n, uint64_t k_, uint8_t omega_block_start, uint64_t num_records)
{
    // with one record per block, ceil(2^64 / records_per_block) does not fit into 64 bits
    if (num_records < 2) throw std::logic_error("interval blocks: fewer than two intervals fit into a block");
    // block_of(x) = floor(x / records_per_block) holds for all x < 2^64 / records_per_block
    assert(k_ < ~uint64_t { 0 } / num_records);
    this->n = n;
    this->k_ = k_;
    this->omega_block_start = omega_block_start;
    mask_block_start = low_bits_mask(omega_block_start);
    records_per_block = num_records;
    slot_record_bits = uint8_t(std::bit_width(num_records - 1));
    records_per_block_reciprocal = ~uint64_t { 0 } / num_records + 1;
    num_blocks = (k_ + num_records) / num_records;
}

void move_interval_blocks::allocate(uint64_t num_threads)
{
    uint64_t num_words = words_per_block * (num_blocks + 1) + padding_words;
    words.resize(num_words); // does not initialize the words (see cache_aligned_no_init_allocator)
    advise_huge_pages(words.data(), blocks_size_in_bytes());

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t w = 0; w < num_words; w++) words[w] = 0;
}

void move_interval_blocks::serialize_base(std::ostream& out) const
{
    out.write((char*) &n, sizeof(uint64_t));
    out.write((char*) &k_, sizeof(uint64_t));
    out.write((char*) &records_per_block, sizeof(uint64_t));
    out.write((char*) &slot_record_bits, sizeof(uint8_t));
    out.write((char*) &num_blocks, sizeof(uint64_t));
    out.write((char*) &omega_block_start, sizeof(uint8_t));
    uint64_t num_words = words.size();
    out.write((char*) &num_words, sizeof(uint64_t));
    write_in_chunks(out, (const char*) words.data(), num_words * sizeof(uint64_t));
}

void move_interval_blocks::load_base(std::istream& in)
{
    in.read((char*) &n, sizeof(uint64_t));
    in.read((char*) &k_, sizeof(uint64_t));
    in.read((char*) &records_per_block, sizeof(uint64_t));
    in.read((char*) &slot_record_bits, sizeof(uint8_t));
    in.read((char*) &num_blocks, sizeof(uint64_t));
    in.read((char*) &omega_block_start, sizeof(uint8_t));
    mask_block_start = low_bits_mask(omega_block_start);
    // records_per_block = 0 <=> the data structure is empty
    records_per_block_reciprocal = records_per_block == 0 ? 0 : ~uint64_t { 0 } / records_per_block + 1;
    uint64_t num_words;
    in.read((char*) &num_words, sizeof(uint64_t));
    words.resize(num_words);
    advise_huge_pages(words.data(), blocks_size_in_bytes());
    read_in_chunks(in, (char*) words.data(), num_words * sizeof(uint64_t));
}

// ############################# INPUT INTERVAL BLOCKS #############################

void move_input_interval_blocks::compute_layout(uint64_t n, uint64_t k_, uint8_t omega_block_start, uint8_t omega_offs,
    uint8_t omega_slot, uint8_t omega_header_value, uint64_t l_max, uint8_t num_extra_rows,
    const std::array<uint8_t, max_extra_rows>& extra_widths, bool store_lengths)
{
    stores_lengths = store_lengths;
    this->omega_offs = omega_offs;
    this->omega_slot = omega_slot;
    this->omega_header_value = omega_header_value;
    num_extra = num_extra_rows;
    omega_extra = extra_widths;
    uint64_t fixed_bits = omega_slot + omega_offs;
    for (uint8_t J = 0; J < num_extra_rows; J++) fixed_bits += omega_extra[J];
    uint64_t header = omega_block_start + omega_header_value;
    // the width of the first field of a record: the length of the interval, or p(x) relative to the block start
    auto rel_p_width = [&](uint64_t num_records) { return uint8_t(std::bit_width(store_lengths ? l_max : (num_records - 1) * l_max)); };

    // returns the number of bits of a block with the given number of records (omit: record 0 omits its rel p)
    auto block_bits = [&](uint64_t num_records, bool omit) {
        uint64_t rel_p_bits = rel_p_width(num_records);
        uint64_t records_bits = num_records * (rel_p_bits + fixed_bits);
        return omit ? std::max(header, rel_p_bits) + records_bits - rel_p_bits : header + records_bits;
    };

    // returns the value of record_read_cost() for the given number of records (see compute_bit_positions)
    auto read_cost = [&](uint64_t num_records, bool omit) {
        uint64_t record_bits = rel_p_width(num_records) + fixed_bits;
        if (!omit && record_bits == 64 && header <= 64 && 64 + num_records * 64 <= bits_per_block) return 0;
        return record_bits <= max_read_field_width ? 1 : 2;
    };

    // choose the maximum number of records that fit into a block
    uint64_t num_records = max_records_per_block;
    while (num_records > 0 && block_bits(num_records, false) > bits_per_block) num_records--;
    if (num_records == 0) throw std::logic_error("the fields of a move data structure do not fit a cache line");
    omits_first_rel_p = false;

    // let record 0 omit its rel p, if more records fit into a block then and record_read_cost() does not increase
    for (uint64_t more = max_records_per_block; !store_lengths && more > num_records; more--) {
        if (block_bits(more, true) <= bits_per_block && read_cost(more, true) <= read_cost(num_records, false)) {
            num_records = more;
            omits_first_rel_p = true;
            break;
        }
    }

    omega_rel_p = rel_p_width(num_records);
    set_layout(n, k_, omega_block_start, num_records);
    compute_bit_positions();
}

void move_input_interval_blocks::serialize_blocks(std::ostream& out) const
{
    serialize_base(out);
    out.write((char*) &omega_rel_p, sizeof(uint8_t));
    out.write((char*) &omega_slot, sizeof(uint8_t));
    out.write((char*) &omega_header_value, sizeof(uint8_t));
    out.write((char*) &omega_offs, sizeof(uint8_t));
    out.write((char*) &num_extra, sizeof(uint8_t));
    out.write((char*) omega_extra.data(), omega_extra.size());
    // the layout of the first field of a record: 0 rel p, 1 rel p omitted in record 0, 2 length
    uint8_t first_field = stores_lengths ? 2 : omits_first_rel_p ? 1 : 0;
    out.write((char*) &first_field, sizeof(uint8_t));
}

void move_input_interval_blocks::load_blocks(std::istream& in)
{
    load_base(in);
    in.read((char*) &omega_rel_p, sizeof(uint8_t));
    in.read((char*) &omega_slot, sizeof(uint8_t));
    in.read((char*) &omega_header_value, sizeof(uint8_t));
    in.read((char*) &omega_offs, sizeof(uint8_t));
    in.read((char*) &num_extra, sizeof(uint8_t));
    in.read((char*) omega_extra.data(), omega_extra.size());
    uint8_t first_field;
    in.read((char*) &first_field, sizeof(uint8_t));
    omits_first_rel_p = first_field == 1;
    stores_lengths = first_field == 2;
    compute_bit_positions();
}

void move_input_interval_blocks::compute_bit_positions()
{
    bits_per_record = omega_rel_p + omega_slot + omega_offs;
    for (uint8_t J = 0; J < num_extra; J++) bits_per_record += omega_extra[J];
    uint64_t header_bits = omega_block_start + omega_header_value;

    if (omits_first_rel_p) {
        // record 0 must not start before the block
        header_bits = std::max<uint64_t>(header_bits, omega_rel_p);
        first_record_bit = header_bits - omega_rel_p;
    } else {
        // pad the header to 64 bits, if the records still fit into the block
        if (header_bits < 64 && 64 + records_per_block * bits_per_record <= bits_per_block) header_bits = 64;
        first_record_bit = header_bits;
    }

    idx_bit = omega_rel_p;
    offs_bit = omega_rel_p + omega_slot;
    uint64_t bit = offs_bit + omega_offs;

    for (uint8_t J = 0; J < num_extra; J++) {
        extra_bit[J] = bit;
        mask_extra[J] = low_bits_mask(omega_extra[J]);
        bit += omega_extra[J];
    }

    mask_rel_p = low_bits_mask(omega_rel_p);
    mask_slot = low_bits_mask(omega_slot);
    mask_offs = low_bits_mask(omega_offs);
    mask_header_value = low_bits_mask(omega_header_value);
    records_are_words = !omits_first_rel_p && !stores_lengths && first_record_bit == 64 && bits_per_record == 64;
    records_are_narrow = bits_per_record <= max_read_field_width;
}

move_input_interval_blocks::record_reader move_input_interval_blocks::reader() const
{
    return { .words = words.data(), .records_per_block = records_per_block, .records_per_block_reciprocal = records_per_block_reciprocal,
        .first_record_bit = first_record_bit, .bits_per_record = bits_per_record, .mask_block_start = mask_block_start, .mask_len = mask_rel_p,
        .mask_slot = mask_slot, .mask_offs = mask_offs, .mask_sym = num_extra > 0 ? mask_extra[0] : 0,
        .idx_bit = uint8_t(idx_bit), .offs_bit = uint8_t(offs_bit), .sym_bit = uint8_t(num_extra > 0 ? extra_bit[0] : 0),
        .slot_record_bits = slot_record_bits, .omega_block_start = omega_block_start, .mask_header_value = mask_header_value,
        .fields_are_narrow = (num_extra > 0 ? extra_bit[0] + omega_extra[0] : bits_per_record) <= max_read_field_width };
}
