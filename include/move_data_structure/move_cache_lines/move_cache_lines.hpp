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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_HPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_HPP

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <data_structures/interleaved_bit_aligned_vectors.hpp>
#include <misc/files.hpp>
#include <misc/utils.hpp>


/**
 * @brief stores the intervals of a move data structure in lines of one cache line (64 bytes) each
 *
 * Bit j of a line is bit j mod 64 of its word j / 64. Each line starts with a header that contains the line
 * start, i.e., the starting position of the first interval of the line, in its lowest omega_line_start bits.
 * After the header, the line stores records_per_line records, one per interval: interval x is stored in record
 * x mod records_per_line of line x / records_per_line. A record contains the starting position of its interval
 * relative to the line start (or values from which it is computed) and the fields defined by
 * move_input_cache_lines.
 *
 * The slot of interval x is (x / records_per_line) << slot_record_bits | (x mod records_per_line), i.e., the
 * line index followed by the record index; the records reference intervals by their slots. The data lines
 * store the intervals [0..k'], where interval k' is the sentinel interval starting at n (and the records after
 * it in the last data line store intervals of length 0 starting at n). They are followed by a sentinel line
 * with the line start n, in which every record references interval k'.
 */
class move_cache_lines {
public:
    // ############################# TYPES AND PARAMETERS #############################

    static constexpr uint64_t words_per_line = cache_aligned_no_init_allocator<uint64_t>::alignment / sizeof(uint64_t); // number of 64-bit words per line
    static constexpr uint64_t bits_per_line = 64 * words_per_line;
    static constexpr uint64_t max_records_per_line = 16; // maximum number of intervals per line
    static constexpr uint8_t max_extra_rows = 5; // maximum number of extra rows

    // maximum width (in bits) of a field read by read_field: it reads 8 bytes starting at the byte that contains the
    // first bit of the field, which is one of the 8 bits of that byte
    static constexpr uint64_t max_read_field_width = 8 * sizeof(uint64_t) - (8 - 1);

protected:
    // number of zero words after the sentinel line; read_field reads 8 bytes and read_word 16 bytes starting at a
    // byte of a line, so neither reads past the end of words
    static constexpr uint64_t padding_words = 4;

    // ############################# VARIABLES #############################

    /* the lines: [0..words_per_line * (num_lines + 1) + padding_words - 1]; line b is
       words[words_per_line * b..words_per_line * (b + 1) - 1] for b in [0..num_lines], line num_lines is the
       sentinel line, the last padding_words words are 0 */
    std::vector<uint64_t, cache_aligned_no_init_allocator<uint64_t>> words;
    uint64_t n = 0; // n, the size of the domain of the permutation
    uint64_t k_ = 0; // k', the number of intervals; interval k' is the sentinel interval starting at n
    uint64_t records_per_line = 0; // number of intervals per line, at least 2
    uint64_t records_per_line_reciprocal = 0; // ceil(2^64 / records_per_line); interval x lies in line hi_mul(x, records_per_line_reciprocal)
    uint64_t num_lines = 0; // number of data lines; line num_lines is the sentinel line
    uint64_t mask_line_start = 0; // mask of a line start
    uint8_t omega_line_start = 0; // bits per line start
    uint8_t slot_record_bits = 0; // bits of the record index in a slot, bit_width(records_per_line - 1)

    // ############################# INTERNAL METHODS #############################

    /**
     * @brief reads a field of at most max_read_field_width bits from a line
     * @param B pointer to the line
     * @param bit position of the first bit of the field in the line
     * @param mask mask of the width of the field
     * @return value of the field
     */
    static inline uint64_t read_field(const uint64_t* B, uint64_t bit, uint64_t mask)
    {
        uint64_t word;
        std::memcpy(&word, reinterpret_cast<const char*>(B) + (bit >> 3), sizeof(uint64_t));
        return (word >> (bit & 7)) & mask;
    }

    // returns the 64 bits of the line B starting at the given bit
    static inline uint64_t read_word(const uint64_t* B, uint64_t bit)
    {
        uint64_t lo, hi;
        std::memcpy(&lo, reinterpret_cast<const char*>(B) + (bit >> 3), sizeof(uint64_t));
        std::memcpy(&hi, reinterpret_cast<const char*>(B) + (bit >> 3) + sizeof(uint64_t), sizeof(uint64_t));
        return (lo >> (bit & 7)) | ((hi << 1) << (63 - (bit & 7)));
    }

    // returns the upper 64 bits of the 128-bit product of a and b
    static inline uint64_t hi_mul(uint64_t a, uint64_t b) { return uint64_t((uint128_t(a) * b) >> 64); }

    /**
     * @brief writes a field of the b-th line; the field must be zero and no other thread may write to the
     *        line at the same time
     * @param b line index
     * @param bit position of the first bit of the field in the line
     * @param width width of the field (in bits)
     * @param value value to write (its lowest width bits are written)
     */
    inline void write_field(uint64_t b, uint64_t bit, uint8_t width, uint64_t value);

    /**
     * @brief writes a field like write_field, but throws an exception if the value does not fit into the field
     * @param b line index
     * @param bit position of the first bit of the field in the line
     * @param width width of the field (in bits)
     * @param value value to write
     * @param name name of the field (for the error message)
     */
    inline void write_checked_field(uint64_t b, uint64_t bit, uint8_t width, uint64_t value, const char* name)
    {
        if (value > low_bits_mask(width)) [[unlikely]] throw_field_too_wide(width, value, name);
        write_field(b, bit, width, value);
    }

    /**
     * @brief overwrites a field of the b-th line atomically per word; other threads may write other fields in
     *        the same words at the same time
     * @param b line index
     * @param bit position of the first bit of the field in the line
     * @param width width of the field (in bits)
     * @param value value to write (its lowest width bits are written)
     */
    inline void overwrite_field_parallel(uint64_t b, uint64_t bit, uint8_t width, uint64_t value);

    /**
     * @brief writes a field of the b-th line with an atomic OR per word; the field must be zero, other threads
     *        may write other fields in the same words at the same time
     * @param b line index
     * @param bit position of the first bit of the field in the line
     * @param width width of the field (in bits), at most 64
     * @param value value to write (must fit into width bits)
     */
    inline void or_field_parallel(uint64_t b, uint64_t bit, uint8_t width, uint64_t value)
    {
        uint64_t pos = bits_per_line * b + bit;
        uint64_t shift = pos & 63;
        __atomic_fetch_or(&words[pos >> 6], value << shift, __ATOMIC_RELAXED);
        if (shift + width > 64) __atomic_fetch_or(&words[(pos >> 6) + 1], value >> (64 - shift), __ATOMIC_RELAXED);
    }

    // throws the exception of write_checked_field
    [[noreturn]] static void throw_field_too_wide(uint8_t width, uint64_t value, const char* name);

    /**
     * @brief sets n, k', the width of a line start and the number of intervals per line, and computes the
     *        number of lines; throws an exception if fewer than two intervals fit into a line
     * @param n size of the domain of the permutation
     * @param k_ k', the number of intervals
     * @param omega_line_start bits per line start
     * @param num_records number of intervals per line
     */
    void set_layout(uint64_t n, uint64_t k_, uint8_t omega_line_start, uint64_t num_records);

    // writes the lines and the variables of this class to an output stream
    void serialize_base(std::ostream& out) const;

    // reads the data written by serialize_base
    void load_base(std::istream& in);

public:
    // ############################# ACCESS METHODS #############################

    // returns a pointer to the b-th line
    inline const uint64_t* line(uint64_t b) const { return words.data() + words_per_line * b; }

    // returns the line start of the b-th line
    inline uint64_t line_start(uint64_t b) const { return line(b)[0] & mask_line_start; }

    // returns the index of the line containing the x-th interval
    inline uint64_t line_of(uint64_t x) const { return hi_mul(x, records_per_line_reciprocal); }

    // returns the slot of the x-th interval
    inline uint64_t slot_of(uint64_t x) const
    {
        uint64_t b = line_of(x);
        return (b << slot_record_bits) | (x - b * records_per_line);
    }

    // returns the line index of a slot
    inline uint64_t slot_line(uint64_t slot) const { return slot >> slot_record_bits; }

    // returns the record index of a slot
    inline uint64_t slot_record(uint64_t slot) const { return slot & ((uint64_t { 1 } << slot_record_bits) - 1); }

    // returns the index of the interval in a slot
    inline uint64_t interval_of_slot(uint64_t slot) const { return slot_line(slot) * records_per_line + slot_record(slot); }

    // returns the largest slot
    inline uint64_t max_slot() const { return (num_lines << slot_record_bits) | (records_per_line - 1); }

    // returns the number of intervals per line
    inline uint64_t intervals_per_line() const { return records_per_line; }

    // returns the number of data lines (the sentinel line is not counted)
    inline uint64_t data_lines() const { return num_lines; }

    // returns the size of the lines in bytes
    inline uint64_t lines_size_in_bytes() const { return words.size() * sizeof(uint64_t); }

    // ############################# CONSTRUCTION METHODS #############################

    // allocates the lines and sets them to zero; the layout has to be computed first (see compute_layout of the
    // derived classes)
    void allocate(uint64_t num_threads = 1);
};

/**
 * @brief lines storing the input intervals of a move data structure (and in the mode BI also its output intervals)
 *
 * Layout of a line (bit positions relative to the start of the line):
 *
 *   header:      [0, omega_line_start) line start; [omega_line_start, omega_line_start + omega_header_value)
 *                the header value: with an inverse, the slot of the interval of the other direction that contains
 *                the line start; without an inverse, a value the user of the lines writes (see set_header_value)
 *   record t:    starts at bit first_record_bit + t * bits_per_record, t in [0..records_per_line-1]:
 *                [0, omega_rel_p) rel p, i.e., p(x) minus the line start (len(x) if stores_lengths);
 *                [idx_bit, idx_bit + omega_slot) idx(x), as a slot; [offs_bit, offs_bit + omega_offs) offs(x);
 *                [extra_bit[J], extra_bit[J] + omega_extra[J]) the J-th extra row, J in [0..num_extra-1]
 *
 * The rel p of record 0 is not written (rel_p returns 0 for it). If omits_first_rel_p is set, record 0 starts
 * omega_rel_p bits before the end of the header, so its rel p field overlaps the header. Else the header is padded
 * to 64 bits if the records still fit into the line. If stores_lengths is set, p(x) is the line start plus the lengths of the
 * records before the record of x (see record_reader).
 *
 * In the mode BI, the output intervals are stored in lines of this type as well, with p_inv, idx_inv and offs_inv
 * in place of p, idx and offs; the header value is the slot of the input interval containing the line start.
 */
class move_input_cache_lines : public move_cache_lines {
public:
    // ############################# TYPES AND PARAMETERS #############################

    // number of lines interval_near searches before and after the line of the input interval it starts at
    static constexpr uint64_t near_lines = 3;

    // the fields of a record that a move query reads
    struct record_fields_t {
        uint64_t rel_p; // p(x) relative to the line start
        uint64_t idx_slot; // idx(x), as a slot
        uint64_t offs; // offs(x)
    };

    /**
     * @brief a copy of the layout of lines that store the lengths of the intervals (see stores_lengths) and of the
     *        pointer to them; reads the fields of a record up to the extra row 0 as one word (see
     *        records_fit_one_word) and the extra rows as another one. A position is addressed by its input
     *        interval (the line B and the record j) and its offset in the interval. A move query is split into
     *        start_move, which reads only the record of the input interval it starts in, and finish_move, which
     *        reads the records from the one of idx(x) to the one it ends in.
     */
    struct record_reader {
        const uint64_t* words; // the lines
        uint64_t records_per_line; // number of intervals per line
        uint64_t records_per_line_reciprocal; // ceil(2^64 / records_per_line)
        uint64_t first_record_bit; // position of the first bit of record 0 in a line
        uint64_t bits_per_record; // bits per record
        uint64_t mask_line_start; // mask of a line start
        uint64_t mask_len; // mask of the length of an interval
        uint64_t mask_slot; // mask of a slot
        uint64_t mask_offs; // mask of offs(x)
        uint64_t mask_sym; // mask of the extra row 0
        uint8_t idx_bit; // position of the first bit of idx(x) in a record
        uint8_t offs_bit; // position of the first bit of offs(x) in a record
        uint8_t sym_bit; // position of the first bit of the extra row 0 in a record
        uint8_t slot_record_bits; // bits of the record index in a slot
        uint8_t omega_line_start; // bits per line start
        uint64_t mask_header_value; // mask of the header value (see mds_params::header_value_width)
        bool fields_are_narrow; // true <=> the fields up to the extra row 0 have at most max_read_field_width bits (they are read with one read_field)

        // returns a pointer to the b-th line
        inline const uint64_t* line(uint64_t b) const { return words + words_per_line * b; }

        // returns the 64 bits of the line B starting at record j
        [[gnu::always_inline]] inline uint64_t record(const uint64_t* B, uint64_t j) const
        {
            const uint64_t bit = first_record_bit + j * bits_per_record;
            if (fields_are_narrow) [[likely]] return read_field(B, bit, ~uint64_t { 0 });
            return read_word(B, bit);
        }

        // returns the length of the interval of a record
        inline uint64_t len(uint64_t record) const { return record & mask_len; }

        // returns the extra row 0 of a record (for M_LF, the symbol L')
        inline uint64_t sym(uint64_t record) const { return (record >> sym_bit) & mask_sym; }

        // returns the 64 bits of the line B starting at the extra row 0 of record j (the extra rows have at most max_read_field_width bits)
        inline uint64_t extras(const uint64_t* B, uint64_t j) const { return read_field(B, first_record_bit + j * bits_per_record + sym_bit, ~uint64_t { 0 }); }

        // returns the header value of the line B (see mds_params::header_value_width)
        inline uint64_t header_value(const uint64_t* B) const { return read_field(B, omega_line_start, mask_header_value); }

        // returns a pointer to the line of the x-th input interval and sets j to its record index in the line
        inline const uint64_t* line_ptr_of(uint64_t x, uint64_t& j) const
        {
            const uint64_t b = hi_mul(x, records_per_line_reciprocal);
            j = x - b * records_per_line;
            return line(b);
        }

        // moves the record j of the line B to the previous record
        inline void previous(const uint64_t*& B, uint64_t& j) const
        {
            const bool first = j == 0;
            B = first ? B - words_per_line : B;
            j = first ? records_per_line - 1 : j - 1;
        }

        // moves the record j of the line B to the next record
        inline void next(const uint64_t*& B, uint64_t& j) const
        {
            const bool last = j + 1 == records_per_line;
            B = last ? B + words_per_line : B;
            j = last ? 0 : j + 1;
        }

        // returns the index of the line B
        inline uint64_t line_index(const uint64_t* B) const { return uint64_t(B - words) / words_per_line; }

        // returns idx(x) of a record, as a slot
        inline uint64_t idx_slot(uint64_t record) const { return (record >> idx_bit) & mask_slot; }

        // fetches the line before the line of a slot into the cache, if the slot is one of the first d records of its line
        [[gnu::always_inline]] inline void prefetch_before(uint64_t slot, uint64_t d) const
        {
            const uint64_t* B = line(slot >> slot_record_bits);
            const uint64_t before = uint64_t((slot & ((uint64_t { 1 } << slot_record_bits) - 1)) < d) & uint64_t(B != words);
            __builtin_prefetch(B - before * words_per_line);
        }

        // fetches the k-th line after the line of a slot into the cache
        [[gnu::always_inline]] inline void prefetch_after(uint64_t slot, uint64_t k) const
        {
            __builtin_prefetch(line(slot >> slot_record_bits) + k * words_per_line);
        }

        // returns the index of the input interval in a slot
        inline uint64_t interval_of_slot(uint64_t slot) const
        {
            return (slot >> slot_record_bits) * records_per_line + (slot & ((uint64_t { 1 } << slot_record_bits) - 1));
        }

        /**
         * @brief starts a move query of the position with offset o in an input interval x: prefetches the line
         *        finish_move reads first and the line after it
         * @param o offset of the position in x
         * @param record the record of x
         * @param slot is set to the slot of idx(x)
         * @return o + offs(x), the offset of the image of the position from p(idx(x))
         */
        [[gnu::always_inline]] inline uint64_t start_move(uint64_t o, uint64_t record, uint64_t& slot) const;

        /**
         * @brief finishes a move query started by start_move
         * @param slot the slot of idx(x)
         * @param offset the value returned by start_move
         * @param B is set to the line of the input interval containing the image of the position
         * @param j is set to its record index in the line
         * @param record is set to its record
         * @return the offset of the image of the position in its input interval
         */
        [[gnu::always_inline]] inline uint64_t finish_move(uint64_t slot, uint64_t offset, const uint64_t*& B, uint64_t& j,
            uint64_t& record) const;
    };

protected:
    // ############################# VARIABLES #############################

    uint8_t omega_rel_p = 0; // bits per rel p, i.e., per p(x) relative to the line start (or per len(x), see stores_lengths)
    uint8_t omega_slot = 0; // bits per slot of an input interval (idx(x) is stored as a slot)
    uint8_t omega_offs = 0; // bits per offs(x)
    uint8_t omega_header_value = 0; // bits per header value
    uint8_t num_extra = 0; // number of extra rows
    std::array<uint8_t, max_extra_rows> omega_extra {}; // [0..max_extra_rows-1] bits per extra row; 0 for J >= num_extra
    bool omits_first_rel_p = false; // true <=> record 0 of a line starts omega_rel_p bits before the end of the header
    bool stores_lengths = false; // true <=> a record stores the length of its interval instead of its rel p
    uint64_t first_record_bit = 0; // position of the first bit of record 0 in a line
    uint64_t bits_per_record = 0; // bits per record
    uint64_t idx_bit = 0; // position of the first bit of idx(x) in a record
    uint64_t offs_bit = 0; // position of the first bit of offs(x) in a record
    std::array<uint64_t, max_extra_rows> extra_bit {}; // [0..max_extra_rows-1] position of the first bit of extra row J in a record
    uint64_t mask_rel_p = 0; // mask of a rel p (or of a length, see stores_lengths)
    uint64_t mask_slot = 0; // mask of a slot of an input interval
    uint64_t mask_offs = 0; // mask of offs(x)
    uint64_t mask_header_value = 0; // mask of a header value
    std::array<uint64_t, max_extra_rows> mask_extra {}; // [0..max_extra_rows-1] mask of extra row J
    bool records_are_words = false; // true <=> the header and each record are 64 bits wide (record j is word 1 + j)
    bool records_are_narrow = false; // true <=> a record has at most max_read_field_width bits (it is read with one read_field)

    // ############################# INTERNAL METHODS #############################

    // returns the field of record j of the line B that starts at the given bit of the record
    inline uint64_t record_field(const uint64_t* B, uint64_t j, uint64_t bit, uint64_t mask) const
    {
        if (records_are_words) return (B[1 + j] >> bit) & mask;
        return read_field(B, first_record_bit + j * bits_per_record + bit, mask);
    }

    // returns the field of the record of the x-th input interval that starts at the given bit of the record
    inline uint64_t field(uint64_t x, uint64_t bit, uint64_t mask) const
    {
        uint64_t b = line_of(x);
        return record_field(line(b), x - b * records_per_line, bit, mask);
    }

    // returns p relative to the line start of record j of the line B
    inline uint64_t rel_p(const uint64_t* B, uint64_t j) const;

    // returns the sum of the lengths of the records 0 to j - 1 of the line B (see stores_lengths)
    inline uint64_t sum_of_lengths(const uint64_t* B, uint64_t j) const
    {
        uint64_t sum = 0;
        for (uint64_t t = 0; t < j; t++) sum += read_field(B, first_record_bit + t * bits_per_record, mask_rel_p);
        return sum;
    }

    // extracts the fields that a move query reads from a record read as one word
    inline record_fields_t unpack_record(uint64_t record) const
    {
        return { record & mask_rel_p, (record >> idx_bit) & mask_slot, (record >> offs_bit) & mask_offs };
    }

    // returns the fields of record j of the line B that a move query reads
    inline record_fields_t read_record(const uint64_t* B, uint64_t j) const;

    // returns p of the input interval after the one in a slot
    inline uint64_t p_after(uint64_t slot) const
    {
        uint64_t b = slot_line(slot);
        uint64_t j = slot_record(slot) + 1;
        if (j == records_per_line) return line_start(b + 1);
        return (line(b)[0] & mask_line_start) + rel_p(line(b), j);
    }

    // returns the index of the input interval containing position i; the b-th line has to contain i
    inline uint64_t interval_in_line(uint64_t b, uint64_t i) const;

    // computes the bit positions and the masks of the fields from their widths and the number of intervals per line
    void compute_bit_positions();

public:
    // ############################# MISC PUBLIC METHODS #############################

    // true <=> the records store the lengths of the intervals, the fields of a record up to the extra row 0 have at
    // most 64 bits, and its extra rows at most max_read_field_width bits (the condition for reading the lines with a record_reader)
    inline bool records_fit_one_word() const
    {
        return stores_lengths && num_extra > 0 && extra_bit[0] + omega_extra[0] <= 64 && bits_per_record - extra_bit[0] <= max_read_field_width;
    }

    // returns true <=> a record stores the length of its interval instead of its rel p (see stores_lengths)
    inline bool records_store_lengths() const { return stores_lengths; }

    // returns a record_reader of the lines (see records_fit_one_word)
    record_reader reader() const;

    // returns how a record is read: 0 = as one aligned word, 1 = with one unaligned read, 2 = with one read per field
    inline uint8_t record_read_cost() const { return records_are_words ? 0 : records_are_narrow ? 1 : 2; }

    // returns the number of bits per offs(x)
    inline uint8_t width_offs() const { return omega_offs; }

    // returns the number of extra rows
    inline uint8_t num_extra_rows() const { return num_extra; }

    // returns the width of the J-th extra row (in bits)
    inline uint8_t row_width(uint8_t J) const { return omega_extra[J]; }

    // returns the widths of the extra rows (in bits)
    inline const std::array<uint8_t, max_extra_rows>& row_widths() const { return omega_extra; }

    // ############################# ACCESS METHODS #############################

    // returns p(x), the starting position of the x-th input interval
    inline uint64_t p(uint64_t x) const
    {
        uint64_t b = line_of(x);
        const uint64_t* B = line(b);
        return (B[0] & mask_line_start) + rel_p(B, x - b * records_per_line);
    }

    // returns the length of the x-th input interval
    inline uint64_t len(uint64_t x) const
    {
        if (stores_lengths) return field(x, 0, mask_rel_p);
        uint64_t slot = slot_of(x);
        return p_after(slot) - p_at(slot);
    }

    // returns q(x), the starting position of the image of the x-th input interval
    inline uint64_t q(uint64_t x) const
    {
        uint64_t b = line_of(x);
        record_fields_t fields = read_record(line(b), x - b * records_per_line);
        return p_at(fields.idx_slot) + fields.offs;
    }

    // returns idx(x), the index of the input interval containing q(x)
    inline uint64_t idx(uint64_t x) const { return interval_of_slot(field(x, idx_bit, mask_slot)); }

    // returns offs(x) = q(x) - p(idx(x))
    inline uint64_t offs(uint64_t x) const { return field(x, offs_bit, mask_offs); }

    // returns p of the input interval in a slot
    inline uint64_t p_at(uint64_t slot) const
    {
        const uint64_t* B = line(slot_line(slot));
        return (B[0] & mask_line_start) + rel_p(B, slot_record(slot));
    }

    // returns the header value of the line containing the x-th input interval
    inline uint64_t header_value(uint64_t x) const { return read_field(line(line_of(x)), omega_line_start, mask_header_value); }

    // returns the value of the J-th extra row of the x-th input interval
    inline uint64_t row(uint64_t x, uint8_t J) const { return field(x, extra_bit[J], mask_extra[J]); }

    // ############################# QUERY METHODS #############################

    /**
     * @brief move query: changes (i, x) to (i', x') with i' = f_I(i) and i' in [p(x'), p(x'+1))
     * @param i [0..n-1] position
     * @param x [0..k'-1] index of the input interval containing i
     */
    inline void move(uint64_t& i, uint64_t& x) const;

    // fetches the line that move(i, x) reads after the record of x (the line of idx(x)) into the cache
    inline void prefetch_move(uint64_t x) const
    {
        __builtin_prefetch(line(slot_line(field(x, idx_bit, mask_slot))));
    }

    /**
     * @brief returns the index of the input interval containing position i, found by a scan starting at the
     *        input interval in a slot
     * @param slot slot of the input interval containing i or of a preceding input interval
     * @param i [0..n-1] position
     * @return index of the input interval containing i
     */
    inline uint64_t scan_from(uint64_t slot, uint64_t i) const;

    /**
     * @brief returns the index of the input interval containing position i, searching the line of the x-th
     *        input interval and the near_lines lines before and after it
     * @param i [0..n-1] position
     * @param x [0..k'-1] index of the input interval to start the search at
     * @return index of the input interval containing i, or k' if it does not lie in the searched lines
     */
    inline uint64_t interval_near(uint64_t i, uint64_t x) const;

    // ############################# CONSTRUCTION METHODS #############################

    /**
     * @brief computes the layout of a line for the given widths: chooses the number of intervals per line,
     *        the width of rel p and whether record 0 omits its rel p
     * @param n size of the domain of the permutation
     * @param k_ k', the number of input intervals
     * @param omega_line_start bits per line start
     * @param omega_offs bits per offs(x)
     * @param omega_slot bits per slot of an input interval
     * @param omega_header_value bits per header value
     * @param l_max maximum length of an input interval
     * @param num_extra_rows number of extra rows
     * @param extra_widths widths of the extra rows (in bits)
     * @param store_lengths true <=> the records store the lengths of the intervals (see stores_lengths)
     */
    void compute_layout(uint64_t n, uint64_t k_, uint8_t omega_line_start, uint8_t omega_offs, uint8_t omega_slot, uint8_t omega_header_value,
        uint64_t l_max, uint8_t num_extra_rows, const std::array<uint8_t, max_extra_rows>& extra_widths, bool store_lengths = false);

    /**
     * @brief allocates and fills the lines; compute_layout has to be called first
     * @param num_threads number of threads to use
     * @param p_of p_of(x) returns p(x)
     * @param idx_of idx_of(x) returns idx(x), as an interval index
     * @param offs_of offs_of(x) returns offs(x)
     * @param extra_of extra_of(x, J) returns the value of the J-th extra row of the x-th input interval
     * @param header_value_of header_value_of(x) returns the header value of the line whose first interval is the
     *        x-th input interval
     * @param sentinel_header_value header value of the sentinel line and of the data lines after interval k'
     */
    template <typename p_fnc_t, typename idx_fnc_t, typename offs_fnc_t, typename extra_fnc_t, typename header_value_fnc_t>
    void fill(uint64_t num_threads, p_fnc_t p_of, idx_fnc_t idx_of, offs_fnc_t offs_of, extra_fnc_t extra_of,
        header_value_fnc_t header_value_of, uint64_t sentinel_header_value);

    /**
     * @brief sets the header value of the b-th line (without an inverse, a value the user of the lines writes);
     *        can be called by several threads at the same time for different lines
     * @param b line index
     * @param value value (must fit into omega_header_value bits)
     */
    inline void set_header_value(uint64_t b, uint64_t value) { overwrite_field_parallel(b, omega_line_start, omega_header_value, value); }

    /**
     * @brief sets the J-th extra row of the x-th input interval; can be called by several threads at the same
     *        time for different intervals
     * @param x [0..k'] input interval index
     * @param J [0..num_extra_rows-1] index of the extra row
     * @param value value to set (must fit into the width of the row)
     */
    inline void set_row(uint64_t x, uint8_t J, uint64_t value)
    {
        uint64_t b = line_of(x);
        overwrite_field_parallel(b, first_record_bit + (x - b * records_per_line) * bits_per_record + extra_bit[J], omega_extra[J], value);
    }

    /* Alternative to fill (see move_data_structure::construction::fill_lines): after allocate, the headers and
       the rel p fields are written with one thread per line (write_header, write_rel_p, write_row), and idx and
       offs are written in any order (by several threads with the methods ending with _parallel). The two kinds of
       writes must not be performed at the same time. */

    /**
     * @brief writes the header of the b-th line; no other thread may write to the line at the same time
     * @param b line index
     * @param start line start
     * @param header_value header value
     */
    inline void write_header(uint64_t b, uint64_t start, uint64_t header_value)
    {
        write_field(b, 0, omega_line_start, start);
        write_checked_field(b, omega_line_start, omega_header_value, header_value, "header value");
    }

    /**
     * @brief writes p(x) relative to the start of its line (nothing is written for record 0 of a line);
     *        no other thread may write to the line at the same time
     * @param x input interval index
     * @param rel_p p(x) minus the line start
     */
    inline void write_rel_p(uint64_t x, uint64_t rel_p)
    {
        uint64_t b = line_of(x);
        uint64_t j = x - b * records_per_line;
        if (j != 0) write_checked_field(b, first_record_bit + j * bits_per_record, omega_rel_p, rel_p, "relative p");
    }

    // returns a pointer to the word containing the first bit of idx(x) (for prefetching)
    inline const uint64_t* idx_word(uint64_t x) const
    {
        uint64_t b = line_of(x);
        return words.data() + ((bits_per_line * b + first_record_bit + (x - b * records_per_line) * bits_per_record + idx_bit) >> 6);
    }

    /**
     * @brief writes idx(x) and offs(x); both fields must be zero and no other thread may write to the lines
     *        at the same time
     * @param x input interval index
     * @param idx idx(x), as an interval index
     * @param offs offs(x)
     */
    inline void write_idx_offs(uint64_t x, uint64_t idx, uint64_t offs)
    {
        uint64_t b = line_of(x);
        uint64_t bit = first_record_bit + (x - b * records_per_line) * bits_per_record;
        write_checked_field(b, bit + idx_bit, omega_slot, slot_of(idx), "idx slot");
        write_checked_field(b, bit + offs_bit, omega_offs, offs, "offs");
    }

    /**
     * @brief writes idx(x) and offs(x); both fields must be zero. Other threads may write other records of the
     *        line at the same time
     * @param x input interval index
     * @param idx idx(x), as an interval index
     * @param offs offs(x)
     */
    inline void write_idx_offs_parallel(uint64_t x, uint64_t idx, uint64_t offs);

    /**
     * @brief writes the J-th extra row of the x-th input interval; the field must be zero and no other thread
     *        may write to the line at the same time
     * @param x input interval index
     * @param J [0..num_extra_rows-1] index of the extra row
     * @param value value to write
     */
    inline void write_row(uint64_t x, uint8_t J, uint64_t value)
    {
        uint64_t b = line_of(x);
        write_checked_field(b, first_record_bit + (x - b * records_per_line) * bits_per_record + extra_bit[J], omega_extra[J], value, "extra row");
    }

    /**
     * @brief writes idx(x), offs(x) and the J-th extra row of the x-th input interval; the fields must be zero.
     *        Other threads may write other records of the line at the same time
     * @param x input interval index
     * @param idx idx(x), as an interval index
     * @param offs offs(x)
     * @param J [0..num_extra_rows-1] index of the extra row
     * @param value value of the extra row
     */
    inline void write_idx_offs_row_parallel(uint64_t x, uint64_t idx, uint64_t offs, uint8_t J, uint64_t value);

    // writes the sentinel line; idx of each record is the sentinel interval k' and the header value is sentinel_header_value
    inline void write_sentinel(uint64_t sentinel_header_value)
    {
        write_field(num_lines, 0, omega_line_start, n);
        write_field(num_lines, omega_line_start, omega_header_value, sentinel_header_value);
        for (uint64_t t = 0; t < records_per_line; t++) write_field(num_lines, first_record_bit + t * bits_per_record + idx_bit, omega_slot, slot_of(k_));
    }

    // ############################# SERIALIZATION METHODS #############################

    /**
     * @brief writes the lines and the widths of their fields to an output stream
     * @param out output stream
     */
    void serialize_lines(std::ostream& out) const;

    /**
     * @brief reads the data written by serialize_lines
     * @param in input stream
     */
    void load_lines(std::istream& in);
};

#include "move_cache_lines.tpp"

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_CACHE_LINES_MOVE_CACHE_LINES_HPP
