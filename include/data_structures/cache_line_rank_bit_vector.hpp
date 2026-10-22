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

#ifndef MOVE_DATA_STRUCTURE_DATA_STRUCTURES_CACHE_LINE_RANK_BIT_VECTOR_HPP
#define MOVE_DATA_STRUCTURE_DATA_STRUCTURES_CACHE_LINE_RANK_BIT_VECTOR_HPP

#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

#include <misc/files.hpp>
#include <misc/utils.hpp>

/**
 * @brief bit vector with rank, stored in lines of one cache line (64 bytes) each: the first word of a line stores
 *        the number of set bits before the line (written by the caller with set_line_rank), the other seven words
 *        store 448 bits of the bit vector
 */
class cache_line_rank_bit_vector {
public:
    static constexpr uint64_t words_per_line = cache_aligned_no_init_allocator<uint64_t>::alignment / sizeof(uint64_t); // number of 64-bit words per line
    static constexpr uint64_t bits_per_line = 64 * (words_per_line - 1); // number of bits of the bit vector per line

protected:
    /* the lines: [0..words_per_line * (num_bits / bits_per_line + 1) - 1]; line l is words[8l..8l+7], words[8l] is
       the number of set bits before line l, and bit i of the bit vector is bit (i mod 448) mod 64 of
       words[8 (i / 448) + 1 + (i mod 448) / 64] */
    std::vector<uint64_t, cache_aligned_no_init_allocator<uint64_t>> words;
    uint64_t num_bits = 0; // number of bits

public:
    // returns the number of bits
    inline uint64_t size() const { return num_bits; }

    // returns the size of the lines in bytes
    inline uint64_t size_in_bytes() const { return words.size() * sizeof(uint64_t); }

    // returns a pointer to the line containing bit i
    inline const uint64_t* line(uint64_t i) const { return words.data() + words_per_line * (i / bits_per_line); }

    /**
     * @brief allocates the lines for a number of bits and sets them to zero
     * @param num_bits number of bits
     * @param num_threads number of threads to use
     */
    void allocate(uint64_t num_bits, uint64_t num_threads = 1)
    {
        this->num_bits = num_bits;
        uint64_t num_words = words_per_line * (num_bits / bits_per_line + 1);
        words.resize(num_words); // does not initialize the words (see cache_aligned_no_init_allocator)
        advise_huge_pages(words.data(), size_in_bytes());

        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t w = 0; w < num_words; w++) words[w] = 0;
    }

    // sets bit i; can be called by several threads at the same time
    inline void set_parallel(uint64_t i)
    {
        uint64_t r = i % bits_per_line;
        __atomic_fetch_or(&words[words_per_line * (i / bits_per_line) + 1 + r / 64],
            uint64_t { 1 } << (r % 64), __ATOMIC_RELAXED);
    }

    /**
     * @brief sets the rank of line l, i.e., the number of set bits before the line; no other thread may write
     *        the rank of the same line at the same time
     * @param l line index (bit i lies in line i / bits_per_line)
     * @param rank number of set bits before the line
     */
    inline void set_line_rank(uint64_t l, uint64_t rank) { words[words_per_line * l] = rank; }

    // returns bit i
    inline bool bit(uint64_t i) const
    {
        uint64_t r = i % bits_per_line;
        return (line(i)[1 + r / 64] >> (r % 64)) & 1;
    }

    // returns the number of set bits before bit i
    inline uint64_t rank(uint64_t i) const
    {
        const uint64_t* L = line(i);
        const int64_t r = int64_t(i % bits_per_line);
        uint64_t count = L[0];

        // word j contributes its bits before bit r: all of them, none, or the lowest r - 64j
        for (uint64_t j = 0; j + 1 < words_per_line; j++) {
            int64_t d = r - int64_t(64 * j);
            uint64_t mask = d <= 0 ? 0 : low_bits_mask(uint64_t(d));
            count += uint64_t(std::popcount(L[1 + j] & mask));
        }

        return count;
    }

    // writes the lines to an output stream
    void serialize(std::ostream& out) const
    {
        out.write((char*) &num_bits, sizeof(uint64_t));
        uint64_t num_words = words.size();
        out.write((char*) &num_words, sizeof(uint64_t));
        write_in_chunks(out, (const char*) words.data(), num_words * sizeof(uint64_t));
    }

    // reads the data written by serialize
    void load(std::istream& in)
    {
        in.read((char*) &num_bits, sizeof(uint64_t));
        uint64_t num_words;
        in.read((char*) &num_words, sizeof(uint64_t));
        words.resize(num_words);
        advise_huge_pages(words.data(), size_in_bytes());
        read_in_chunks(in, (char*) words.data(), num_words * sizeof(uint64_t));
    }
};

#endif // MOVE_DATA_STRUCTURE_DATA_STRUCTURES_CACHE_LINE_RANK_BIT_VECTOR_HPP
