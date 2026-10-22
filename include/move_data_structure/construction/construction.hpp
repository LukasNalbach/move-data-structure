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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_HPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_HPP

#include <move_data_structure/move_data_structure.hpp>

#include <sux/support/common.hpp>
#include <data_structures/interleaved_bit_aligned_vectors.hpp>
#include <misc/files.hpp>
#include <misc/log.hpp>
#include <misc/sort.hpp>
#include <misc/utils.hpp>

#include <omp.h>


// a split of the pair (p_x, q_x) at offset o, i.e., of its input interval at p_x + o and of its output
// interval at q_x + o
struct mds_split_t {
    uint64_t x; // index of the pair (in the order of p)
    uint64_t y; // rank of q_x among the output interval starting positions
    uint64_t o; // offset of the split
};

// radix sort key of a split: x
struct mds_split_key {
    inline uint64_t operator()(const mds_split_t& s) const { return s.x; }
};

// a piece to check in the next round of the balancing
struct mds_piece_t {
    uint64_t interval; // index of the pair (input interval piece) or rank of the output interval (output interval piece)
    uint64_t offset; // offset of the piece in its interval
};

// radix sort key of a piece: interval
struct mds_piece_key {
    inline uint64_t operator()(const mds_piece_t& c) const { return c.interval; }
};

/**
 * @brief sorts v by (key(e), second(e)) and removes duplicates: radix sort by key, then each run of elements
 *        with equal keys is sorted by second
 * @tparam elem_t type of the elements
 * @tparam key_t type of key
 * @tparam second_fnc_t type of second
 * @param v the elements
 * @param num_threads the number of threads to use
 * @param key function returning the first sort key of an element
 * @param second function returning the second sort key of an element
 */
template <typename elem_t, typename key_t, typename second_fnc_t>
inline void mds_sort_unique(no_init_vector<elem_t>& v, uint64_t num_threads, key_t key, second_fnc_t second);

// second sort key of a split in mds_sort_unique: o
struct mds_split_offset {
    inline uint64_t operator()(const mds_split_t& s) const { return s.o; }
};

// second sort key of a piece in mds_sort_unique: offset
struct mds_piece_offset {
    inline uint64_t operator()(const mds_piece_t& c) const { return c.offset; }
};

/**
 * @brief computes the number of ones before each word of a bit vector, in parallel
 * @param words [0..num_words-1] the words of the bit vector
 * @param ranks is resized to num_words + 1; ranks[w] is set to the number of ones in words[0..w-1]
 * @param num_threads the number of threads to use
 */
inline void mds_build_word_ranks(const no_init_vector<uint64_t>& words, no_init_vector<uint64_t>& ranks, uint64_t num_threads);

/**
 * @brief Elias-Fano coded increasing sequence of num_values values in [0, universe) with rank, select and a cursor
 *        to iterate over the values in ascending order. The value v_i is split into its low_width low bits, stored
 *        in lower, and its high part v_i >> low_width, stored as a one at position (v_i >> low_width) + i of upper.
 */
class mds_elias_fano {
protected:
    // rank and select scan up to this many words or values linearly before they switch to a binary search
    static constexpr uint64_t linear_scan_length = cache_line_bytes / sizeof(uint64_t); // one cache line

    uint8_t low_width = 0; // number of bits per low part
    uint64_t num_values = 0; // number of values
    uint64_t universe = 0; // exclusive upper bound of the values
    // the low parts: [0..(num_values * low_width) / 64 + 1] value i's low part is bits [i * low_width, (i + 1) * low_width)
    no_init_vector<uint64_t> lower;
    // the high parts in unary coding: [0..(num_values + (universe >> low_width) + 1) / 64 + 1] bit (v_i >> low_width) + i
    // is set for each value i
    no_init_vector<uint64_t> upper;
    // [0..upper.size()] ones_before_word[w] = number of ones in upper[0..w-1]; the last entry is num_values
    no_init_vector<uint64_t> ones_before_word;
    // [0..num_values / 64 + 1] select1_samples[j] = index of the word of upper containing the (64 j)-th one (0-based);
    // the entries after the last sample are the index of the last word
    no_init_vector<uint64_t> select1_samples;
    // [0..(number of zeros in upper) / 64 + 1] the same for the zeros of upper
    no_init_vector<uint64_t> select0_samples;

    // returns the low part of the i-th value
    inline uint64_t low(uint64_t i) const;

    // returns the number of zeros in upper before the w-th word
    inline uint64_t zeros_before(uint64_t w) const { return 64 * w - ones_before_word[w]; }

    // returns the position of the j-th one in upper (0-based); its word lies between two samples and is found with
    // a linear scan (at most linear_scan_length words) or a binary search over ones_before_word
    inline uint64_t select1(uint64_t j) const;

    // returns the position of the j-th zero in upper (see select1)
    inline uint64_t select0(uint64_t j) const;

public:
    mds_elias_fano() = default;

    /**
     * @brief builds the sequence
     * @tparam read_fnc_t type of read
     * @param read function returning the i-th value for i in [0..num_values-1] (increasing in i); called by
     *        multiple threads
     * @param num_values number of values
     * @param universe_ exclusive upper bound of the values
     * @param num_threads the number of threads to use
     */
    template <typename read_fnc_t>
    mds_elias_fano(read_fnc_t read, uint64_t num_values, uint64_t universe_, uint64_t num_threads);

    // returns the i-th value (i in [0..num_values-1], 0-based)
    inline uint64_t select(uint64_t i) const { return ((select1(i) - i) << low_width) | low(i); }

    // returns the number of values smaller than v
    inline uint64_t rank(uint64_t v) const;

    // iterates over the values in ascending order
    struct cursor {
        const mds_elias_fano* ef;
        uint64_t i; // index of the current value
        uint64_t w; // index of the word of upper containing the one of the current value
        uint64_t word; // upper[w] without the ones of the previous values

        // returns the current value
        inline uint64_t value() const { return ((64 * w + std::countr_zero(word) - i) << ef->low_width) | ef->low(i); }

        // advances to the next value (the current value must not be the last one)
        inline void next()
        {
            word &= word - 1;
            i++;
            while (word == 0) word = ef->upper[++w];
        }
    };

    // returns a cursor at the i-th value (i in [0..num_values-1])
    inline cursor cursor_at(uint64_t i) const
    {
        uint64_t pos = select1(i);
        return { this, i, pos >> 6, upper[pos >> 6] & (~uint64_t { 0 } << (pos & 63)) };
    }
};

/**
 * @brief bit vector with rank and select that can be set and built in parallel; marks the first piece of each
 *        pair among the pieces of all pairs (see move_data_structure::construction::mark_pieces)
 */
class mds_piece_marks {
protected:
    no_init_vector<uint64_t> bits; // the bit vector: [0..size / 64 + 1], bit i is bit i mod 64 of bits[i / 64]; one word more than needed
    no_init_vector<uint64_t> ones_before_word; // [0..bits.size()] ones_before_word[w] = number of ones in bits[0..w-1]
    no_init_vector<uint64_t> select_samples; // [0..(number of ones) / 64] select_samples[j] = index of the word containing the (64 j)-th one (0-based)

public:
    /**
     * @brief resizes the bit vector to size bits and sets all bits to zero
     * @param size number of bits
     * @param num_threads the number of threads to use
     */
    void resize(uint64_t size, uint64_t num_threads)
    {
        resize_no_init(bits, size / 64 + 2);

        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t w = 0; w < bits.size(); w++) bits[w] = 0;
    }

    // sets bit i; can be called by multiple threads
    inline void set_parallel(uint64_t i) { __atomic_fetch_or(&bits[i >> 6], uint64_t { 1 } << (i & 63), __ATOMIC_RELAXED); }

    /**
     * @brief builds the rank and the select support
     * @param num_threads the number of threads to use
     */
    void build(uint64_t num_threads);

    // returns the number of ones in [0, i)
    inline uint64_t rank(uint64_t i) const
    {
        return ones_before_word[i >> 6] + std::popcount(bits[i >> 6] & ((uint64_t { 1 } << (i & 63)) - 1));
    }

    // returns the position of the r-th one (r starts at 0)
    inline uint64_t select(uint64_t r) const
    {
        uint64_t w = select_samples[r >> 6];
        while (ones_before_word[w + 1] <= r) w++;
        return (w << 6) + sux::select64(bits[w], r - ones_before_word[w]);
    }

    // frees the bit vector
    void clear()
    {
        no_init_vector<uint64_t>().swap(bits);
        no_init_vector<uint64_t>().swap(ones_before_word);
        no_init_vector<uint64_t>().swap(select_samples);
    }
};

/**
 * @brief queue of writes; a write is executed when depth further writes have been queued (or by flush()), and
 *        the word it writes to is prefetched when it is queued
 * @tparam write_fnc_t type of the write function, called as write(a, b, c, d) with the values of a queued write
 */
template <typename write_fnc_t>
struct mds_deferred_writes {
    static constexpr uint64_t depth = 32; // maximum number of queued writes

    write_fnc_t write;
    // the last depth queued writes (a, b, c, d): [0..depth-1] the write number t is stored at t mod depth
    std::array<std::array<uint64_t, 4>, depth> queue {};
    uint64_t num_queued = 0; // number of writes queued so far

    // queues the write (a, b, c, d) and prefetches word, the word it writes to
    inline void push(const void* word, uint64_t a, uint64_t b, uint64_t c, uint64_t d = 0);

    // executes the queued writes
    inline void flush();
};

/**
 * @brief constructs a move data structure out of a disjoint interval sequence I of k pairs (p_x, q_x)
 *
 * Splitting the pair (p_x, q_x) at an offset o splits its input interval at p_x + o and its output interval at
 * q_x + o. The intervals resulting from the splits are called pieces; x' and y' denote the rank of a piece among
 * the input and among the output interval pieces (the indices of the input and output intervals of the move data
 * structure). Every interval is split at the multiples of l_max; the construction stores the other split offsets
 * of each pair in splits.
 *
 * 1. order_pairs(): sorts the pairs by q and builds P and Q
 * 2. balance(): splits the a-heavy intervals in rounds; the first round checks all pieces, each further round
 *    the pieces containing a starting position that was added in the previous round
 * 3. mark_pieces(): computes k' and marks the first piece of each pair
 * 4. fill_blocks() (POS) / fill_ef() (EF): writes the pieces to the move data structure
 */
template <move_pos_encoding_t enc, move_mode_t mode>
class move_data_structure<enc, mode>::construction {

public:
    construction() = delete;
    construction(construction&&) = delete;
    construction(const construction&) = delete;
    construction& operator=(construction&&) = delete;
    construction& operator=(const construction&) = delete;

    // ############################# TYPES AND PARAMETERS #############################

    using pair_arr_t = mds_pairs;

    using split_t = mds_split_t;
    using piece_t = mds_piece_t;

    // the split offsets of the pairs found by the balancing; the splits at the multiples of l_max are not stored
    // (pieces_t adds them)
    struct split_lists {
        // [0..k/64] bit x is set <=> the pair x has stored split offsets
        no_init_vector<uint64_t> has_splits_p;
        // [0..k/64+1] rank_has_splits_p[w] = number of set bits in has_splits_p[0..w-1]; the index in pairs of a
        // pair x with split offsets is rank_has_splits_p[x / 64] plus the number of set bits before bit x in its word
        no_init_vector<uint64_t> rank_has_splits_p;
        // [0..k/64] bit y is set <=> the pair of the y-th output interval (in the order of q) has stored split offsets
        no_init_vector<uint64_t> has_splits_q;
        // [0..s-1] the indices x of the s pairs with stored split offsets, increasing
        no_init_vector<uint40_t> pairs;
        // [0..s-1] pairs_y[i] = rank of q_{pairs[i]} among the output interval starting positions
        no_init_vector<uint40_t> pairs_y;
        // [0..s] the split offsets of the pair pairs[i] are offsets[offsets_begin[i]..offsets_begin[i+1]-1];
        // offsets_begin[s] = offsets.size()
        no_init_vector<uint40_t> offsets_begin;
        // all stored split offsets, grouped by pair in the order of pairs, increasing within a pair; values < n
        uint_bytes_array<1> offsets;
        uint64_t n = 0; // the split offsets are smaller than n

        // a pair with more than max_short_splits split offsets is a long pair; for a long pair, the number of
        // split offsets up to an offset is computed with rank on split_pos_p or split_pos_q (see
        // construction::num_splits_leq)
        static constexpr uint64_t max_short_splits = 32;
        // [0..s] long_before_p[i] = number of split offsets of the long pairs among pairs[0..i-1]; entry s is the total
        no_init_vector<uint40_t> long_before_p;
        // [0..s] long_before_q[i] = number of split offsets of the long pairs whose output intervals start before
        // the one of pairs[i] (only set if pairs[i] is a long pair)
        no_init_vector<uint40_t> long_before_q;
        mds_elias_fano split_pos_p; // p_x + o for each split (x, o) of a long pair, in ascending order
        mds_elias_fano split_pos_q; // q_x + o for each split (x, o) of a long pair, in ascending order

        /**
         * @brief initializes the lists for k pairs without split offsets
         * @param k the number of pairs
         * @param n_ the split offsets are smaller than n_
         */
        void init(uint64_t k, uint64_t n_);

        // returns whether the pair of the y-th output interval has split offsets
        inline bool output_has_splits(uint64_t y) const { return (has_splits_q[y >> 6] >> (y & 63)) & 1; }

        // returns the range [first, last) of the indices in offsets of the split offsets of the pair x
        inline std::pair<uint64_t, uint64_t> splits_of(uint64_t x) const
        {
            uint64_t word = has_splits_p[x >> 6];
            if (!((word >> (x & 63)) & 1)) return { 0, 0 };
            uint64_t i = rank_has_splits_p[x >> 6] + std::popcount(word & ((uint64_t { 1 } << (x & 63)) - 1));
            return { uint64_t(offsets_begin[i]), uint64_t(offsets_begin[i + 1]) };
        }

        // returns the index in pairs of the pair x (x must have split offsets)
        inline uint64_t index_of(uint64_t x) const
        {
            return rank_has_splits_p[x >> 6] + std::popcount(has_splits_p[x >> 6] & ((uint64_t { 1 } << (x & 63)) - 1));
        }

        /**
         * @brief inserts the splits found in a round of the balancing
         * @param new_splits the splits, sorted by (x, o), without duplicates
         * @param num_threads the number of threads to use
         */
        void merge(const no_init_vector<split_t>& new_splits, uint64_t num_threads);

        // returns the number of stored split offsets
        inline uint64_t size() const { return offsets.size(); }

        // frees the lists
        void clear();
    };

    // iterates over the pieces of a pair, i.e., its interval split at the multiples of l_max and at its stored
    // split offsets
    struct pieces_t {
        uint64_t len; // length of the interval of the pair
        uint64_t l_max; // maximum interval length
        uint64_t next_multiple; // the smallest multiple of l_max greater than beg
        const uint_bytes_array<1>* offsets; // the stored split offsets (see split_lists::offsets)
        uint64_t next_split; // index in offsets of the first stored split offset of the pair greater than beg
        uint64_t first_split; // index in offsets of the first stored split offset of the pair
        uint64_t end_split; // index in offsets after the last stored split offset of the pair
        uint64_t beg; // offset of the current piece
        uint64_t end; // offset of the next piece (len, if the current piece is the last one)
        uint64_t piece_idx; // index of the current piece within the pair

        // returns the offset of the piece after the piece starting at beg (len, if there is none)
        inline uint64_t next_piece_offset() const
        {
            uint64_t e = next_multiple < len ? next_multiple : len;
            if (next_split != end_split && offsets->get(next_split) < e) e = offsets->get(next_split);
            return e;
        }

        /**
         * @brief moves to the first piece of a pair
         * @param len_ length of the interval of the pair
         * @param l_max_ maximum interval length
         * @param offsets_ the stored split offsets of all pairs (see split_lists::offsets)
         * @param splits the range of the indices in offsets_ of the split offsets of the pair (see split_lists::splits_of)
         */
        inline void init(uint64_t len_, uint64_t l_max_, const uint_bytes_array<1>& offsets_,
            std::pair<uint64_t, uint64_t> splits);

        // advances to the next piece; returns false if the current piece is the last one of the pair
        inline bool next();

        /**
         * @brief moves to the piece containing the offset d
         * @param d [0..len-1] offset in the interval of the pair
         * @param num_splits number of stored split offsets <= d (see construction::num_splits_leq)
         */
        inline void seek(uint64_t d, uint64_t num_splits);

        // advances to the piece with the index idx within the pair
        inline void seek_index(uint64_t idx)
        {
            while (piece_idx < idx) next();
        }
    };

    // iterates over the pieces of the input intervals in ascending order of their starting positions
    struct input_cursor {
        const construction* c;
        uint64_t x; // index of the current pair
        uint64_t p; // p_x
        uint64_t p_next; // p_{x+1} (n for x = k - 1)
        pieces_t pieces; // the pieces of the pair x
        mds_elias_fano::cursor p_cursor; // cursor in P at p_{x+1} (if x + 1 < k)

        // moves to the first piece of the pair x_
        inline void go_to_pair(uint64_t x_);

        // moves to the first piece of the next pair
        inline void go_to_next_pair();

        // moves to the piece containing position i
        inline void seek(uint64_t i)
        {
            go_to_pair(c->pair_containing(i));
            pieces.seek(i - p, c->num_splits_leq(pieces, [&]() { return c->splits.split_pos_p.rank(i + 1) - c->splits.long_before_p[c->splits.index_of(x)]; }, i - p));
        }

        // moves to the piece of the pair x_ that contains the offset offset
        inline void seek(uint64_t x_, uint64_t offset)
        {
            go_to_pair(x_);
            pieces.seek(offset, c->num_splits_leq(pieces, [&]() { return c->splits.split_pos_p.rank(p + offset + 1) - c->splits.long_before_p[c->splits.index_of(x)]; }, offset));
        }

        // advances to the next piece
        inline void next()
        {
            if (!pieces.next()) go_to_next_pair();
        }

        // returns the starting position of the current piece
        inline uint64_t beg() const { return p + pieces.beg; }

        // returns the starting position of the next piece
        inline uint64_t end() const { return p + pieces.end; }
    };

    // iterates over the pieces of the output intervals in ascending order of their starting positions
    struct output_cursor {
        const construction* c;
        uint64_t y; // rank of the current output interval
        uint64_t q; // starting position of the y-th output interval
        uint64_t q_next; // starting position of the (y+1)-th output interval
        uint64_t pair; // index of the pair of the y-th output interval (only set if the pair has split offsets)
        pieces_t pieces; // the pieces of the y-th output interval

        // moves to the first piece of the y_-th output interval
        inline void go_to_output(uint64_t y_);

        // moves to the piece containing position i
        inline void seek(uint64_t i)
        {
            seek(c->output_containing(i), i);
        }

        // moves to the piece containing position i, which lies in the y_-th output interval
        inline void seek(uint64_t y_, uint64_t i)
        {
            go_to_output(y_);
            pieces.seek(i - q, c->num_splits_leq(pieces, [&]() { return c->splits.split_pos_q.rank(i + 1) - c->splits.long_before_q[c->splits.index_of(pair)]; }, i - q));
        }

        // advances to the next piece
        inline void next()
        {
            if (!pieces.next()) go_to_output(y + 1);
        }

        // returns the starting position of the current piece
        inline uint64_t beg() const { return q + pieces.beg; }

        // returns the starting position of the next piece
        inline uint64_t end() const { return q + pieces.end; }
    };

    // ############################# VARIABLES #############################

    move_data_structure& mds; // the move data structure to construct
    row_widths_t row_widths; // widths (in bits) of the extra rows
    row_widths_t row_widths_inv; // widths (in bits) of the extra rows of the output intervals
    // [0..k'] vector to store pi in (pi_out[y] = pi(y), pi_out[k'] = k'), if not nullptr
    no_init_vector<uint40_t>* pi_out = nullptr;
    pair_arr_t& I; // the disjoint interval sequence to construct the move data structure out of
    uint64_t n; // the size of the domain, p_{k-1} plus the length of the last interval of I
    uint64_t k; // number of pairs of I, 0 < k
    uint64_t k_; // k', the number of intervals in the a-balanced interval sequence B_a(I), k <= k'
    uint64_t a; // balancing parameter a >= 2 (see mds_params::a)
    uint64_t num_threads; // number of threads to use
    uint64_t chunks_per_thread; // number of chunks per thread of the loops with a dynamic schedule (see mds_params)
    bool log; // toggles log messages
    uint64_t max_interval_length; // maximum interval length of I, if I is already balanced, else 0 (see mds_params)
    bool delete_I; // true <=> I is sorted in place and freed (see mds_params)
    const uint8_t* q_order_keys; // [0..k-1] counting sort keys of the pairs, or nullptr (see mds_params)
    // [0..max_extra_rows-1] the extra rows filled by the construction (see mds_pair_row); values = nullptr for the other rows
    std::array<mds_pair_row, max_extra_rows> pair_rows {};
    std::array<uint8_t, max_extra_rows> rows_in_p_order {}; // [0..num_rows_in_p_order-1] indices J of the rows given in the order of p
    uint8_t num_rows_in_p_order = 0; // number of rows given in the order of p
    uint8_t row_in_q_order = max_extra_rows; // index J of the row given in the order of q; max_extra_rows <=> there is none
    mds_piece_marks* first_pieces = nullptr; // if not nullptr, first_p is moved to it (see mds_params)
    std::ostream* mf; // measurement file
    uint64_t l_max; // maximum interval length, the pairs are split at the multiples of l_max
    uint64_t max_piece_length = 0; // maximum length of a piece
    std::chrono::steady_clock::time_point time; // time point of the start of the last construction phase
    std::chrono::steady_clock::time_point time_start; // time point of the start of the entire construction

    pair_arr_t I_copy; // stores pairs_by_q, if I is not sorted in place
    // [0..k-1] the pairs (x, q_x), sorted by q_x: field 0 of entry y is the index x of the pair of the y-th output
    // interval, field 1 its q_x; points to I or to I_copy
    pair_arr_t* pairs_by_q = nullptr;
    mds_elias_fano P; // [0..k-1] the input interval starting positions p_0 < p_1 < ... < p_{k-1}
    mds_elias_fano Q; // [0..k-1] the output interval starting positions in ascending order, Q[y] = q_{pairs_by_q[y]}
    split_lists splits; // the split offsets of the pairs
    // (has_inverse) [0..k-1] rho[x] = rank of q_x among the output interval starting positions
    uint_bytes_array<1> rho;
    // [0..k'] bit x' is set <=> the x'-th input interval piece is the first piece of its pair; bit k' is set
    mds_piece_marks first_p;
    // [0..k'] bit y' is set <=> the y'-th output interval piece is the first piece of its output interval; bit k' is set
    mds_piece_marks first_q;
    uint64_t num_rounds = 0; // number of rounds of the balancing

    // (name, time in ns) of the steps of the current phase; the times of steps with the same name are added up
    std::vector<std::pair<std::string, uint64_t>> step_times;
    std::chrono::steady_clock::time_point step_start; // time point of the start of the current step

    // ############################# INTERNAL METHODS #############################

    // returns the chunk length of a parallel loop over num_items items with a dynamic schedule, such that there are
    // about chunks_per_thread chunks per thread
    inline int dynamic_chunk_length(uint64_t num_items) const
    {
        return int(std::clamp<uint64_t>(num_items / (chunks_per_thread * num_threads), 1, std::numeric_limits<int>::max()));
    }

    // starts timing a step
    inline void step_begin() { if (mf != nullptr) step_start = now(); }

    // ends the current step and adds its time to step_times
    inline void step_end(const std::string& name)
    {
        if (mf == nullptr) return;
        uint64_t ns = time_diff_ns(step_start, now());
        for (auto& [n_, t] : step_times) if (n_ == name) { t += ns; return; }
        step_times.emplace_back(name, ns);
    }

    // writes step_times to the measurement file (as name=time) and clears it
    inline void write_steps()
    {
        if (mf == nullptr) return;
        for (auto& [n_, t] : step_times) *mf << " " << n_ << "=" << t;
        step_times.clear();
    }

    // ############################# CONSTRUCTOR #############################

    /**
     * @brief builds the move data structure mds
     * @param mds the move data structure to build
     * @param I disjoint interval sequence
     * @param n the size of the domain, p_{k-1} plus the length of the last interval of I
     * @param params construction parameters
     * @param row_widths widths (in bits) of the extra rows
     * @param row_widths_inv widths (in bits) of the extra rows of the output intervals
     * @param pi_out vector to store pi in, if not nullptr
     */
    construction(
        move_data_structure& mds,
        pair_arr_t& I,
        uint64_t n,
        mds_params params,
        row_widths_t row_widths = {},
        row_widths_t row_widths_inv = {},
        no_init_vector<uint40_t>* pi_out = nullptr);

    // ############################# ACCESS #############################

    // returns p_x for x in [0..k] (p_k = n)
    inline uint64_t p_of(uint64_t x) const { return x < k ? P.select(x) : n; }

    // returns the starting position of the y-th output interval for y in [0..k] (n for y = k)
    inline uint64_t q_of_output(uint64_t y) const { return y < k ? pairs_by_q->template get<1>(y) : n; }

    // returns the index of the pair of the y-th output interval
    inline uint64_t pair_of_output(uint64_t y) const { return pairs_by_q->template get<0>(y); }

    // returns the index of the pair whose input interval contains position i
    inline uint64_t pair_containing(uint64_t i) const { return P.rank(i + 1) - 1; }

    // returns the rank of the output interval containing position i
    inline uint64_t output_containing(uint64_t i) const { return Q.rank(i + 1) - 1; }

    /**
     * @brief returns the number of stored split offsets <= d of a pair; computed with a binary search, if the
     *        pair has at most split_lists::max_short_splits split offsets, else with rank_fnc()
     * @tparam rank_fnc_t type of rank_fnc
     * @param pieces the pieces of the pair
     * @param rank_fnc function computing the result with rank (see split_lists::split_pos_p)
     * @param d offset in the interval of the pair
     */
    template <typename rank_fnc_t>
    static inline uint64_t num_splits_leq(const pieces_t& pieces, rank_fnc_t rank_fnc, uint64_t d)
    {
        uint64_t num = pieces.end_split - pieces.first_split;
        if (num > split_lists::max_short_splits) return rank_fnc();
        uint64_t lo = pieces.first_split, hi = pieces.end_split; // the first index with a split offset > d is in [lo, hi]

        while (lo < hi) {
            const uint64_t mid = (lo + hi) / 2;
            if (pieces.offsets->get(mid) <= d) lo = mid + 1;
            else hi = mid;
        }

        return lo - pieces.first_split;
    }

    // returns rho[x], the rank of the output interval of the pair x
    inline uint64_t rho_of(uint64_t x) const { return rho.get(x); }

    /**
     * @brief returns the value of the J-th extra row of a piece (see mds_pair_row)
     * @param J row index
     * @param rank rank of the pair of the piece, in the order of the values of the row (by p or by q)
     * @param d offset of the piece in the interval of its pair
     */
    inline uint64_t pair_row_value(uint8_t J, uint64_t rank, uint64_t d) const
    {
        uint64_t value = pair_rows[J].values->template get<0, uint64_t>(rank);
        return pair_rows[J].subtract_offset ? value - d : value;
    }

    // ############################# PHASES #############################

    /**
     * @brief chooses the width of D_offs (the maximum interval length is l_max = 2^omega_offs - 1): the default
     *        width, or a width of up to params.max_omega_offs_widening bits more, if that reduces the number of
     *        blocks by more than params.min_block_reduction_percent % and does not increase the cost of reading a
     *        record (see move_input_interval_blocks::record_read_cost)
     * @param omega_default the default width, the bit width of params.max_length_factor * n / k
     * @param params the construction parameters
     * @return the chosen width
     */
    uint8_t choose_omega_offs(uint8_t omega_default, const mds_params& params) requires(enc == POS);

    /**
     * @brief sorts the pairs by q (in place, if delete_I is set) and builds P, Q and (if has_inverse) rho
     */
    void order_pairs();

    /**
     * @brief builds split_pos_p, split_pos_q, long_before_p and long_before_q of splits (see split_lists) from the
     *        stored split offsets
     */
    void index_splits();

    /**
     * @brief splits a-heavy intervals until the interval sequence is a-balanced, in rounds: the first round
     *        checks all pieces, each further round checks the pieces containing a starting position that was
     *        added by a split of the previous round
     */
    void balance();

    /**
     * @brief counts the starting positions of the pieces of other in the piece cursor points to and stores the
     *        (a+1)-th, (2a+1)-th, ... of them in split_pos
     * @param cursor cursor at the piece (input or output interval piece)
     * @param other cursor at a piece of the other side that starts at or before the starting position of the
     *        piece; it is advanced to the piece containing the last position of the piece
     * @param split_pos is set to the (a+1)-th, (2a+1)-th, ... of the counted starting positions
     * @return the number of counted starting positions
     */
    template <typename cursor_t, typename other_cursor_t>
    inline uint64_t count_other_starts(const cursor_t& cursor, other_cursor_t& other, no_init_vector<uint64_t>& split_pos) const;

    /**
     * @brief counts the input interval starting positions in the output interval piece oc points to; if
     *        there are at least 2a, splits the piece at the (a+1)-th, (2a+1)-th, ... of them
     * @param oc cursor at the output interval piece
     * @param ic cursor at an input interval piece that starts at or before the starting position of the output
     *        interval piece; it is advanced to the piece containing the last position of the output interval piece
     * @param found vector to append the splits to
     * @param split_pos buffer for the split positions
     */
    inline void split_heavy_output_piece(const output_cursor& oc, input_cursor& ic, std::vector<split_t>& found, no_init_vector<uint64_t>& split_pos) const;

    /**
     * @brief counts the output interval starting positions in the input interval piece ic points to; if
     *        there are at least 2a, splits the piece at the (a+1)-th, (2a+1)-th, ... of them
     * @param ic cursor at the input interval piece
     * @param oc cursor at an output interval piece that starts at or before the starting position of the input
     *        interval piece; it is advanced to the piece containing the last position of the input interval piece
     * @param found vector to append the splits to
     * @param split_pos buffer for the split positions
     */
    inline void split_heavy_input_piece(const input_cursor& ic, output_cursor& oc, std::vector<split_t>& found, no_init_vector<uint64_t>& split_pos) const;

    /**
     * @brief builds first_p and first_q and computes k' and max_piece_length
     */
    void mark_pieces();

    /**
     * @brief builds the blocks of a positional move data structure
     */
    void fill_blocks() requires(enc == POS);

    /**
     * @brief builds data and high_part_increments of a move data structure in the encoding EF
     */
    void fill_ef() requires(enc == EF);

    /**
     * @brief iterates over the output interval pieces with the ranks [y_beg, y_end) and calls f for each of them
     * @tparam fnc_t type of f
     * @param y_beg rank of the first output interval piece
     * @param y_end rank of the last output interval piece + 1
     * @param f function called as f(y', x', t, idx, offs, y, d) with
     *        - y', x': the ranks of the piece among the output and among the input interval pieces
     *        - t: its starting position
     *        - idx, offs: the rank of the input interval piece containing t and the offset of t in it
     *        - y: the rank of the output interval of the piece (the rank of q of its pair)
     *        - d: the offset of t in that output interval
     */
    template <typename fnc_t>
    void for_each_output_piece(uint64_t y_beg, uint64_t y_end, fnc_t f) const;

    /**
     * @brief iterates over the input interval pieces with the ranks [x_beg, x_end) and calls f for each of them
     * @tparam fnc_t type of f
     * @param x_beg rank of the first input interval piece
     * @param x_end rank of the last input interval piece + 1
     * @param y_start (if has_inverse) rank of the output interval containing the starting position of the
     *        x_beg-th input interval piece
     * @param f function called as f(x', s, len, y', e, offs, x, d) with
     *        - x': the rank of the piece
     *        - s, len: its starting position and its length
     *        - y': the rank of the piece among the output interval pieces (0 without an inverse)
     *        - e, offs: the rank of the output interval piece containing s and the offset of s in it (0 without
     *          an inverse)
     *        - x: the index of the pair of the piece
     *        - d: the offset of s in the input interval of x
     */
    template <typename fnc_t>
    void for_each_input_piece(uint64_t x_beg, uint64_t x_end, uint64_t y_start, fnc_t f) const;

    // returns the starting position of the x_-th input interval piece
    uint64_t start_of_input_piece(uint64_t x_) const;
};

#include "helpers.tpp"
#include "construction.tpp"
#include "split.tpp"
#include "balance.tpp"
#include "fill.tpp"

// explicit instantiations of a radix sort (see misc/sort.hpp) for the iterator type it_t and the key type key_t
#define MOVE_DATA_STRUCTURE_RADIX_SORT(prefix, it_t, key_t) \
    prefix template void radix_sort_parallel<it_t, key_t>(it_t, it_t, key_t, uint64_t); \
    prefix template void radix_sort_sequential<it_t, key_t>(it_t, it_t, key_t);

// explicit instantiations of the sorts of the construction; prefix is empty in src/instantiations.cpp (where they are
// compiled) and extern everywhere else
#define MOVE_DATA_STRUCTURE_SORTS(prefix) \
    prefix template void uint_bytes_array<2>::sort_by<0>(uint64_t); \
    prefix template void uint_bytes_array<2>::sort_by<1>(uint64_t); \
    MOVE_DATA_STRUCTURE_RADIX_SORT(prefix, no_init_vector<uint40_t>::iterator, key_of_index<uint40_t>) \
    MOVE_DATA_STRUCTURE_RADIX_SORT(prefix, no_init_vector<mds_split_t>::iterator, mds_split_key) \
    MOVE_DATA_STRUCTURE_RADIX_SORT(prefix, no_init_vector<mds_piece_t>::iterator, mds_piece_key)

// the editor parser of VS Code (__INTELLISENSE__) does not resolve the extern declarations of constrained member
// templates; it does not need them
#if !defined(STRING_UTILS_NO_EXTERN_TEMPLATES) && !defined(__INTELLISENSE__)
MOVE_DATA_STRUCTURE_SORTS(extern)
#endif

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_HPP
