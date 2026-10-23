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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_HPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_HPP

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <data_structures/interleaved_bit_aligned_vectors.hpp>
#include <data_structures/uint_bytes_array.hpp>
#include <misc/files.hpp>
#include <misc/search.hpp>
#include <misc/utils.hpp>
#include <data_structures/cache_line_rank_bit_vector.hpp>
#include <move_data_structure/move_interval_blocks/move_interval_blocks.hpp>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>


/**
 * @brief encoding of the interval starting positions p(x) of a move data structure
 */
enum move_pos_encoding_t : uint8_t {
    // stores p(x) in omega_p = bit_width(n) bits per input interval, in the blocks of move_input_interval_blocks;
    // p(x) in O(1) time
    POS = 0,
    // stores D_low[x] = p(x) mod 2^omega_offs in omega_offs bits per input interval and the high parts
    // p(x) >> omega_offs as one bit per interval in D_high; p(x) in O(1) time; besides move on
    // positions, it supports move_offset on (interval, offset)-pairs, which does not read D_high;
    // only for the mode UNI
    EF = 1
};

/**
 * @brief direction(s) in which a move data structure supports move queries
 *
 * The input intervals [p(x), p(x+1)) partition [0, n), and so do the output intervals [p_inv(y), p_inv(y+1)),
 * i.e., the images of the input intervals ordered by their starting positions. pi(y) is the index of the input
 * interval whose image is the y-th output interval.
 */
enum move_mode_t : uint8_t {
    // supports move
    UNI = 0,
    // supports move and move_inv; the output intervals are stored like the input intervals, as a move data
    // structure for the inverse permutation
    BI = 1
};

// the pairs (p_i, q_i) of a disjoint interval sequence: p_i is the field 0 and q_i the field 1 of the i-th group
// (see uint_bytes_array)
using mds_pairs = uint_bytes_array<2>;

/**
 * @brief returns the pairs (p_i, q_i) of a disjoint interval sequence of [0, n) as mds_pairs
 * @param pairs the pairs, sorted by p_i
 * @param n the size of the domain, i.e., p_{k-1} plus the length of the last interval
 */
inline mds_pairs make_mds_pairs(const std::vector<std::pair<uint64_t, uint64_t>>& pairs, uint64_t n)
{
    mds_pairs I(pairs.size(), n);
    for (uint64_t i = 0; i < pairs.size(); i++) { I.set<0>(i, pairs[i].first); I.set<1>(i, pairs[i].second); }
    return I;
}

/**
 * @brief an extra row that the construction fills from one value per pair of the disjoint interval sequence
 *        (e.g., L' from the symbols of the BWT runs, or PLCP from one value per pair of Phi)
 */
struct mds_pair_row {
    // the values of the pairs: [0..k-1] field 0 of entry i is the value of the pair with the i-th smallest p (or q,
    // see in_q_order); nullptr <=> the caller fills the row
    const interleaved_bit_aligned_vectors* values = nullptr;
    // true <=> values[i] belongs to the pair with the i-th smallest q, false <=> to the pair with the i-th
    // smallest p (at most one row can be given by q)
    bool in_q_order = false;
    // true <=> an interval starting d positions after the start of its pair gets the value of the pair minus d,
    // false <=> each interval gets the value of its pair
    bool subtract_offset = false;
};

class mds_piece_marks;

// default of the balancing parameter a (see mds_params::a)
inline constexpr uint64_t default_balancing_parameter = 8;

// the smallest balancing parameter a: the balancing bounds k' by k * a/(a-1) (see mds_params::a)
inline constexpr uint64_t min_balancing_parameter = 2;

// the largest number of intervals k' (after the balancing) of a move data structure: its construction stores
// interval indices and pi(0..k'), with pi(k') = k', as uint40_t
inline constexpr uint64_t mds_max_num_intervals = uint40_t::max;

/**
 * @brief throws std::length_error if a move data structure would have more than mds_max_num_intervals intervals
 * @param num_intervals k', or a lower bound of it (k, or k plus the split offsets found so far)
 */
inline void mds_check_num_intervals(uint64_t num_intervals)
{
    if (num_intervals > mds_max_num_intervals) [[unlikely]] {
        throw std::length_error("move data structure construction: " + std::to_string(num_intervals)
            + " intervals, at most 2^" + std::to_string(8 * uint40_t::width) + " - 1 are supported (for an index: "
            "the BWT of the input has too many runs)");
    }
}

/**
 * @brief parameters of the construction of a move data structure
 */
struct mds_params {
    uint64_t num_threads = omp_get_max_threads(); // number of threads to use during the construction
    // balancing parameter a >= 2: after the balancing, every output interval contains fewer than 2a starting positions
    // of input intervals (with an inverse, also vice versa); restricts k' to at most k * a/(a-1)
    uint64_t a = default_balancing_parameter;
    // maximum interval length of I, if I is already balanced: the balancing is skipped and D_offs gets the bit
    // width of this value; 0 <=> I is balanced by the construction
    uint64_t max_interval_length = 0;
    // true <=> the construction frees I (set by the constructors of move_data_structure)
    bool delete_I = false;
    // [0..k-1] one byte per pair, s.t. the stable order of the pairs by these bytes is their order by q (e.g.,
    // the symbols of the BWT runs for LF), or nullptr; replaces the radix sort of the pairs by a counting sort
    const uint8_t* q_order_keys = nullptr;
    // [0..number of extra rows - 1] the extra rows filled by the construction (see mds_pair_row); nullptr <=> none
    const mds_pair_row* pair_rows = nullptr;
    // if not nullptr, receives a bit vector with select marking the first input interval of each pair, i.e.,
    // select(i) is the index of the first input interval of the pair with the i-th smallest p
    mds_piece_marks* first_pieces = nullptr;
    // width of D_offs, limits the interval length to l_max = 2^omega_offs - 1; 0 <=> chosen by the construction
    // (see construction::choose_omega_offs)
    uint8_t omega_offs = 0;
    // with omega_offs = 0: the width of D_offs is at least bit_width(max_length_factor * n / k), i.e., l_max is at least
    // about max_length_factor times the mean interval length of I
    double max_length_factor = 8.0;
    // with omega_offs = 0 and the encoding POS: the construction tries widths of D_offs up to this many bits larger
    // than the one given by max_length_factor ...
    uint8_t max_omega_offs_widening = 3;
    // ... and takes a larger width only if it reduces the number of blocks by more than this percentage
    uint64_t min_block_reduction_percent = 1;
    // number of chunks per thread of the parallel loops of the construction with a dynamic schedule (over pairs, pieces
    // or blocks, whose work differs); more chunks balance the work better, but each chunk has a constant cost
    uint64_t chunks_per_thread = 16;
    // bits of the header value of each block of the input intervals, a value the user of the move data structure
    // writes (see set_header_value; positional encoding without an inverse only)
    uint8_t header_value_width = 0;
    bool log = false; // controls whether to print log messages during the construction
    std::ostream* mf = nullptr; // measurement file to write runtime data to
};

/**
 * @brief move data structure: stores a disjoint interval sequence of [0, n) as k' input intervals
 *        [p(x), p(x+1)) and maps a position i in the x-th input interval to f_I(i) = q(x) + (i - p(x)). For each
 *        input interval x it stores p(x), idx(x), the index of the input interval containing q(x),
 *        offs(x) = q(x) - p(idx(x)), and up to max_extra_rows extra rows, i.e., one integer per interval and row
 *        (e.g., L' for M_LF). With an inverse, it stores the output intervals as well (see move_mode_t).
 *
 * Storage of the fields by encoding (see move_pos_encoding_t):
 * - POS: input_blocks (and output_blocks), blocks of one cache line (see move_input_interval_blocks)
 * - EF: D_rows (D_low[x] = p(x) mod 2^omega_offs, idx(x), offs(x), extra rows) and D_high (the increments of the
 *   high parts p(x) >> omega_offs); p(x) = (D_high.rank(x) << omega_offs) | D_low[x]
 * @tparam enc encoding of the interval starting positions
 * @tparam mode direction(s) of the move queries (UNI or BI)
 */
template <move_pos_encoding_t enc = POS, move_mode_t mode = UNI>
class move_data_structure {
public:
    // ############################# TYPES AND PARAMETERS #############################

    static_assert(mode == UNI || enc == POS, "the mode BI requires the encoding POS");
    static constexpr bool has_inverse = mode != UNI;

    static constexpr move_pos_encoding_t encoding = enc;
    static constexpr move_mode_t query_mode = mode;

    static constexpr uint8_t num_rows = 8; // maximum number of fields of an interval in D_rows (core fields and extra rows)
    static constexpr uint8_t num_core_rows = 3; // number of core fields (D_low[x], idx(x), offs(x))
    static constexpr uint8_t max_extra_rows = num_rows - num_core_rows; // maximum number of extra rows
    using row_widths_t = std::vector<uint8_t>; // widths (in bits) of the extra rows; its size is their number

    using pair_arr_t = mds_pairs; // the pairs of the disjoint interval sequence I

    // pair type of the queries, e.g., (position, interval index)
    using query_pair_t = std::pair<uint64_t, uint64_t>;

    // defined in construction/construction.hpp
    class construction;

protected:
    static_assert(max_extra_rows == move_interval_blocks::max_extra_rows);

    struct no_inverse { };

    // type of the blocks of the output intervals: move_input_interval_blocks (storing the inverse permutation)
    // for BI
    using output_blocks_t = std::conditional_t<mode == BI, move_input_interval_blocks, no_inverse>;

    // ############################# VARIABLES #############################

    uint64_t n = 0; // n = p(k') = p(k'-1) + len(k'-1), the size of the domain of the permutation
    uint64_t k = 0; // k, the number of intervals in the original disjoint interval sequence I
    uint64_t k_ = 0; // k', the number of intervals in the balanced disjoint interval sequence B_a(I), k <= k'
    uint64_t a = 0; // balancing parameter a (see mds_params::a)
    uint8_t omega_p = 0; // bits per p(x), bit_width(n)
    uint8_t omega_idx = 0; // bits per idx(x), bit_width(k')
    uint8_t omega_offs = 0; // bits per offs(x) (also per D_low[x] in EF)
    uint8_t num_extra_rows = 0; // number of extra rows
    uint8_t header_value_width = 0; // bits of the header value of each block (see mds_params::header_value_width)
    // true <=> the records of the input intervals store their lengths (see move_input_interval_blocks::stores_lengths)
    bool records_store_lengths = false;

    /* (EF) the fields of the intervals: [0..k'] entry x stores in field 0 D_low[x] = p(x) mod 2^omega_offs, in
       field 1 idx(x), in field 2 offs(x) and in field 3 + J the J-th extra row (widths: see interleaved_widths);
       entry k' is the sentinel interval with D_low n mod 2^omega_offs, idx k', offs 0 and the extra rows 0 */
    interleaved_bit_aligned_vectors D_rows;
    /* (EF) the increments of the high parts p(x) >> omega_offs: [0..k'] bit x is set <=> p(x+1) >> omega_offs =
       (p(x) >> omega_offs) + 1 (it grows by at most 1, since len(x) < 2^omega_offs); bit k' is 0; the rank of
       line l is p(448 l) >> omega_offs, so rank(x) = p(x) >> omega_offs */
    cache_line_rank_bit_vector D_high;

    // (POS) the blocks of the input intervals
    move_input_interval_blocks input_blocks;
    // (POS, with an inverse) the blocks of the output intervals (see output_blocks_t)
    output_blocks_t output_blocks;

    // ############################# INTERNAL METHODS #############################

    /**
     * @brief returns the widths of the fields of D_rows (EF): D_low[x], idx(x), offs(x), extra rows
     * @param row_widths widths (in bits) of the extra rows
     * @return the widths (in bits)
     */
    std::array<uint8_t, num_rows> interleaved_widths(const row_widths_t& row_widths) const;

    // returns D_low[x] = p(x) mod 2^omega_offs
    inline uint64_t low(uint64_t x) const requires(enc == EF) { return D_rows.template get<0, uint64_t>(x); }

    // returns the mask of the omega_offs lowest bits
    inline uint64_t low_mask() const requires(enc == EF) { return (uint64_t { 1 } << omega_offs) - 1; }

    /**
     * @brief advances the (interval, offset)-pair (x, o) over the input intervals until o < len(x), i.e., while
     *        o >= len(x), subtracts len(x) from o and increments x (EF); the lengths are the differences of the
     *        D_low of consecutive intervals modulo 2^omega_offs, since every input interval is shorter than
     *        2^omega_offs
     * @param o offset from p(x), with p(x) + o < n; is set to the offset within the resulting interval
     * @param x [0..k'-1] input interval index; is set to the index of the input interval containing p(x) + o
     * @return D_low[x] of the resulting interval x
     */
    inline uint64_t advance_to_interval(uint64_t& o, uint64_t& x) const requires(enc == EF);

    /**
     * @brief builds the move data structure from the disjoint interval sequence I
     * @param I a disjoint interval sequence
     * @param n the size of the domain, p_{k-1} plus the length of the last interval of I
     * @param params construction parameters
     * @param row_widths widths (in bits) of the extra rows
     * @param row_widths_inv widths (in bits) of the extra rows of the output intervals
     * @param pi_out vector to store pi in, if not nullptr
     */
    void build(mds_pairs& I, uint64_t n, mds_params params, row_widths_t row_widths, row_widths_t row_widths_inv,
        no_init_vector<uint40_t>* pi_out);

    /**
     * @brief computes the layout of the blocks of the input intervals and (with an inverse) of the output
     *        intervals. The blocks of the input intervals store slots of output intervals and vice versa (see
     *        move_interval_blocks); both slot widths start at std::bit_width(k') and are increased until the
     *        slots of each side fit into the slot field of the other side.
     * @param l_max maximum length of an input interval
     * @param extra_widths widths (in bits) of the extra rows
     * @param l_max_inv maximum length of an output interval (unused without an inverse)
     * @param omega_offs_inv width (in bits) of D_offs_inv
     * @param num_extra_rows_inv number of extra rows of the output intervals
     * @param extra_widths_inv widths (in bits) of the extra rows of the output intervals
     */
    void compute_layout(uint64_t l_max, const std::array<uint8_t, max_extra_rows>& extra_widths, uint64_t l_max_inv,
        uint8_t omega_offs_inv, uint8_t num_extra_rows_inv, const std::array<uint8_t, max_extra_rows>& extra_widths_inv);

public:
    // ############################# CONSTRUCTORS #############################

    move_data_structure() = default;

    /**
     * @brief constructs a new move data structure from a disjoint interval sequence I
     * @param I a disjoint interval sequence
     * @param n the size of the domain, p_{k-1} plus the length of the last interval of I
     * @param params construction parameters
     * @param row_widths widths (in bits) of the extra rows
     * @param row_widths_inv widths (in bits) of the extra rows of the output intervals (only with an inverse)
     * @param pi_out vector to store pi in, if not nullptr
     */
    move_data_structure(mds_pairs& I, uint64_t n, mds_params params = {}, row_widths_t row_widths = {},
        row_widths_t row_widths_inv = {}, no_init_vector<uint40_t>* pi_out = nullptr)
    {
        build(I, n, params, row_widths, row_widths_inv, pi_out);
    }

    /**
     * @brief constructs a new move data structure from a disjoint interval sequence I and frees I
     * @param I a disjoint interval sequence
     * @param n the size of the domain, p_{k-1} plus the length of the last interval of I
     * @param params construction parameters
     * @param row_widths widths (in bits) of the extra rows
     * @param row_widths_inv widths (in bits) of the extra rows of the output intervals (only with an inverse)
     * @param pi_out vector to store pi in, if not nullptr
     */
    move_data_structure(mds_pairs&& I, uint64_t n, mds_params params = {}, row_widths_t row_widths = {},
        row_widths_t row_widths_inv = {}, no_init_vector<uint40_t>* pi_out = nullptr)
    {
        params.delete_I = true;
        build(I, n, params, row_widths, row_widths_inv, pi_out);
    }

    // ############################# MISC PUBLIC METHODS #############################

    // returns the size of the data structure in bytes
    uint64_t size_in_bytes() const;

    // returns n = p(k'), the size of the domain
    inline uint64_t max_value() const { return n; }

    // returns k', the number of intervals
    inline uint64_t num_intervals() const { return k_; }

    // returns the balancing parameter a
    inline uint64_t balancing_parameter() const { return a; }

    // returns whether the move data structure is empty (i.e., has not been built)
    inline bool empty() const { return num_intervals() == 0; }

    // returns the number of bits per p(x)
    inline uint8_t width_p() const { return omega_p; }

    // returns the number of bits per idx(x)
    inline uint8_t width_idx() const { return omega_idx; }

    // returns the number of bits per offs(x) (and per D_low[x] in EF)
    inline uint8_t width_offs() const { return omega_offs; }

    // returns the number of extra rows of the output intervals
    inline uint8_t num_extra_rows_inv() const requires(has_inverse) { return output_blocks.num_extra_rows(); }

    /**
     * @brief returns the width (in bits) of the J-th extra row
     * @tparam J row index
     */
    template <uint8_t J>
    inline uint8_t row_width() const
    {
        if constexpr (enc == POS) return input_blocks.row_width(J);
        else return D_rows.template width<num_core_rows + J>();
    }

    /**
     * @brief returns the width (in bits) of the J-th extra row of the output intervals
     * @tparam J row index
     */
    template <uint8_t J>
    inline uint8_t row_width_inv() const requires(has_inverse) { return output_blocks.row_width(J); }

    // returns the number of input intervals per block (positional encoding only)
    inline uint64_t intervals_per_block() const requires(enc == POS) { return input_blocks.intervals_per_block(); }

    // returns the number of blocks of the input intervals (positional encoding only)
    inline uint64_t num_blocks() const requires(enc == POS) { return input_blocks.data_blocks(); }

    // true <=> the records of the input intervals can be read by a record_reader (see move_input_interval_blocks)
    inline bool records_fit_one_word() const requires(enc == POS) { return input_blocks.records_fit_one_word(); }

    // returns a record_reader of the input intervals (see move_input_interval_blocks::record_reader)
    inline move_input_interval_blocks::record_reader reader() const requires(enc == POS) { return input_blocks.reader(); }

    /**
     * @brief returns the size in bytes of the blocks of the input intervals after add_rows (positional encoding
     *        without an inverse only; the records store the lengths of the intervals)
     * @param bits the total width (in bits) of the added rows
     * @param header_value_width bits of the header value of each block
     * @param l_max the maximum length of an input interval
     */
    uint64_t blocks_size_with_rows(uint8_t bits, uint8_t header_value_width, uint64_t l_max) const
        requires(enc == POS && mode == UNI);

    // prints p, q and idx (and with an inverse p_inv, idx_inv and pi) of all intervals
    void log_data_structures() const;

    // ############################# ACCESS METHODS #############################

    /**
     * @brief returns the value of the J-th extra row of the x-th input interval
     * @tparam J row index
     * @param x [0..k'-1] input interval index
     */
    template <uint8_t J>
    inline uint64_t row(uint64_t x) const
    {
        if constexpr (enc == POS) return input_blocks.row(x, J);
        else return D_rows.template get<num_core_rows + J, uint64_t>(x);
    }

    /**
     * @brief returns the value of the J-th extra row of the y-th output interval
     * @tparam J row index
     * @param y [0..k'-1] output interval index
     */
    template <uint8_t J>
    inline uint64_t row_inv(uint64_t y) const requires(has_inverse) { return output_blocks.row(y, J); }

    /**
     * @brief returns p(x), the starting position of the x-th input interval; O(1) time
     * @param x [0..k']
     * @return p(x)
     */
    inline uint64_t p(uint64_t x) const;

    // returns len(x) = p(x+1) - p(x), the length of the x-th input interval
    inline uint64_t len(uint64_t x) const
    {
        if constexpr (enc == POS) return input_blocks.len(x);
        else return (low(x + 1) - low(x)) & low_mask();
    }

    // returns q(x), the starting position of the image of the x-th input interval
    inline uint64_t q(uint64_t x) const
    {
        if constexpr (enc == POS) return input_blocks.q(x);
        else return p(idx(x)) + offs(x);
    }

    // returns idx(x), the index of the input interval containing q(x)
    inline uint64_t idx(uint64_t x) const
    {
        if constexpr (enc == POS) return input_blocks.idx(x);
        else return D_rows.template get<1, uint64_t>(x);
    }

    // returns offs(x) = q(x) - p(idx(x))
    inline uint64_t offs(uint64_t x) const
    {
        if constexpr (enc == POS) return input_blocks.offs(x);
        else return D_rows.template get<2, uint64_t>(x);
    }

    // returns the position p(x) + o of the (interval, offset)-pair (x, o)
    inline uint64_t pos(uint64_t x, uint64_t o) const { return p(x) + o; }

    // returns p_inv(y), the starting position of the y-th output interval
    inline uint64_t p_inv(uint64_t y) const requires(has_inverse) { return output_blocks.p(y); }

    // returns p_inv(y+1) - p_inv(y), the length of the y-th output interval
    inline uint64_t len_inv(uint64_t y) const requires(has_inverse) { return output_blocks.len(y); }

    // returns offs_inv(y) = p(pi(y)) - p_inv(idx_inv(y))
    inline uint64_t offs_inv(uint64_t y) const requires(has_inverse) { return output_blocks.offs(y); }

    // returns idx_inv(y), the index of the output interval containing p(pi(y))
    inline uint64_t idx_inv(uint64_t y) const requires(has_inverse) { return output_blocks.idx(y); }

    // returns pi(y), the index of the input interval whose image is the y-th output interval
    inline uint64_t pi(uint64_t y) const requires(has_inverse)
    {
        // scan for the input interval starting at q_inv(y), from the header value of the block of idx_inv(y)
        return input_blocks.scan_from(output_blocks.header_value(output_blocks.idx(y)), output_blocks.q(y));
    }

    // returns q_inv(y) = p(pi(y)), the starting position of the preimage of the y-th output interval
    inline uint64_t q_inv(uint64_t y) const requires(has_inverse) { return output_blocks.q(y); }

    // returns idx(pi(y)), the index of the input interval containing p_inv(y)
    inline uint64_t input_interval_of_output(uint64_t y) const requires(has_inverse)
    {
        return input_blocks.scan_from(output_blocks.header_value(y), output_blocks.p(y));
    }

    // returns the index of the output interval containing p(x)
    inline uint64_t output_interval_of_input(uint64_t x) const requires(has_inverse) { return output_interval_of(p(x), x); }

    // ############################# QUERY METHODS #############################

    /**
     * @brief returns (p(x2) + o2) - (p(x1) + o1) for two (interval, offset)-pairs; in EF, the lengths of the
     *        intervals x1, ..., x2 - 1 are added instead of reading D_high, if the entries of D_rows
     *        of x1, ..., x2 take at most one cache line
     * @param x1 first input interval index
     * @param o1 offset within the x1-th input interval
     * @param x2 second input interval index (with p(x2) + o2 >= p(x1) + o1)
     * @param o2 offset within the x2-th input interval
     * @return (p(x2) + o2) - (p(x1) + o1)
     */
    inline uint64_t distance(uint64_t x1, uint64_t o1, uint64_t x2, uint64_t o2) const;

    /**
     * @brief returns the (interval, offset)-pair (x, i - p(x)) of a position i, where x is the input interval
     *        containing i; binary search over the p(x)
     * @param i [0..n-1] a position
     * @return the (interval, offset)-pair (x, i - p(x))
     */
    inline query_pair_t interval_and_offset(uint64_t i) const;

    // fetches the data that move(i, x) reads after the input interval x into the cache (positional encoding only)
    inline void prefetch_move(uint64_t x) const
    {
        if constexpr (enc == POS) input_blocks.prefetch_move(x);
    }

    /**
     * @brief move query on positions: changes (i, x), i in [p(x), p(x+1)), to (i', x') with i' = f_I(i) and
     *        i' in [p(x'), p(x'+1))
     * @param i [0..n-1] position in the x-th input interval
     * @param x [0..k'-1] index of the input interval containing i
     */
    inline void move(uint64_t& i, uint64_t& x) const;

    /**
     * @brief move query on a (position, interval)-pair (i, x) (see move(i, x))
     * @param pair the pair (i, x) to move
     * @return the moved pair
     */
    inline query_pair_t move(query_pair_t pair) const
    {
        move(pair.first, pair.second);
        return pair;
    }

    /**
     * @brief move query on (interval, offset)-pairs (EF only): changes (o, x), o < len(x), to (o', x') with
     *        p(x') + o' = f_I(p(x) + o) and o' < len(x'); reads only D_low, D_idx and D_offs, not D_high
     * @param o [0..len(x)-1] offset within the x-th input interval
     * @param x [0..k'-1] input interval index
     */
    inline void move_offset(uint64_t& o, uint64_t& x) const requires(enc == EF);

    /**
     * @brief returns the index of the output interval containing position i; scans the output intervals
     *        starting at the header value of the block of x (the output interval containing the block start)
     * @param i [0..n-1] a position in the x-th input interval
     * @param x [0..k'-1] input interval index
     */
    inline uint64_t output_interval_of(uint64_t i, uint64_t x) const requires(has_inverse) { return output_blocks.scan_from(input_blocks.header_value(x), i); }

    /**
     * @brief returns the index of the input interval containing position i; scans the input intervals
     *        starting at the header value of the block of y (the input interval containing the block start)
     * @param i [0..n-1] a position in the y-th output interval
     * @param y [0..k'-1] output interval index
     */
    inline uint64_t input_interval_of(uint64_t i, uint64_t y) const requires(has_inverse) { return input_blocks.scan_from(output_blocks.header_value(y), i); }

    /**
     * @brief returns the index of the input interval containing position i, if it lies in the block of the x-th
     *        input interval or in one of the move_input_interval_blocks::near_blocks blocks before or after it (see
     *        move_input_interval_blocks::interval_near)
     * @param i [0..n-1] a position
     * @param x [0..k'-1] index of an input interval near position i
     * @return the index of the input interval containing i, or num_intervals() if it was not found
     */
    inline uint64_t interval_near(uint64_t i, uint64_t x) const requires(has_inverse) { return input_blocks.interval_near(i, x); }

    /**
     * @brief move query on the inverse permutation: changes (i, y), i in [p_inv(y), p_inv(y+1)), to (i', y')
     *        with i' = f_I^{-1}(i) and i' in [p_inv(y'), p_inv(y'+1))
     * @param i [0..n-1] position
     * @param y [0..k'-1] index of the output interval containing i
     */
    inline void move_inv(uint64_t& i, uint64_t& y) const requires(has_inverse);

    /**
     * @brief move query (see move(i, x)) that also computes the index of the output interval containing
     *        f_I(i) (see output_interval_of)
     * @param i [0..n-1] position
     * @param x [0..k'-1] index of the input interval containing i
     * @param y is set to the index of the output interval containing f_I(i)
     */
    inline void move(uint64_t& i, uint64_t& x, uint64_t& y) const requires(has_inverse)
    {
        move(i, x);
        y = output_interval_of(i, x);
    }

    /**
     * @brief move query on the inverse permutation (see move_inv(i, y)) that also computes the index of the
     *        input interval containing f_I^{-1}(i)
     * @param i [0..n-1] position
     * @param x is set to the index of the input interval containing f_I^{-1}(i)
     * @param y [0..k'-1] index of the output interval containing i
     */
    inline void move_inv(uint64_t& i, uint64_t& x, uint64_t& y) const requires(has_inverse);

    // ############################# CONSTRUCTION METHODS #############################

    /**
     * @brief sets the J-th extra row of the x-th input interval to v; can be called by multiple threads for
     *        different x
     * @tparam J row index
     * @param x [0..k'-1] input interval index
     * @param v value; has to fit into the width of the row
     */
    template <uint8_t J>
    inline void set_row(uint64_t x, uint64_t v)
    {
        if constexpr (enc == POS) input_blocks.set_row(x, J, v);
        else D_rows.template set_parallel<num_core_rows + J, uint64_t>(x, v);
    }

    /**
     * @brief sets the J-th extra row of the y-th output interval to v; can be called by multiple threads for
     *        different y
     * @tparam J row index
     * @param y [0..k'-1] output interval index
     * @param v value; has to fit into the width of the row
     */
    template <uint8_t J>
    inline void set_row_inv(uint64_t y, uint64_t v) requires(has_inverse) { output_blocks.set_row(y, J, v); }

    /**
     * @brief changes the width of the J-th extra row, preserving the stored values (width 0 removes the row);
     *        in the positional encoding, the blocks are rebuilt with a new layout
     * @tparam J row index
     * @param width the new width (in bits); every value stored in the row must fit into it
     * @param num_threads number of threads to use (default: 1)
     */
    template <uint8_t J>
    void set_row_width(uint8_t width, uint64_t num_threads = 1) requires(mode == UNI);

    /**
     * @brief adds extra rows to the input intervals and the header value to each block, and lets the records store
     *        the lengths of the intervals (positional encoding without an inverse only; see
     *        move_input_interval_blocks::stores_lengths); the blocks are rebuilt with a new layout, and the header
     *        values are 0
     * @tparam row_fnc_t type of row_of
     * @param widths the widths (in bits) of the added rows
     * @param row_of row_of(x, J) returns the value of the J-th added row of the x-th input interval
     * @param header_value_width bits of the header value of each block
     * @param l_max the maximum length of an input interval
     * @param num_threads number of threads to use
     */
    template <typename row_fnc_t>
    void add_rows(const row_widths_t& widths, row_fnc_t row_of, uint8_t header_value_width, uint64_t l_max,
        uint64_t num_threads) requires(enc == POS && mode == UNI);

    /**
     * @brief sets the header value of the b-th block of the input intervals (see mds_params::header_value_width);
     *        can be called by several threads at the same time for different blocks
     * @param b block index
     * @param value the header value
     */
    inline void set_header_value(uint64_t b, uint64_t value) requires(enc == POS && !has_inverse) { input_blocks.set_header_value(b, value); }

    // ############################# SERIALIZATION METHODS #############################

    /**
     * @brief serializes the move data structure to an output stream
     * @param out output stream
     */
    void serialize(std::ostream& out) const;

    /**
     * @brief loads the move data structure from an input stream
     * @param in input stream
     * @param num_extra_rows the number of extra rows the data structure was built with; the positional encoding
     *        reads it from the stream and ignores the argument
     */
    void load(std::istream& in, uint8_t num_extra_rows);

    /**
     * @brief serializes the move data structure to an output stream
     * @param os output stream
     * @return the output stream
     */
    std::ostream& operator>>(std::ostream& os) const
    {
        serialize(os);
        return os;
    }

    /**
     * @brief loads the move data structure from an input stream, using its current number of extra rows
     * @param is input stream
     * @return the input stream
     */
    std::istream& operator<<(std::istream& is)
    {
        load(is, num_extra_rows);
        return is;
    }
};

#include "queries.tpp"
#include "construction/construction.hpp"
#include "move_data_structure.tpp"

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_HPP
