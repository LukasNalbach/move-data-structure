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

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <sstream>
#include "test-inputs.hpp"
#include "test-fuzz.hpp"
#include <move_data_structure/move_data_structure.hpp>

static std::mt19937 gen(test_seed());
static uint64_t max_num_threads = omp_get_max_threads();

// maximum input size, calibrated by run_fuzz to its time budget
uint64_t max_input_size = 1800000;
// smallest value run_fuzz reduces max_input_size to
constexpr uint64_t min_fuzz_input_size = 30000;
// number of random move data structures the fuzz test builds
constexpr uint64_t num_fuzz_iterations = 200;

// the input size of build_random_mds is drawn log-uniformly from [max_input_size / min_input_size_divisor,
// max_input_size]
constexpr uint64_t min_input_size_divisor = 3;
// the average input interval length of build_random_mds is log-normal with these parameters
constexpr double avg_interval_length_log_mean = 4.0;
constexpr double avg_interval_length_log_deviation = 2.0;
// the balancing parameter a of build_random_mds is min_balancing_parameter plus a log-normal value with these
// parameters, capped at max_balancing_parameter (the largest int16_t)
constexpr double balancing_parameter_log_mean = 2.0;
constexpr double balancing_parameter_log_deviation = 3.0;
constexpr uint64_t max_balancing_parameter = std::numeric_limits<int16_t>::max();
// with probability 1 / tail_inverse_probability, the last input interval of build_random_mds covers a tail of at
// least min_tail_share of the input plus fewer than input size / extra_tail_divisor positions
constexpr uint64_t tail_inverse_probability = 4;
constexpr double min_tail_share = 0.1;
constexpr uint64_t extra_tail_divisor = 2;
// verify_mds_move moves about this many positions per encoding, at distances in [avg / step_size_spread,
// avg * step_size_spread] for the average distance avg
constexpr uint64_t num_move_queries = 10000;
constexpr double step_size_spread = 1.5;
// extra row 1 of input interval x in verify_extra_rows stores x * extra_row_1_factor + 1
constexpr uint64_t extra_row_1_factor = 7;

// build_random_bidirectional_mds draws k from [bi_min_num_intervals, bi_max_num_intervals]; an interval is long with
// probability 1 / bi_long_interval_odds and then has a length in [bi_min_long_length, bi_max_long_length], else in
// [1, bi_max_short_length]; a is drawn from [min_balancing_parameter, bi_max_balancing_parameter]
constexpr uint64_t bi_min_num_intervals = 4500;
constexpr uint64_t bi_max_num_intervals = 9000;
constexpr uint64_t bi_long_interval_odds = 10;
constexpr uint64_t bi_max_short_length = 6;
constexpr uint64_t bi_min_long_length = 40;
constexpr uint64_t bi_max_long_length = 300;
constexpr uint64_t bi_max_balancing_parameter = 12;
// the number of threads of a bidirectional construction in odd iterations (at most max_num_threads)
constexpr uint64_t bi_max_num_threads = 4;
// width of the second extra row of the output intervals of random_bidirectional_mds::mds_with_rows
constexpr uint8_t bi_small_row_width = 7;
// distances of consecutive query positions of verify_bidirectional_move, verify_interval_near and
// verify_bidirectional_serialization (for mds and mds_with_rows)
constexpr uint64_t bi_move_query_stride = 17;
constexpr uint64_t bi_interval_near_query_stride = 23;
constexpr uint64_t bi_serialization_query_stride = 29;
constexpr uint64_t bi_serialization_query_stride_with_rows = 37;
// the hint of interval_near in verify_interval_near for a position i in the input interval x is
// (x + (i * hint_factor + hint_offset) mod n) mod k'
constexpr uint64_t hint_factor = 7919;
constexpr uint64_t hint_offset = 13;
// interval_near must find an input interval at most this many intervals away from the hint: it searches near_blocks
// blocks before and after the block of the hint, and every block holds at least one interval
constexpr uint64_t interval_near_guaranteed_distance = move_input_interval_blocks::near_blocks;

// a disjoint interval sequence: [0..k-1] the pairs (p_i, q_i), input interval i starts at p_i (increasing) and is
// mapped to the interval starting at q_i
using interval_sequence_t = std::vector<std::pair<uint64_t, uint64_t>>;

// returns the pairs of seq (values smaller than n), stored with 4, 5 or 6 bytes per value (drawn at random)
static mds_pairs make_mds_pairs_random_width(const interval_sequence_t& seq, uint64_t n)
{
    const std::array<uint64_t, 3> max_values { n, max_short_value + 1, max_narrow_value + 1 };
    return make_mds_pairs(seq, std::max(n, max_values[gen() % max_values.size()]));
}

/**
 * @brief maps the input intervals to output intervals in a random order: for a random permutation pi of the k input
 *        intervals, the image of input interval pi[j] is the j-th output interval
 * @param intervals [0..k] the pairs (p_i, 0) of the input intervals, followed by (n, 0); afterwards [0..k-1] the
 *        pairs (p_i, q_i)
 */
static void assign_random_output_intervals(interval_sequence_t& intervals)
{
    // [0..k-1] order[j] = input interval whose image is the j-th output interval
    std::vector<uint64_t> order(intervals.size() - 1);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), gen);
    uint64_t q = 0;

    for (uint64_t x : order) {
        intervals[x].second = q;
        q += intervals[x + 1].first - intervals[x].first;
    }

    intervals.pop_back();
}

// ############################# UNIDIRECTIONAL #############################

// a random disjoint interval sequence and the move data structures built from it in the encodings POS and EF
struct random_mds_instance {
    move_data_structure<POS, UNI> mds_pos;
    move_data_structure<EF, UNI> mds_ef;
    interval_sequence_t interval_sequence; // [0..k-1] the pairs (p_i, q_i) the data structures are built from
    uint64_t input_size; // n
    uint64_t num_intervals; // k, the number of input intervals before the balancing
    uint64_t a; // the balancing parameter
};

// returns a random disjoint interval sequence (input intervals of random lengths, mapped in a random order) and the
// move data structures built from it
static random_mds_instance build_random_mds()
{
    std::lognormal_distribution<double> avg_interval_length_distrib(avg_interval_length_log_mean,
        avg_interval_length_log_deviation);
    std::lognormal_distribution<double> a_distrib(balancing_parameter_log_mean, balancing_parameter_log_deviation);

    uint64_t input_size = random_log_uniform_size(
        std::max<uint64_t>(1, max_input_size / min_input_size_divisor), max_input_size, gen);
    uint64_t num_intervals = 0;
    interval_sequence_t interval_sequence;

    // choose input intervals of random lengths (uniform in [1, 2 * average])
    const uint64_t max_interval_length = std::max<uint64_t>(
        1, uint64_t(2 * avg_interval_length_distrib(gen)));
    std::uniform_int_distribution<uint64_t> interval_length_distrib(
        1, max_interval_length);
    uint64_t interval_length_prefix_sum = 0;
    interval_sequence.emplace_back(std::make_pair(0, 0));

    // the tail covered by the last interval (see tail_inverse_probability)
    const uint64_t tail = (gen() % tail_inverse_probability == 0)
        ? uint64_t(min_tail_share * double(input_size)) + gen() % std::max<uint64_t>(1, input_size / extra_tail_divisor)
        : 0;
    const uint64_t before_tail = input_size - std::min(tail, input_size);

    while (interval_length_prefix_sum < before_tail) {
        interval_length_prefix_sum = std::min<uint64_t>(
            interval_length_prefix_sum + interval_length_distrib(gen), before_tail);
        interval_sequence.emplace_back(std::make_pair(interval_length_prefix_sum, 0));
        num_intervals++;
    }

    if (tail != 0 && interval_length_prefix_sum < input_size) {
        interval_length_prefix_sum = input_size;
        interval_sequence.emplace_back(std::make_pair(interval_length_prefix_sum, 0));
        num_intervals++;
    }

    assign_random_output_intervals(interval_sequence);

    const uint64_t a = min_balancing_parameter + uint64_t(std::min(a_distrib(gen),
        double(max_balancing_parameter - min_balancing_parameter)));
    // build with a random number of threads
    uint64_t threads = std::uniform_int_distribution<uint64_t>(1, max_num_threads)(gen);
    move_data_structure<POS, UNI> mds_pos(make_mds_pairs_random_width(interval_sequence, input_size), input_size,
        { .num_threads = threads, .a = a });

    // EF with the omega_offs of POS (only then the intervals are split equally); the pairs of each encoding are
    // stored with a random width of their own
    move_data_structure<EF, UNI> mds_ef(make_mds_pairs_random_width(interval_sequence, input_size), input_size,
        { .num_threads = threads, .a = a, .omega_offs = mds_pos.width_offs() });

    return random_mds_instance { std::move(mds_pos), std::move(mds_ef), std::move(interval_sequence), input_size,
        num_intervals, a };
}

// checks that EF stores the same intervals as POS, and that the balanced sequence has increasing p, correct D_idx and
// no a-heavy output interval
static void verify_mds_structure(random_mds_instance& r)
{
    move_data_structure<POS, UNI>& mds = r.mds_pos;
    move_data_structure<EF, UNI>& mds_ef = r.mds_ef;

    #pragma omp parallel for num_threads(max_num_threads)
    for (uint64_t i = 0; i < mds.num_intervals(); i++) {
        EXPECT_EQ(mds_ef.p(i), mds.p(i)) << "EF p mismatch (i = " << i << ")";
        EXPECT_EQ(mds_ef.q(i), mds.q(i)) << "EF q mismatch (i = " << i << ")";
        EXPECT_EQ(mds_ef.idx(i), mds.idx(i)) << "EF idx mismatch (i = " << i << ")";
        EXPECT_EQ(mds_ef.offs(i), mds.offs(i)) << "EF offs mismatch (i = " << i << ")";
        EXPECT_EQ(mds_ef.len(i), mds.len(i)) << "EF len mismatch (i = " << i << ")";
    }
    EXPECT_EQ(mds_ef.p(mds.num_intervals()), mds.p(mds.num_intervals())) << "EF p(k') mismatch";

    #pragma omp parallel for num_threads(max_num_threads)
    for (uint64_t i = 0; i < mds.num_intervals(); i++) {
        EXPECT_LT(mds.p(i), mds.p(i + 1)) << "input interval starting positions must ascend (i = " << i << ")";

        // D_idx maps each output-interval start q(i) into the input interval that contains it
        EXPECT_TRUE(mds.p(mds.idx(i)) <= mds.q(i) && mds.q(i) < mds.p(mds.idx(i) + 1))
            << "D_idx does not map q(i) into its input interval (i = " << i << ")";

        // the output interval [q(i), q(i) + len(i)) contains fewer than 2a starting positions of input intervals
        uint64_t j = bin_search_min_geq(mds.q(i), 0, mds.num_intervals() - 1, [&mds](auto x) { return mds.p(x); });
        uint64_t num_input_starts = 0;

        while (mds.p(j) < mds.q(i) + (mds.p(i + 1) - mds.p(i))) {
            num_input_starts++;
            j++;
        }

        EXPECT_LT(num_input_starts, 2 * r.a) << "a-heavy output interval (i = " << i << ")";
    }
}

// checks move() of both encodings and move_offset() and distance() of EF against the interval sequence at about
// num_move_queries positions
static void verify_mds_move(random_mds_instance& r)
{
    move_data_structure<POS, UNI>& mds = r.mds_pos;
    move_data_structure<EF, UNI>& mds_ef = r.mds_ef;
    uint64_t input_size = r.input_size;
    interval_sequence_t& interval_sequence = r.interval_sequence;

    // [0..number of queries - 1] the query positions, increasing, at random distances in [avg / step_size_spread,
    // avg * step_size_spread]; avg is large enough that the smallest distance is at least 1
    uint64_t avg_step_size = std::max(uint64_t(std::ceil(step_size_spread)), input_size / num_move_queries);
    std::uniform_int_distribution<uint64_t> step_size_distrib(avg_step_size / step_size_spread,
        step_size_spread * avg_step_size);
    std::vector<uint64_t> query_positions;
    for (uint64_t i = 0; i < input_size; i += step_size_distrib(gen)) query_positions.emplace_back(i);

    #pragma omp parallel for num_threads(max_num_threads)
    for (uint64_t q = 0; q < query_positions.size(); q++) {
        const uint64_t i = query_positions[q];

        // the (position, input interval) pairs of i in POS and in the interval sequence
        std::pair<uint64_t, uint64_t> pair_pos { i,
            bin_search_max_leq(i, 0, mds.num_intervals() - 1, [&mds](uint64_t x) { return mds.p(x); }) };
        std::pair<uint64_t, uint64_t> pair_ref { i,
            bin_search_max_leq(i, 0, r.num_intervals - 1, [&interval_sequence](uint64_t x) { return interval_sequence[x].first; }) };

        std::pair<uint64_t, uint64_t> pair_ef = pair_pos;
        pair_pos = mds.move(pair_pos);
        pair_ref.first = interval_sequence[pair_ref.second].second + (pair_ref.first - interval_sequence[pair_ref.second].first);
        EXPECT_EQ(pair_pos.first, pair_ref.first);

        // the EF move takes and returns positions like POS
        EXPECT_EQ(mds_ef.interval_and_offset(i), std::make_pair(pair_ef.second, i - mds.p(pair_ef.second)));
        pair_ef = mds_ef.move(pair_ef);
        EXPECT_EQ(pair_ef, pair_pos);

        // the input interval returned by move contains the moved position
        EXPECT_TRUE(mds.p(pair_pos.second) <= pair_pos.first && pair_pos.first < mds.p(pair_pos.second + 1));

        // the EF move_offset takes and returns (offset, input interval) pairs
        auto [x, offset] = mds_ef.interval_and_offset(i);
        mds_ef.move_offset(offset, x);
        uint64_t out_pos = mds_ef.pos(x, offset);
        EXPECT_EQ(out_pos, pair_ref.first);
        EXPECT_TRUE(offset < mds_ef.len(x));
        EXPECT_EQ(x, pair_pos.second);

        // the distance between the pairs of i and of the next query position (or of n - 1)
        const uint64_t j = q + 1 < query_positions.size() ? query_positions[q + 1] : input_size - 1;
        auto [x_i, o_i] = mds_ef.interval_and_offset(i);
        auto [x_j, o_j] = mds_ef.interval_and_offset(j);
        EXPECT_EQ(mds_ef.distance(x_i, o_i, x_j, o_j), j - i);
    }
}

// value of extra row 0 (8 bits) of input interval x in verify_extra_rows: the low bits of x that fit in a signed char
static char extra_row_0_value(uint64_t x) { return char(x & uint64_t(std::numeric_limits<int8_t>::max())); }

// value of extra row 1 (16 bits) of input interval x in verify_extra_rows
static uint16_t extra_row_1_value(uint64_t x) { return uint16_t(x * extra_row_1_factor + 1); }

// fills two extra rows (char, uint16_t) and reads them back, before and after serialize/load, in encoding enc
template <move_pos_encoding_t enc>
static void verify_extra_rows(const interval_sequence_t& seq, uint64_t n, uint64_t a)
{
    using mds_t = move_data_structure<enc, UNI>;
    const typename mds_t::row_widths_t row_widths { 8 * sizeof(char), 8 * sizeof(uint16_t) };

    mds_params params { .num_threads = max_num_threads, .a = a };
    mds_t mds(make_mds_pairs_random_width(seq, n), n, params, row_widths);

    for (uint64_t x = 0; x < mds.num_intervals(); x++) {
        mds.template set_row<0>(x, extra_row_0_value(x));
        mds.template set_row<1>(x, extra_row_1_value(x));
    }

    for (uint64_t x = 0; x < mds.num_intervals(); x++) {
        EXPECT_EQ(mds.template row<0>(x), extra_row_0_value(x)) << "extra row 0 (x = " << x << ")";
        EXPECT_EQ(mds.template row<1>(x), extra_row_1_value(x)) << "extra row 1 (x = " << x << ")";
    }

    std::stringstream stream;
    mds.serialize(stream);
    mds_t loaded;
    loaded.load(stream, row_widths.size());
    ASSERT_EQ(loaded.num_intervals(), mds.num_intervals());

    for (uint64_t x = 0; x < mds.num_intervals(); x++) {
        EXPECT_EQ(loaded.template row<0>(x), extra_row_0_value(x));
        EXPECT_EQ(loaded.template row<1>(x), extra_row_1_value(x));
        EXPECT_EQ(loaded.p(x), mds.p(x));
    }
}

static void verify_extra_rows_all_encodings(random_mds_instance& r)
{
    verify_extra_rows<POS>(r.interval_sequence, r.input_size, r.a);
    verify_extra_rows<EF>(r.interval_sequence, r.input_size, r.a);
}

// ############################# BIDIRECTIONAL #############################

/**
 * @brief a random disjoint interval sequence with many short and few long intervals, the bidirectional move data
 *        structures built from it and the permutation it describes
 * @tparam mode BI
 */
template <move_mode_t mode>
struct random_bidirectional_mds {
    using mds_t = move_data_structure<POS, mode>;

    interval_sequence_t intervals; // [0..k-1] the pairs (p_i, q_i)
    uint64_t num_input_intervals; // k, the number of input intervals before the balancing
    uint64_t input_size; // n
    uint64_t a; // the balancing parameter
    mds_t mds;
    // the same move data structure with one extra row of the input intervals (row 0 of x is x) and two of the
    // output intervals (row 0 of y is y, row 1 is the low bi_small_row_width bits of y)
    mds_t mds_with_rows;
    // the permutation: [0..n-1] mapping[i] = image of position i
    std::vector<uint64_t> mapping;
    // the inverse permutation: [0..n-1] inverse[mapping[i]] = i
    std::vector<uint64_t> inverse;
};

// returns a random_bidirectional_mds built with 1 thread in even iterations, else with up to max_bi_num_threads threads
template <move_mode_t mode>
static random_bidirectional_mds<mode> build_random_bidirectional_mds(uint64_t iteration)
{
    using mds_t = move_data_structure<POS, mode>;
    random_bidirectional_mds<mode> r;
    const uint64_t k = std::uniform_int_distribution<uint64_t>(bi_min_num_intervals, bi_max_num_intervals)(gen);
    r.num_input_intervals = k;
    r.intervals.reserve(k + 1);
    uint64_t prefix = 0;

    for (uint64_t x = 0; x < k; x++) {
        r.intervals.emplace_back(prefix, 0);
        const bool long_interval = std::uniform_int_distribution<uint64_t>(0, bi_long_interval_odds - 1)(gen) == 0;
        prefix += long_interval
            ? std::uniform_int_distribution<uint64_t>(bi_min_long_length, bi_max_long_length)(gen)
            : std::uniform_int_distribution<uint64_t>(1, bi_max_short_length)(gen);
    }

    const uint64_t n = prefix;
    r.input_size = n;
    r.intervals.emplace_back(n, 0);
    assign_random_output_intervals(r.intervals);

    r.a = std::uniform_int_distribution<uint16_t>(min_balancing_parameter, bi_max_balancing_parameter)(gen);
    const uint64_t threads = iteration % 2 == 0 ? 1 : std::min<uint64_t>(bi_max_num_threads, max_num_threads);
    r.mds = mds_t(make_mds_pairs_random_width(r.intervals, n), n, { .num_threads = threads, .a = r.a });
    r.mds_with_rows = mds_t(make_mds_pairs_random_width(r.intervals, n), n, { .num_threads = threads, .a = r.a },
        { uint8_t(std::bit_width(n)) }, { uint8_t(std::bit_width(n)), bi_small_row_width });

    for (uint64_t x = 0; x < r.mds_with_rows.num_intervals(); x++) r.mds_with_rows.template set_row<0>(x, x);

    for (uint64_t y = 0; y < r.mds_with_rows.num_intervals(); y++) {
        r.mds_with_rows.template set_row_inv<0>(y, y);
        r.mds_with_rows.template set_row_inv<1>(y, y & low_bits_mask(bi_small_row_width));
    }

    r.mapping.resize(n);
    r.inverse.resize(n);

    for (uint64_t x = 0; x < k; x++) {
        const uint64_t end = x + 1 < k ? r.intervals[x + 1].first : n;

        for (uint64_t i = r.intervals[x].first; i < end; i++) {
            const uint64_t image = r.intervals[x].second + (i - r.intervals[x].first);
            r.mapping[i] = image;
            r.inverse[image] = i;
        }
    }

    return r;
}

// checks p_inv, q_inv, idx_inv, offs_inv, len_inv, the extra rows of the output intervals and the a-balancing in
// both directions
template <move_mode_t mode>
static void verify_bidirectional_structure(const random_bidirectional_mds<mode>& r)
{
    const auto& mds = r.mds;
    static_assert(std::remove_cvref_t<decltype(mds)>::query_mode == mode);
    EXPECT_EQ(r.mds_with_rows.num_extra_rows_inv(), 2);
    EXPECT_EQ(r.mds_with_rows.template row_width_inv<1>(), bi_small_row_width);

    // [0..k'-1] the starting positions p(x) of the input intervals, increasing
    std::vector<uint64_t> input_starts(mds.num_intervals());
    // [0..k'-1] the starting positions p_inv(y) of the output intervals, increasing
    std::vector<uint64_t> output_starts(mds.num_intervals());

    for (uint64_t j = 0; j < mds.num_intervals(); j++) {
        input_starts[j] = mds.p(j);
        output_starts[j] = mds.p_inv(j);
        EXPECT_LT(mds.p_inv(j), mds.p_inv(j + 1));

        // output_interval_of_input(x) is the output interval containing p(x), input_interval_of_output(y) is the input
        // interval containing p_inv(y)
        EXPECT_TRUE(mds.p_inv(mds.output_interval_of_input(j)) <= mds.p(j) && mds.p(j) < mds.p_inv(mds.output_interval_of_input(j) + 1))
            << "output_interval_of_input does not place p(x) in its output interval (x = " << j << ")";
        EXPECT_TRUE(mds.p(mds.input_interval_of_output(j)) <= mds.p_inv(j) && mds.p_inv(j) < mds.p(mds.input_interval_of_output(j) + 1))
            << "input_interval_of_output does not place p_inv(y) in its input interval (y = " << j << ")";

        // move_inv maps p_inv(y) to q_inv(y) = p(x_pre), where x_pre = pi(y) is the input interval whose image is the
        // y-th output interval
        uint64_t pos = mds.p_inv(j), y = j, x_pre;
        mds.move_inv(pos, x_pre, y);
        EXPECT_EQ(pos, mds.q_inv(j)) << "the inverse of p_inv(y) is not q_inv(y) (y = " << j << ")";
        EXPECT_EQ(mds.p(x_pre), mds.q_inv(j)) << "wrong preimage interval (y = " << j << ")";
        EXPECT_EQ(mds.q(x_pre), mds.p_inv(j)) << "the preimage's image does not start at p_inv(y)";
        EXPECT_EQ(mds.len_inv(j), mds.len(x_pre)) << "an output interval and its preimage differ in length";
        EXPECT_EQ(mds.idx_inv(j), mds.output_interval_of_input(x_pre));
        EXPECT_TRUE(mds.p_inv(mds.idx_inv(j)) <= mds.q_inv(j) && mds.q_inv(j) < mds.p_inv(mds.idx_inv(j) + 1));
        // offs_inv(y) is the offset of q_inv(y) in the output interval idx_inv(y)
        EXPECT_EQ(mds.p_inv(mds.idx_inv(j)) + mds.offs_inv(j), mds.q_inv(j));
    }

    EXPECT_EQ(mds.p_inv(mds.num_intervals()), r.input_size);

    // each input interval contains < 2a output interval starts, each output interval < 2a input interval starts
    for (uint64_t x = 0; x < mds.num_intervals(); x++) {
        const uint64_t q_end = mds.q(x) + mds.len(x);
        auto p_first = std::lower_bound(output_starts.begin(), output_starts.end(), mds.p(x));
        auto p_last = std::lower_bound(output_starts.begin(), output_starts.end(), mds.p(x + 1));
        EXPECT_LT(std::distance(p_first, p_last), 2 * r.a) << "input interval " << x << " is heavy";
        auto q_first = std::lower_bound(input_starts.begin(), input_starts.end(), mds.q(x));
        auto q_last = std::lower_bound(input_starts.begin(), input_starts.end(), q_end);
        EXPECT_LT(std::distance(q_first, q_last), 2 * r.a) << "output interval " << x << " is heavy";
    }
}

// checks the variants of move and move_inv, output_interval_of and input_interval_of against the permutation
template <move_mode_t mode>
static void verify_bidirectional_move(const random_bidirectional_mds<mode>& r)
{
    const auto& mds = r.mds;
    const auto& mds_with_rows = r.mds_with_rows;

    for (uint64_t i = 0; i < r.input_size; i += bi_move_query_stride) {
        uint64_t value = i;
        uint64_t x = mds.interval_and_offset(value).first;
        uint64_t y = mds.output_interval_of(value, x);
        EXPECT_EQ(mds.input_interval_of(value, y), x);

        mds.move(value, x, y);
        EXPECT_EQ(value, r.mapping[i]);
        EXPECT_TRUE(mds.p(x) <= value && value < mds.p(x + 1));
        EXPECT_TRUE(mds.p_inv(y) <= value && value < mds.p_inv(y + 1));

        // move(value, x) and move_inv(value, y) return the same results as the variants that compute both x and y
        uint64_t value_2 = i;
        uint64_t x_2 = mds.interval_and_offset(value_2).first;
        mds.move(value_2, x_2);
        EXPECT_EQ(value_2, value);
        EXPECT_EQ(x_2, x);

        value = i;
        x = mds.interval_and_offset(value).first;
        y = mds.output_interval_of(value, x);
        uint64_t y_2 = y;
        mds.move_inv(value, x, y);
        EXPECT_EQ(value, r.inverse[i]);
        EXPECT_TRUE(mds.p(x) <= value && value < mds.p(x + 1));
        EXPECT_TRUE(mds.p_inv(y) <= value && value < mds.p_inv(y + 1));

        value_2 = i;
        mds.move_inv(value_2, y_2);
        EXPECT_EQ(value_2, value);
        EXPECT_EQ(y_2, y);

        value = i;
        x = mds_with_rows.interval_and_offset(value).first;
        y = mds_with_rows.output_interval_of(value, x);
        mds_with_rows.move_inv(value, x, y);
        EXPECT_EQ(value, r.inverse[i]);
        EXPECT_TRUE(mds_with_rows.p(x) <= value && value < mds_with_rows.p(x + 1));
        EXPECT_TRUE(mds_with_rows.p_inv(y) <= value && value < mds_with_rows.p_inv(y + 1));
        EXPECT_EQ(mds_with_rows.template row<0>(x), x);
        EXPECT_EQ(mds_with_rows.template row_inv<0>(y), y);
        EXPECT_EQ(mds_with_rows.template row_inv<1>(y), y & low_bits_mask(bi_small_row_width));
    }
}

// checks interval_near
template <move_mode_t mode>
static void verify_interval_near(const random_bidirectional_mds<mode>& r)
{
    const auto& mds = r.mds;

    for (uint64_t i = 0; i < r.input_size; i += bi_interval_near_query_stride) {
        // interval_near(i, hint) returns the input interval containing i, or num_intervals() if it is too far from
        // the input interval hint
        uint64_t x = mds.interval_and_offset(i).first;
        uint64_t hint = (x + (i * hint_factor + hint_offset) % r.input_size) % mds.num_intervals();
        uint64_t x_near = mds.interval_near(i, hint);
        EXPECT_TRUE(x_near == x || x_near == mds.num_intervals()) << "interval_near returned a wrong interval (i = " << i << ")";
        uint64_t distance = x > hint ? x - hint : hint - x;
        if (distance <= interval_near_guaranteed_distance)
            EXPECT_EQ(x_near, x) << "interval_near did not find a near interval (i = " << i << ")";
        EXPECT_EQ(mds.interval_near(i, x), x);
    }
}

// checks that serialize/load preserves move_inv, pi, p and the extra rows
template <move_mode_t mode>
static void verify_bidirectional_serialization(const random_bidirectional_mds<mode>& r)
{
    using mds_t = typename random_bidirectional_mds<mode>::mds_t;
    const auto& mds = r.mds;

    std::stringstream file;
    mds.serialize(file);
    mds_t loaded;
    loaded.load(file, 0);
    ASSERT_EQ(loaded.num_intervals(), mds.num_intervals());

    for (uint64_t i = 0; i < r.input_size; i += bi_serialization_query_stride) {
        uint64_t pos_1 = i, x_1 = mds.interval_and_offset(pos_1).first, y_1 = mds.output_interval_of(pos_1, x_1);
        uint64_t pos_2 = pos_1, x_2 = x_1, y_2 = y_1;
        mds.move_inv(pos_1, x_1, y_1);
        loaded.move_inv(pos_2, x_2, y_2);
        EXPECT_EQ(pos_2, pos_1);
        EXPECT_EQ(x_2, x_1);
        EXPECT_EQ(y_2, y_1);
        EXPECT_EQ(loaded.pi(y_1), mds.pi(y_1));
        EXPECT_EQ(loaded.p(x_1), mds.p(x_1));
    }

    std::stringstream stream;
    r.mds_with_rows.serialize(stream);
    mds_t loaded_with_rows;
    loaded_with_rows.load(stream, 1);

    for (uint64_t i = 0; i < r.input_size; i += bi_serialization_query_stride_with_rows) {
        uint64_t value = i;
        uint64_t x = loaded_with_rows.interval_and_offset(value).first;
        uint64_t y = loaded_with_rows.output_interval_of(value, x);
        loaded_with_rows.move_inv(value, x, y);
        EXPECT_EQ(value, r.inverse[i]);
        EXPECT_EQ(loaded_with_rows.template row<0>(x), x);
        EXPECT_EQ(loaded_with_rows.template row_inv<0>(y), y);
        EXPECT_EQ(loaded_with_rows.template row_inv<1>(y), y & low_bits_mask(bi_small_row_width));
    }
}

// builds a random bidirectional move data structure and verifies its queries in both directions
template <move_mode_t mode>
static void verify_bidirectional_mds(uint64_t iteration)
{
    random_bidirectional_mds<mode> r = build_random_bidirectional_mds<mode>(iteration);
    ASSERT_GE(r.mds.num_intervals(), r.num_input_intervals);
    verify_bidirectional_structure(r);
    verify_bidirectional_move(r);
    verify_interval_near(r);
    verify_bidirectional_serialization(r);
}

// builds one random move data structure in each encoding and runs every check on it; the mode BI is verified on
// interval sequences of its own
static void verify_move_data_structure(uint64_t iteration)
{
    random_mds_instance r = build_random_mds();
    verify_mds_structure(r);
    verify_mds_move(r);
    verify_extra_rows_all_encodings(r);
    verify_bidirectional_mds<BI>(iteration);
}

TEST(test_move_data_structure, move_data_structure)
{
    run_fuzz("move-data-structure", { { "move-data-structure", verify_move_data_structure } },
        fuzz_iterations(num_fuzz_iterations), &max_input_size, min_fuzz_input_size);
}

// max_extra_rows extra rows of 64 bits leave room for only one input interval per block
TEST(test_move_data_structure, add_rows_too_wide_throws_logic_error)
{
    using mds_t = move_data_structure<POS, UNI>;
    const uint64_t n = 2;
    mds_t mds(make_mds_pairs({ { 0, 1 }, { 1, 0 } }, n), n, { .num_threads = 1 });
    const mds_t::row_widths_t widths(mds_t::max_extra_rows, 8 * sizeof(uint64_t));
    EXPECT_THROW(mds.add_rows(widths, [](uint64_t, uint8_t) { return uint64_t { 0 }; }, 0, n, 1), std::logic_error);
}

// the number of intervals is checked against the 40-bit interval indices of the construction
TEST(test_move_data_structure, too_many_intervals_throws_length_error)
{
    EXPECT_EQ(uint64_t(uint40_t(mds_max_num_intervals)), mds_max_num_intervals);
    EXPECT_NO_THROW(mds_check_num_intervals(mds_max_num_intervals));
    EXPECT_THROW(mds_check_num_intervals(mds_max_num_intervals + 1), std::length_error);
}
