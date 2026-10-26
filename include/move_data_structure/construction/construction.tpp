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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_TPP

#include <move_data_structure/construction/construction.hpp>

// ############################# CONSTRUCTOR #############################

template <move_pos_encoding_t enc, move_mode_t mode>
move_data_structure<enc, mode>::construction::construction(
    move_data_structure& mds,
    pair_arr_t& I,
    uint64_t n,
    mds_params params,
    row_widths_t row_widths,
    row_widths_t row_widths_inv,
    no_init_vector<uint40_t>* pi_out)
    : mds(mds)
    , row_widths(row_widths)
    , row_widths_inv(row_widths_inv)
    , pi_out(pi_out)
    , I(I)
{
    this->n = n;
    this->k = I.size();
    mds_check_num_intervals(k);
    this->a = params.a;
    this->num_threads = params.num_threads;
    this->chunks_per_thread = params.chunks_per_thread;
    this->delete_I = params.delete_I;
    this->q_order_keys = params.q_order_keys;

    // sort the extra rows given by the construction into the ones in the order of p and the one in the order of q
    if (params.pair_rows != nullptr) {
        for (uint8_t J = 0; J < row_widths.size(); J++) {
            const mds_pair_row& row = params.pair_rows[J];
            if (row.values == nullptr || row_widths[J] == 0) continue;
            if (row.values->size() < k) throw std::invalid_argument("move data structure construction: fewer row values than pairs");
            pair_rows[J] = row;

            if (!row.in_q_order) {
                rows_in_p_order[num_rows_in_p_order++] = J;
            } else {
                if (row_in_q_order != max_extra_rows) throw std::invalid_argument("move data structure construction: more than one row by q");
                row_in_q_order = J;
            }
        }
    }

    first_pieces = params.first_pieces;

    this->log = params.log;
    this->max_interval_length = params.max_interval_length;
    this->mf = params.mf;

    if (log) {
        time = now();
        time_start = time;
        log_out() << std::endl;
    }

    // every thread gets at least one cache line of the pairs
    const uint64_t max_threads = std::max<uint64_t>(1, I.size_in_bytes() / cache_line_bytes);

    if (num_threads > max_threads) {
        num_threads = max_threads;
        if (log) log_out() << "warning: fewer than one cache line of pairs per thread, setting p to " << num_threads << std::endl;
    }

    omp_set_num_threads(num_threads);
    uint8_t omega_offs;

    if (max_interval_length > 0) {
        // I is already balanced and its maximum interval length is given (see mds_params)
        l_max = max_interval_length;
        omega_offs = std::bit_width(l_max);
    } else {
        // width of D_offs: params.omega_offs, or chosen by choose_omega_offs starting at
        // bit_width(params.max_length_factor * n / k)
        omega_offs = params.omega_offs != 0 ? params.omega_offs : std::bit_width((uint64_t) ((params.max_length_factor * n) / k));
        if constexpr (enc == POS) if (params.omega_offs == 0) omega_offs = choose_omega_offs(omega_offs, params);

        // limit the interval length to l_max = 2^omega_offs - 1, so that D_offs fits into omega_offs bits
        l_max = (uint64_t { 1 } << omega_offs) - 1;
    }

    mds.k = k;
    mds.a = a;
    mds.omega_offs = omega_offs;

    order_pairs();
    splits.init(k, n);
    if (max_interval_length == 0) balance();
    mark_pieces();

    if constexpr (enc == POS) fill_blocks();
    else fill_ef();

    // free the data of the construction
    splits.clear();
    if (first_pieces != nullptr) *first_pieces = std::move(first_p);
    first_p.clear();
    first_q.clear();
    P = mds_elias_fano();
    Q = mds_elias_fano();
    rho.clear();
    I_copy.clear();

    if (delete_I) I.clear();

    if (log) {
        log_message("move data structure built");
        log_runtime(time_start);
        if (mf != nullptr)
            *mf << " time_total=" << time_diff_ns(time_start, time);
    }

    this->mf = nullptr;
}

// ############################# MAXIMUM INTERVAL LENGTH #############################

template <move_pos_encoding_t enc, move_mode_t mode>
uint8_t move_data_structure<enc, mode>::construction::choose_omega_offs(uint8_t omega_default, const mds_params& params)
    requires(enc == POS)
{
    const uint8_t max_widening = params.max_omega_offs_widening;
    const uint64_t min_block_reduction_percent = params.min_block_reduction_percent;
    std::array<uint8_t, max_extra_rows> extra_widths {};
    std::array<uint8_t, max_extra_rows> extra_widths_inv {};
    for (uint8_t j = 0; j < row_widths.size(); j++) extra_widths[j] = row_widths[j];
    for (uint8_t j = 0; j < row_widths_inv.size(); j++) extra_widths_inv[j] = row_widths_inv[j];

    struct layout_t {
        uint64_t blocks = 0; // number of blocks of the input and of the output intervals
        uint8_t cost_fwd = 0; // cost of reading a record of the input intervals (see move_input_interval_blocks::record_read_cost)
        uint8_t cost_inv = 0; // cost of reading a record of the output intervals (only with an inverse)
    };

    // returns the layout of the blocks for num_intervals intervals of length at most l_max_ and the width omega
    // of D_offs
    auto layout_of = [&](uint8_t omega, uint64_t num_intervals, uint64_t l_max_) {
        move_data_structure probe;
        probe.n = n;
        probe.k_ = num_intervals;
        probe.omega_p = std::bit_width(n);
        probe.omega_offs = omega;
        probe.num_extra_rows = uint8_t(row_widths.size());
        probe.compute_layout(l_max_, extra_widths, l_max_, omega, uint8_t(row_widths_inv.size()), extra_widths_inv);
        layout_t lay { .blocks = probe.input_blocks.data_blocks(), .cost_fwd = probe.input_blocks.record_read_cost() };

        if constexpr (has_inverse) {
            lay.blocks += probe.output_blocks.data_blocks();
            lay.cost_inv = probe.output_blocks.record_read_cost();
        }

        return lay;
    };

    auto cap_of = [](uint8_t omega) { return (uint64_t { 1 } << omega) - 1; };

    // return the default width, if even with k intervals (i.e., without any split) each larger width
    // increases the cost of reading a record
    layout_t lay_default = layout_of(omega_default, k, cap_of(omega_default));
    bool has_candidate = false;

    for (uint8_t d = 1; d <= max_widening; d++) {
        layout_t lay = layout_of(omega_default + d, k, cap_of(omega_default + d));
        has_candidate = has_candidate || (lay.cost_fwd <= lay_default.cost_fwd && lay.cost_inv <= lay_default.cost_inv);
    }

    if (!has_candidate) return omega_default;

    // num[d] = number of intervals after splitting the intervals of I at the multiples of l_max for the width
    // omega_default + d (without balancing); max_len = maximum interval length in I
    std::vector<uint64_t> num(max_widening + 1, 0);
    uint64_t max_len = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<uint64_t> num_t(max_widening + 1, 0);
        uint64_t max_len_t = 0;

        #pragma omp for
        for (uint64_t x = 0; x < k; x++) {
            const uint64_t len = (x + 1 < k ? I.get<0>(x + 1) : n) - I.get<0>(x);
            max_len_t = std::max(max_len_t, len);

            for (uint8_t d = 0; d <= max_widening; d++) {
                const uint64_t cap = cap_of(omega_default + d);
                num_t[d] += len <= cap ? 1 : 1 + (len - 1) / cap;
            }
        }

        #pragma omp critical
        {
            for (uint8_t d = 0; d <= max_widening; d++) num[d] += num_t[d];
            max_len = std::max(max_len, max_len_t);
        }
    }

    layout_t base = layout_of(omega_default, num[0], std::min(max_len, cap_of(omega_default)));
    uint8_t best = omega_default;
    uint64_t best_blocks = base.blocks;

    for (uint8_t d = 1; d <= max_widening; d++) {
        const uint8_t omega = omega_default + d;
        layout_t lay = layout_of(omega, num[d], std::min(max_len, cap_of(omega)));
        if (lay.cost_fwd > base.cost_fwd || lay.cost_inv > base.cost_inv) continue;

        // choose a larger width only if it reduces the number of blocks by more than min_block_reduction_percent %
        if (100 * lay.blocks < (100 - min_block_reduction_percent) * best_blocks) {
            best = omega;
            best_blocks = lay.blocks;
        }
    }

    if (mf != nullptr) *mf << " omega_offs_default=" << int(omega_default) << " omega_offs=" << int(best);
    return best;
}

// ############################# SORTING THE PAIRS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::order_pairs()
{
    log_phase_start(log, time, "sorting the pairs by q");

    // build P (before I is sorted by q)
    step_begin();
    P = mds_elias_fano([&](uint64_t x) { return I.template get<0>(x); }, k, n, num_threads);
    step_end("step_ef_p");
    step_begin();

    if (q_order_keys != nullptr) {
        // the order by q is the stable order by q_order_keys (see mds_params): counting sort of the pairs (x, q_x)
        // into I_copy in O(k + 256 p) time, p = number of threads; start[256 t + c] = number of pairs with the key c
        // in the part of thread t, then the first position in I_copy of these pairs
        constexpr uint64_t num_keys = 1 << (8 * sizeof(uint8_t));
        I_copy.reset(k, n);
        std::vector<uint64_t> start(num_keys * num_threads, 0);

        #pragma omp parallel num_threads(num_threads)
        {
            const uint64_t t = omp_get_thread_num();
            const uint64_t T = omp_get_num_threads();
            const uint64_t x_beg = t * k / T;
            const uint64_t x_end = (t + 1) * k / T;
            uint64_t* start_t = start.data() + num_keys * t;
            for (uint64_t x = x_beg; x < x_end; x++) start_t[q_order_keys[x]]++;

            #pragma omp barrier
            #pragma omp single
            {
                // prefix sums over the keys and, within a key, over the threads
                uint64_t before = 0;

                for (uint64_t c = 0; c < num_keys; c++) {
                    for (uint64_t u = 0; u < T; u++) {
                        const uint64_t num = start[num_keys * u + c];
                        start[num_keys * u + c] = before;
                        before += num;
                    }
                }
            }

            for (uint64_t x = x_beg; x < x_end; x++) {
                const uint64_t y = start_t[q_order_keys[x]]++;
                I_copy.template set<0>(y, x);
                I_copy.template set<1>(y, I.template get<1>(x));
            }
        }

        if (delete_I) I.clear();
        pairs_by_q = &I_copy;
        step_end("step_sort_q");
    } else {
        // replace the pairs (p_x, q_x) by (x, q_x)
        if (delete_I) {
            pairs_by_q = &I;

            #pragma omp parallel for num_threads(num_threads)
            for (uint64_t x = 0; x < k; x++) pairs_by_q->template set<0>(x, x);
        } else {
            I_copy.reset(k, n);

            #pragma omp parallel for num_threads(num_threads)
            for (uint64_t x = 0; x < k; x++) {
                I_copy.template set<0>(x, x);
                I_copy.template set<1>(x, I.template get<1>(x));
            }

            pairs_by_q = &I_copy;
        }

        step_end("step_relabel");
        step_begin();

        // sort the pairs by q
        pairs_by_q->template sort_by<1>(num_threads);

        step_end("step_sort_q");
    }

    step_begin();
    Q = mds_elias_fano([&](uint64_t y) { return pairs_by_q->template get<1>(y); }, k, n, num_threads);
    step_end("step_ef_q");
    step_begin();

    // build rho
    if constexpr (has_inverse) {
        rho.reset(k, k);

        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t y = 0; y < k; y++) rho.set(pair_of_output(y), y);
    }

    step_end("step_rho");
    write_steps();
    log_phase_end(log, time, mf, "time_order_pairs");
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_CONSTRUCTION_TPP
