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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_FILL_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_FILL_TPP

#include <move_data_structure/construction/construction.hpp>

// ############################# BUILDING THE MOVE DATA STRUCTURE #############################

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::mark_pieces()
{
    log_phase_start(log, time, "building the piece marks");
    uint64_t chunks = num_threads == 1 ? 1 : chunks_per_thread * num_threads;
    // pieces_before[c + 1] = number of pieces of the pairs of chunk c, then the number of pieces of the chunks before c + 1
    std::vector<uint64_t> pieces_before(chunks + 1, 0);
    std::vector<uint64_t> chunk_max_len(chunks, 0); // chunk_max_len[c] = maximum length of a piece of chunk c

    // returns the number of pieces of a pair with an interval of length len and the split offsets pair_splits; sets
    // max_len to the maximum of max_len and the lengths of the pieces
    auto pieces_of = [&](uint64_t len, std::pair<uint64_t, uint64_t> pair_splits, uint64_t& max_len) {
        if (pair_splits.first == pair_splits.second) {
            max_len = std::max(max_len, std::min(len, l_max));
            return 1 + (len - 1) / l_max;
        }

        pieces_t pieces;
        pieces.init(len, l_max, splits.offsets, pair_splits);
        uint64_t num = 0;

        do {
            num++;
            max_len = std::max(max_len, pieces.end - pieces.beg);
        } while (pieces.next());

        return num;
    };

    step_begin();

    // first_p: the rank of the first piece of the pair x is the number of pieces of the pairs before x; pass 0
    // counts the pieces of each chunk, pass 1 sets the bits
    for (uint8_t pass = 0; pass < 2; pass++) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (uint64_t c = 0; c < chunks; c++) {
            uint64_t x_beg = (c * k) / chunks;
            uint64_t x_end = ((c + 1) * k) / chunks;
            uint64_t num = pass == 0 ? 0 : pieces_before[c];
            uint64_t max_len = 0;
            if (x_beg == x_end) continue;
            mds_elias_fano::cursor p_cursor = P.cursor_at(x_beg);
            uint64_t p_x = p_cursor.value();

            for (uint64_t x = x_beg; x < x_end; x++) {
                uint64_t p_next = n;

                if (x + 1 < k) {
                    p_cursor.next();
                    p_next = p_cursor.value();
                }

                if (pass == 1) first_p.set_parallel(num);
                num += pieces_of(p_next - p_x, splits.splits_of(x), max_len);
                p_x = p_next;
            }

            if (pass == 0) {
                pieces_before[c + 1] = num;
                chunk_max_len[c] = max_len;
            }
        }

        if (pass == 0) {
            for (uint64_t c = 0; c < chunks; c++) pieces_before[c + 1] += pieces_before[c];
            k_ = pieces_before[chunks];
            mds_check_num_intervals(k_);
            first_p.resize(k_ + 1, num_threads);
        }
    }

    first_p.set_parallel(k_);
    first_p.build(num_threads);
    step_end("step_first_p");
    step_begin();
    max_piece_length = *std::max_element(chunk_max_len.begin(), chunk_max_len.end());
    std::fill(pieces_before.begin(), pieces_before.end(), 0);

    // first_q: the same in the order of q
    for (uint8_t pass = 0; pass < 2; pass++) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (uint64_t c = 0; c < chunks; c++) {
            uint64_t y_beg = (c * k) / chunks;
            uint64_t y_end = ((c + 1) * k) / chunks;
            uint64_t num = pass == 0 ? 0 : pieces_before[c];
            uint64_t max_len = 0;

            for (uint64_t y = y_beg; y < y_end; y++) {
                if (pass == 1) first_q.set_parallel(num);
                uint64_t len = q_of_output(y + 1) - q_of_output(y);
                num += pieces_of(len, splits.output_has_splits(y) ? splits.splits_of(pair_of_output(y)) : std::pair<uint64_t, uint64_t> { 0, 0 }, max_len);
            }

            if (pass == 0) pieces_before[c + 1] = num;
        }

        if (pass == 0) {
            for (uint64_t c = 0; c < chunks; c++) pieces_before[c + 1] += pieces_before[c];
            if (pieces_before[chunks] != k_) throw std::logic_error("move data structure construction: the numbers of input and output interval pieces differ");
            first_q.resize(k_ + 1, num_threads);
        }
    }

    first_q.set_parallel(k_);
    first_q.build(num_threads);
    step_end("step_first_q");
    write_steps();

    if (mf != nullptr) *mf << " k=" << k << " k_=" << k_;
    log_phase_end(log, time, mf, "time_mark_pieces");
    if (log) std::cout << "k' = " << format_number(k_) << ", k'/k = " << std::round(100.0 * k_ / k) / 100.0 << std::endl;
}

template <move_pos_encoding_t enc, move_mode_t mode>
template <typename fnc_t>
void move_data_structure<enc, mode>::construction::for_each_output_piece(uint64_t y_beg, uint64_t y_end, fnc_t f) const
{
    if (y_beg >= y_end) return;
    uint64_t y = first_q.rank(y_beg + 1) - 1;
    output_cursor oc { this };
    oc.go_to_output(y);
    oc.pieces.seek_index(y_beg - first_q.select(y));
    input_cursor ic { this };
    ic.seek(oc.beg());
    uint64_t x_ic = first_p.select(ic.x) + ic.pieces.piece_idx; // rank of the piece of ic among the input interval pieces

    // the ranks of the first input interval pieces of the pairs are computed for batch output intervals at a time;
    // x_first[i] = rank of the first input interval piece of the pair of the (y_batch + i)-th output interval
    constexpr uint64_t batch = 32;
    uint64_t x_first[batch];
    uint64_t y_batch = y;

    auto fill_batch = [&]() {
        uint64_t num = std::min(batch, k - y_batch);
        for (uint64_t i = 0; i < num; i++) x_first[i] = first_p.select(pair_of_output(y_batch + i));
    };

    fill_batch();

    for (uint64_t y_ = y_beg; y_ < y_end; y_++) {
        uint64_t t = oc.beg();

        while (ic.end() <= t) {
            ic.next();
            x_ic++;
        }

        f(y_, x_first[oc.y - y_batch] + oc.pieces.piece_idx, t, x_ic, t - ic.beg(), oc.y, oc.pieces.beg);

        if (!oc.pieces.next()) {
            oc.go_to_output(oc.y + 1);

            if (oc.y - y_batch == batch && oc.y < k) {
                y_batch = oc.y;
                fill_batch();
            }
        }
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
uint64_t move_data_structure<enc, mode>::construction::start_of_input_piece(uint64_t x_) const
{
    uint64_t x = first_p.rank(x_ + 1) - 1;
    input_cursor ic { this };
    ic.go_to_pair(x);
    ic.pieces.seek_index(x_ - first_p.select(x));
    return ic.beg();
}

template <move_pos_encoding_t enc, move_mode_t mode>
template <typename fnc_t>
void move_data_structure<enc, mode>::construction::for_each_input_piece(uint64_t x_beg, uint64_t x_end, uint64_t y_start, fnc_t f) const
{
    if (x_beg >= x_end) return;
    uint64_t x = first_p.rank(x_beg + 1) - 1;
    input_cursor ic { this };
    ic.go_to_pair(x);
    ic.pieces.seek_index(x_beg - first_p.select(x));
    output_cursor oc { this };
    uint64_t y_oc = 0; // rank of the piece of oc among the output interval pieces

    // the ranks of the first output interval pieces of the pairs are computed for batch pairs at a time
    constexpr uint64_t batch = 32;
    uint64_t y_first[batch]; // y_first[i] = rank of the first output interval piece of the pair x_batch + i
    uint64_t x_batch = x;

    auto fill_batch = [&]() {
        uint64_t num = std::min(batch, k - x_batch);
        for (uint64_t i = 0; i < num; i++) y_first[i] = first_q.select(rho_of(x_batch + i));
    };

    if constexpr (has_inverse) {
        oc.seek(y_start, ic.beg());
        y_oc = first_q.select(oc.y) + oc.pieces.piece_idx;
        fill_batch();
    }

    for (uint64_t x_ = x_beg; x_ < x_end; x_++) {
        uint64_t s = ic.beg();

        if constexpr (has_inverse) {
            while (oc.end() <= s) {
                oc.next();
                y_oc++;
            }

            f(x_, s, ic.end() - s, y_first[ic.x - x_batch] + ic.pieces.piece_idx, y_oc, s - oc.beg(), ic.x, ic.pieces.beg);
        } else {
            f(x_, s, ic.end() - s, 0, 0, 0, ic.x, ic.pieces.beg);
        }

        if (!ic.pieces.next()) {
            ic.go_to_next_pair();

            if constexpr (has_inverse) {
                if (ic.x - x_batch == batch && ic.x < k) {
                    x_batch = ic.x;
                    fill_batch();
                }
            }
        }
    }
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::fill_blocks() requires(enc == POS)
{
    log_phase_start(log, time, "building the blocks");

    mds.n = n;
    mds.k_ = k_;
    mds.num_extra_rows = uint8_t(row_widths.size());
    mds.omega_p = std::bit_width(n);
    mds.omega_idx = std::bit_width(k_);
    std::array<uint8_t, max_extra_rows> extra_widths {};
    std::array<uint8_t, max_extra_rows> extra_widths_inv {};
    for (uint8_t j = 0; j < row_widths.size(); j++) extra_widths[j] = row_widths[j];
    for (uint8_t j = 0; j < row_widths_inv.size(); j++) extra_widths_inv[j] = row_widths_inv[j];

    // max_piece_length is the maximum length of the input and of the output intervals
    mds.compute_layout(max_piece_length, extra_widths, max_piece_length, mds.omega_offs, uint8_t(row_widths_inv.size()), extra_widths_inv);
    uint64_t records_fwd = mds.input_blocks.intervals_per_block();
    uint64_t blocks_fwd = mds.input_blocks.data_blocks();
    uint64_t chunks = std::min<uint64_t>(blocks_fwd, num_threads == 1 ? 1 : chunks_per_thread * num_threads);

    // y_start[c] = rank of the output interval containing the starting position of the first input interval
    // piece of the chunk c (computed before Q is freed)
    std::vector<uint64_t> y_start(chunks, 0);

    if constexpr (has_inverse) {
        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t c = 0; c < chunks; c++) {
            uint64_t x_ = ((c * blocks_fwd) / chunks) * records_fwd;
            if (x_ < k_) y_start[c] = output_containing(start_of_input_piece(x_));
        }
    }

    Q = mds_elias_fano();

    move_input_interval_blocks& input_blocks = mds.input_blocks;
    step_begin();
    input_blocks.allocate(num_threads);
    if constexpr (has_inverse) mds.output_blocks.allocate(num_threads);
    step_end("step_allocate");

    if (pi_out != nullptr) {
        resize_no_init(*pi_out, k_ + 1);
        (*pi_out)[k_] = k_;
    }

    // header values of the sentinel blocks of the input and of the output intervals
    uint64_t sentinel_fwd = 0;
    uint64_t sentinel_inv = 0;

    if constexpr (has_inverse) {
        sentinel_fwd = mds.output_blocks.slot_of(k_);
        sentinel_inv = input_blocks.slot_of(k_);
    }

    /* pass over the input interval pieces (each chunk is a range of whole blocks): writes the block headers, the
       starting positions and the rows given in the order of p to the blocks of the input intervals and (with an
       inverse) idx_inv and offs_inv to the blocks of the output intervals */
    step_begin();

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t b_beg = (c * blocks_fwd) / chunks;
        uint64_t b_end = ((c + 1) * blocks_fwd) / chunks;
        uint64_t x_end = std::min(b_end * records_fwd, k_);
        uint64_t start = 0;

        // the atomic write functions are only used with more than one thread
        mds_deferred_writes inv_writes { [&](uint64_t y_, uint64_t e, uint64_t offs, uint64_t) {
            if constexpr (has_inverse) {
                if (num_threads == 1) mds.output_blocks.write_idx_offs(y_, e, offs);
                else mds.output_blocks.write_idx_offs_parallel(y_, e, offs);
            }
        } };

        for_each_input_piece(b_beg * records_fwd, x_end, y_start[c], [&](uint64_t x_, uint64_t s, uint64_t, uint64_t y_, uint64_t e,
            uint64_t offs, uint64_t x, uint64_t d) {
            uint64_t b = x_ / records_fwd;

            if (x_ == b * records_fwd) {
                start = s;
                uint64_t header_value = 0;
                if constexpr (has_inverse) header_value = mds.output_blocks.slot_of(e);
                input_blocks.write_header(b, s, header_value);
            }

            input_blocks.write_rel_p(x_, s - start);
            for (uint8_t i = 0; i < num_rows_in_p_order; i++) input_blocks.write_row(x_, rows_in_p_order[i], pair_row_value(rows_in_p_order[i], x, d));
            if constexpr (has_inverse) inv_writes.push(mds.output_blocks.idx_word(y_), y_, e, offs);
        });

        inv_writes.flush();

        // the records from the sentinel interval k' on get p = n, idx = k' and offs = 0
        for (uint64_t x_ = std::max(x_end, b_beg * records_fwd); x_ < b_end * records_fwd; x_++) {
            uint64_t b = x_ / records_fwd;

            if (x_ == b * records_fwd) {
                start = n;
                input_blocks.write_header(b, n, sentinel_fwd);
            }

            input_blocks.write_rel_p(x_, n - start);
            input_blocks.write_idx_offs_parallel(x_, k_, 0);
        }
    }

    input_blocks.write_sentinel(sentinel_fwd);
    step_end("step_pass_input");
    step_begin();

    /* pass over the output interval pieces: writes idx, offs and the row given in the order of q to the blocks of
       the input intervals and (with an inverse) the block headers and the starting positions to the blocks of the
       output intervals; without an inverse, each chunk is a range of output interval pieces */
    uint64_t records_inv = 1;
    uint64_t blocks_inv = 0;

    if constexpr (has_inverse) {
        records_inv = mds.output_blocks.intervals_per_block();
        blocks_inv = mds.output_blocks.data_blocks();
    } else {
        blocks_inv = k_;
    }

    chunks = std::min<uint64_t>(blocks_inv, num_threads == 1 ? 1 : chunks_per_thread * num_threads);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t b_beg = (c * blocks_inv) / chunks;
        uint64_t b_end = ((c + 1) * blocks_inv) / chunks;
        uint64_t y_end = std::min(b_end * records_inv, k_);
        uint64_t start = 0;
        mds_deferred_writes fwd_writes { [&](uint64_t x_, uint64_t idx, uint64_t offs, uint64_t row_value) {
            if (row_in_q_order == max_extra_rows) {
                if (num_threads == 1) input_blocks.write_idx_offs(x_, idx, offs);
                else input_blocks.write_idx_offs_parallel(x_, idx, offs);
            } else if (num_threads == 1) {
                input_blocks.write_idx_offs(x_, idx, offs);
                input_blocks.write_row(x_, row_in_q_order, row_value);
            } else {
                input_blocks.write_idx_offs_row_parallel(x_, idx, offs, row_in_q_order, row_value);
            }
        } };

        for_each_output_piece(b_beg * records_inv, y_end, [&](uint64_t y_, uint64_t x_, uint64_t t, uint64_t idx, uint64_t offs,
            uint64_t y, uint64_t d) {
            // the value of the row given in the order of q belongs to the pair with the y-th smallest q
            fwd_writes.push(input_blocks.idx_word(x_), x_, idx, offs, row_in_q_order == max_extra_rows ? 0 : pair_row_value(row_in_q_order, y, d));
            if (pi_out != nullptr) (*pi_out)[y_] = x_;

            if constexpr (has_inverse) {
                uint64_t b = y_ / records_inv;

                if (y_ == b * records_inv) {
                    start = t;
                    mds.output_blocks.write_header(b, t, input_blocks.slot_of(idx));
                }

                mds.output_blocks.write_rel_p(y_, t - start);
            }
        });

        fwd_writes.flush();

        if constexpr (has_inverse) {
            // the records from the sentinel interval k' on get p_inv = n and idx_inv = k'
            for (uint64_t y_ = std::max(y_end, b_beg * records_inv); y_ < b_end * records_inv; y_++) {
                uint64_t b = y_ / records_inv;

                if (y_ == b * records_inv) {
                    start = n;
                    mds.output_blocks.write_header(b, n, sentinel_inv);
                }

                mds.output_blocks.write_rel_p(y_, n - start);
                mds.output_blocks.write_idx_offs_parallel(y_, k_, 0);
            }
        }
    }

    if constexpr (has_inverse) mds.output_blocks.write_sentinel(sentinel_inv);
    step_end("step_pass_output");
    write_steps();
    log_phase_end(log, time, mf, "time_fill_blocks");
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::fill_ef() requires(enc == EF)
{
    log_phase_start(log, time, "building D_low, D_high, D_idx, D_offs");

    mds.n = n;
    mds.k_ = k_;
    mds.num_extra_rows = uint8_t(row_widths.size());
    mds.omega_p = std::bit_width(n);
    mds.omega_idx = std::bit_width(k_);
    const uint8_t omega_low = mds.omega_offs; // width of D_low
    const uint64_t mask_low = (uint64_t { 1 } << omega_low) - 1;
    interleaved_bit_aligned_vectors& D_rows = mds.D_rows;
    D_rows = interleaved_bit_aligned_vectors(mds.interleaved_widths(row_widths));
    D_rows.resize_no_init(k_ + 1, num_threads);

    // the sentinel interval k'
    D_rows.template set_parallel<0, uint64_t>(k_, n & mask_low);
    D_rows.template set_parallel<1, uint64_t>(k_, k_);
    D_rows.template set_parallel<2, uint64_t>(k_, 0);

    for_constexpr<0, max_extra_rows, 1>([&](auto j) {
        if (uint8_t(j) < row_widths.size()) D_rows.template set_parallel<num_core_rows + j, uint64_t>(k_, uint64_t(0));
    });

    Q = mds_elias_fano(); // free Q
    cache_line_rank_bit_vector& D_high = mds.D_high;
    constexpr uint64_t bits_per_line = cache_line_rank_bit_vector::bits_per_line;
    D_high.allocate(k_ + 1, num_threads);
    if (k_ % bits_per_line == 0) D_high.set_line_rank(k_ / bits_per_line, n >> omega_low);

    if (pi_out != nullptr) {
        resize_no_init(*pi_out, k_ + 1);
        (*pi_out)[k_] = k_;
    }

    uint64_t chunks = std::min<uint64_t>(k_, num_threads == 1 ? 1 : chunks_per_thread * num_threads);

    // sets the J-th extra row of the x_-th interval (J is a runtime value)
    auto set_extra_row = [&](uint8_t J, uint64_t x_, uint64_t value) {
        for_constexpr<0, max_extra_rows, 1>([&](auto j) {
            if (uint8_t(j) == J) D_rows.template set_parallel<num_core_rows + j, uint64_t>(x_, value);
        });
    };

    // pass over the input interval pieces: writes D_low, D_high and the rows given in the order of p
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint64_t c = 0; c < chunks; c++) {
        for_each_input_piece((c * k_) / chunks, ((c + 1) * k_) / chunks, 0, [&](uint64_t x_, uint64_t s, uint64_t len, uint64_t,
            uint64_t, uint64_t, uint64_t x, uint64_t dist) {
            D_rows.template set_parallel<0, uint64_t>(x_, s & mask_low);
            // set bit x' iff the high part grows from this piece to the next one (pieces are shorter than 2^omega_low)
            if ((s + len) >> omega_low != s >> omega_low) D_high.set_parallel(x_);
            if (x_ % bits_per_line == 0) D_high.set_line_rank(x_ / bits_per_line, s >> omega_low);

            for (uint8_t i = 0; i < num_rows_in_p_order; i++) set_extra_row(rows_in_p_order[i], x_, pair_row_value(rows_in_p_order[i], x, dist));
        });
    }

    // pass over the output interval pieces: writes idx, offs and the row given in the order of q
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint64_t c = 0; c < chunks; c++) {
        for_each_output_piece((c * k_) / chunks, ((c + 1) * k_) / chunks, [&](uint64_t y_, uint64_t x_, uint64_t, uint64_t idx, uint64_t offs,
            uint64_t y, uint64_t dist) {
            D_rows.template set_parallel<1, uint64_t>(x_, idx);
            D_rows.template set_parallel<2, uint64_t>(x_, offs);
            if (row_in_q_order != max_extra_rows) set_extra_row(row_in_q_order, x_, pair_row_value(row_in_q_order, y, dist));
            if (pi_out != nullptr) (*pi_out)[y_] = x_;
        });
    }

    log_phase_end(log, time, mf, "time_fill_ef");
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_FILL_TPP
