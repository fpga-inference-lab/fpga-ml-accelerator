// Fully-connected layer on the N x N systolic array:  y = x @ W + bias * 64
//
// x is M x K (a batch of M examples), W is K x COLS, y is M x COLS.
// The output is computed in N x N tiles. For each tile the array is cleared,
// then the K-long contraction is streamed in with the usual diagonal skew
// (row i of x delayed by i cycles, column j of W delayed by j cycles).
// Once the tile has drained, its N*N results get bias (and optionally ReLU +
// int8 requant) and are written to y one per cycle.
//
// Weights and bias are ROMs loaded with $readmemh (one int8 per line, row-major).
module dense #(
    parameter int N = 4,
    parameter int M = 5,           // examples in the batch
    parameter int K = 20,          // inputs per example
    parameter int COLS = 16,       // outputs per example
    parameter bit APPLY_ACT = 1,   // 1: ReLU then requant to int8
    parameter W_FILE = "",
    parameter B_FILE = ""
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0]  x   [M*K],
    output logic signed [31:0] y   [M*COLS],   // bias added (and ReLU if APPLY_ACT)
    output logic signed [7:0]  y_q [M*COLS],   // y requantized: (y + 32) >>> 6, saturated
    output logic done
);

localparam int ROW_TILES = (M + N - 1) / N;
localparam int COL_TILES = (COLS + N - 1) / N;
localparam int FEED_LAST = K + 2*N - 2;   // last cycle the corner PE is still accumulating

logic signed [7:0] w_mem [K*COLS];
logic signed [7:0] b_mem [COLS];

initial begin
    $readmemh(W_FILE, w_mem);
    $readmemh(B_FILE, b_mem);
end

typedef enum logic [2:0] {IDLE, CLEAR, FEED, WRITE, FINISH} state_t;
state_t state;

// Counters are narrow and unsigned (not int) so that / N, % N and * COLS reduce
// to wiring and small adders in synthesis instead of 32-bit signed arithmetic.
logic [7:0]  rt, ct;   // current row tile, column tile
logic [15:0] t;        // feed cycle within a tile
logic [7:0]  w;        // result being written within a tile

logic signed [7:0]  a_feed [N];
logic signed [7:0]  b_feed [N];
logic signed [31:0] c [N][N];

systolic_array #(.N(N)) array (
    .clk     (clk),
    .rst     (rst || state == CLEAR),
    .en      (1'b1),
    .a_west  (a_feed),
    .b_north (b_feed),
    .c       (c)
);

// Write-back is a two-stage pipeline so each stage fits in one 100 MHz cycle:
//   stage 1 (WRITE state): pick result w from the array, add bias, ReLU -> z_r
//   stage 2 (one cycle later): requant z_r and store both into y / y_q
logic [7:0]  wi, wj;
logic [15:0] row, col;
logic signed [31:0] bias, z;

assign wi  = w / N;
assign wj  = w % N;
assign row = rt*N + wi;
assign col = ct*N + wj;

always_comb begin
    bias = (col < COLS) ? b_mem[col] : 0;
    z = c[wi][wj] + bias * 64;
    if (APPLY_ACT && z < 0)
        z = 0;
end

logic               wr_valid;
logic [15:0]        wr_addr;
logic signed [31:0] z_r, zr;

assign zr = (z_r + 32) >>> 6;

always_ff @(posedge clk) begin
    if (rst)
        wr_valid <= 0;
    else begin
        wr_valid <= (state == WRITE) && row < M && col < COLS;
        wr_addr  <= row*COLS + col;
        z_r      <= z;
        if (wr_valid) begin
            y[wr_addr]   <= z_r;
            y_q[wr_addr] <= (zr > 127) ? 8'sd127 : (zr < -128) ? -8'sd128 : zr[7:0];
        end
    end
end

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        done  <= 0;
        rt <= 0; ct <= 0; t <= 0; w <= 0;
        for (int i = 0; i < N; i++) begin
            a_feed[i] <= 0;
            b_feed[i] <= 0;
        end
    end else begin
        case (state)
            IDLE: begin
                if (start) begin
                    done  <= 0;
                    rt <= 0; ct <= 0;
                    state <= CLEAR;
                end
            end

            CLEAR: begin
                t <= 0;
                for (int i = 0; i < N; i++) begin
                    a_feed[i] <= 0;
                    b_feed[i] <= 0;
                end
                state <= FEED;
            end

            FEED: begin
                for (int i = 0; i < N; i++) begin
                    int k, r, cc;
                    k  = t - i;
                    r  = rt*N + i;
                    cc = ct*N + i;
                    a_feed[i] <= (k >= 0 && k < K && r  < M)    ? x[r*K + k]        : 0;
                    b_feed[i] <= (k >= 0 && k < K && cc < COLS) ? w_mem[k*COLS + cc] : 0;
                end
                t <= t + 1;
                if (t == FEED_LAST) begin
                    w <= 0;
                    state <= WRITE;
                end
            end

            WRITE: begin
                w <= w + 1;
                if (w == N*N - 1) begin
                    if (ct == COL_TILES - 1) begin
                        ct <= 0;
                        if (rt == ROW_TILES - 1)
                            state <= FINISH;
                        else begin
                            rt <= rt + 1;
                            state <= CLEAR;
                        end
                    end else begin
                        ct <= ct + 1;
                        state <= CLEAR;
                    end
                end
            end

            FINISH: begin
                done  <= 1;   // the last result is written by stage 2 on this same edge
                state <= IDLE;
            end
        endcase
    end
end

endmodule
