// Latency-optimized version of the same MLP, for one example at a time.
//
// Fully unrolled: every one of the 20*16 + 16*2 multiplies has its own hardware,
// and the trained weights are built into the logic as constants (loaded from the
// hex files at synthesis), so each multiply is a cheap shift-and-add circuit.
// Each neuron then sums its products with a balanced adder tree. Pipeline:
//
//   products:  p1[j][k] = x[k] * W1[k][j]                  (320 multiplies at once)
//   tree:      z1[j] = sum_k p1[j][k] + b1[j]*64           (adder_tree, LAT1 cycles)
//   requant:   a1[j] = min((relu(z1[j]) + 32) >> 6, 127)
//   products:  p2[o][j] = a1[j] * W2[j][o]                 (32 multiplies at once)
//   tree:      z2[o] = sum_j p2[o][j] + b2[o]*64           (adder_tree, LAT2 cycles)
//
// Same ports and same integer math as mlp.sv with M = 1, so top.sv can use either.
// done goes high when z2 is ready and stays high until the next start.
module fast_mlp #(
    parameter int K1 = 20,
    parameter int H  = 16,
    parameter int O  = 2,
    parameter int TREE_REG_EVERY = 3,   // adder-tree levels between pipeline registers
    parameter W1_FILE = "",
    parameter B1_FILE = "",
    parameter W2_FILE = "",
    parameter B2_FILE = ""
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0]  x    [K1],
    output logic signed [31:0] z2   [O],
    output logic [$clog2(O > 1 ? O : 2)-1:0] pred [1],
    output logic done
);

// |sums| stay under 20 * 128 * 128 + 128 * 64 < 2^19, so 20 bits signed is enough
localparam int W = 20;
localparam int LAT1 = $clog2(K1 + 1) / TREE_REG_EVERY;
localparam int LAT2 = $clog2(H + 1) / TREE_REG_EVERY;
localparam int PIPE = 4 + LAT1 + LAT2;   // products, requant, products, output register + trees

logic signed [7:0] w1 [K1*H];
logic signed [7:0] b1 [H];
logic signed [7:0] w2 [H*O];
logic signed [7:0] b2 [O];

initial begin
    $readmemh(W1_FILE, w1);
    $readmemh(B1_FILE, b1);
    $readmemh(W2_FILE, w2);
    $readmemh(B2_FILE, b2);
end

logic signed [W-1:0] terms1 [H][K1+1];   // products + bias for each hidden neuron
logic signed [W-1:0] z1 [H];
logic        [6:0]   a1 [H];             // post-ReLU and requant, so 0..127
logic signed [W-1:0] terms2 [O][H+1];
logic signed [W-1:0] z2_sum [O];
logic signed [W-1:0] z2_r [O];
logic [PIPE-1:0] valid;                  // valid[s] = pipeline stage s holds a live example

always_ff @(posedge clk) begin
    if (rst)
        valid <= 0;
    else
        valid <= {valid[PIPE-2:0], start};

    for (int j = 0; j < H; j++) begin
        for (int k = 0; k < K1; k++)
            terms1[j][k] <= x[k] * w1[k*H + j];
        terms1[j][K1] <= b1[j] * 64;
    end

    for (int j = 0; j < H; j++) begin
        logic signed [W-1:0] r;
        r = (z1[j] < 0) ? '0 : (z1[j] + 32) >>> 6;
        a1[j] <= (r > 127) ? 7'd127 : r[6:0];
    end

    // a1 is unsigned, so widen it with a 0 sign bit before the signed multiply
    for (int o = 0; o < O; o++) begin
        for (int j = 0; j < H; j++)
            terms2[o][j] <= $signed({1'b0, a1[j]}) * w2[j*O + o];
        terms2[o][H] <= b2[o] * 64;
    end

    z2_r <= z2_sum;
end

genvar j, o;
generate
    for (j = 0; j < H; j++) begin : hidden
        adder_tree #(.N(K1 + 1), .W(W), .REG_EVERY(TREE_REG_EVERY)) tree (
            .clk(clk), .in(terms1[j]), .sum(z1[j])
        );
    end
    for (o = 0; o < O; o++) begin : out
        adder_tree #(.N(H + 1), .W(W), .REG_EVERY(TREE_REG_EVERY)) tree (
            .clk(clk), .in(terms2[o]), .sum(z2_sum[o])
        );
    end
endgenerate

always_comb
    for (int o = 0; o < O; o++)
        z2[o] = z2_r[o];     // sign-extends to 32 bits

always_comb begin
    pred[0] = 0;
    for (int o = 1; o < O; o++)
        if (z2[o] > z2[pred[0]])
            pred[0] = o;
end

always_ff @(posedge clk) begin
    if (rst || start)
        done <= 0;
    else if (valid[PIPE-1])
        done <= 1;
end

endmodule
