// Balanced, pipelined adder tree: sum = in[0] + in[1] + ... + in[N-1].
//
// Adds pairs, then pairs of pairs, so N inputs take ceil(log2(N)) adder levels
// instead of a chain of N-1. Inputs are padded with zeros up to a power of two
// (synthesis removes those adders). A register is placed after every REG_EVERY
// levels; LATENCY says how many clock cycles that adds.
module adder_tree #(
    parameter int N = 16,
    parameter int W = 20,           // width of inputs, partial sums and the result
    parameter int REG_EVERY = 3
)(
    input  logic                clk,
    input  logic signed [W-1:0] in [N],
    output logic signed [W-1:0] sum
);

localparam int LEVELS = (N > 1) ? $clog2(N) : 0;
localparam int NP = 1 << LEVELS;
localparam int LATENCY = LEVELS / REG_EVERY;

genvar l, i;
generate
    for (l = 0; l <= LEVELS; l++) begin : lv
        logic signed [W-1:0] s [NP >> l];

        if (l == 0) begin : leaves
            for (i = 0; i < NP; i++) begin : leaf
                if (i < N) assign s[i] = in[i];
                else       assign s[i] = '0;
            end
        end else if (l % REG_EVERY == 0) begin : registered
            always_ff @(posedge clk)
                for (int k = 0; k < (NP >> l); k++)
                    s[k] <= lv[l-1].s[2*k] + lv[l-1].s[2*k+1];
        end else begin : comb
            always_comb
                for (int k = 0; k < (NP >> l); k++)
                    s[k] = lv[l-1].s[2*k] + lv[l-1].s[2*k+1];
        end
    end
endgenerate

assign sum = lv[LEVELS].s[0];

endmodule
