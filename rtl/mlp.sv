// Two-layer MLP: K1 inputs -> H hidden (ReLU, requant to int8) -> O outputs.
// Both layers are the same dense module. Layer 1's int8 output feeds layer 2
// directly. pred[m] is the argmax of example m's outputs (ties go to index 0).
module mlp #(
    parameter int N  = 4,
    parameter int M  = 5,    // examples in the batch
    parameter int K1 = 20,
    parameter int H  = 16,
    parameter int O  = 2,
    parameter W1_FILE = "",
    parameter B1_FILE = "",
    parameter W2_FILE = "",
    parameter B2_FILE = ""
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [7:0]  x    [M*K1],
    output logic signed [31:0] z2   [M*O],
    output logic [$clog2(O > 1 ? O : 2)-1:0] pred [M],
    output logic done
);

logic signed [31:0] z1   [M*H];
logic signed [7:0]  a1_q [M*H];
logic signed [7:0]  z2_q [M*O];
logic done1, done1_d, done2, done2_d, start2;

dense #(.N(N), .M(M), .K(K1), .COLS(H), .APPLY_ACT(1),
        .W_FILE(W1_FILE), .B_FILE(B1_FILE)) layer1 (
    .clk(clk), .rst(rst), .start(start), .x(x), .y(z1), .y_q(a1_q), .done(done1)
);

dense #(.N(N), .M(M), .K(H), .COLS(O), .APPLY_ACT(0),
        .W_FILE(W2_FILE), .B_FILE(B2_FILE)) layer2 (
    .clk(clk), .rst(rst), .start(start2), .x(a1_q), .y(z2), .y_q(z2_q), .done(done2)
);

// dense holds done high until its next start, so act on the rising edges
always_ff @(posedge clk) begin
    done1_d <= rst ? 1'b0 : done1;
    done2_d <= rst ? 1'b0 : done2;
end
assign start2 = done1 && !done1_d;

always_comb begin
    for (int m = 0; m < M; m++) begin
        pred[m] = 0;
        for (int o = 1; o < O; o++)
            if (z2[m*O + o] > z2[m*O + pred[m]])
                pred[m] = o;
    end
end

always_ff @(posedge clk) begin
    if (rst || start)
        done <= 0;
    else if (done2 && !done2_d)
        done <= 1;
end

endmodule
