module CLA4bit (
    input [3:0] A, // 输入 A
    input [3:0] B, // 输入 B
    input Cin, // 输入进位
    output [3:0] Sum, // 输出和
    output Cout // 输出最高位的进位
);

wire [3:0] C;


    assign Sum[0] = A[0] ^ B[0] ^ Cin;
    assign Sum[1] = A[1] ^ B[1] ^ C[0];
    assign Sum[2] = A[2] ^ B[2] ^ C[1];
    assign Sum[3] = A[3] ^ B[3] ^ C[2];
    assign C[0] = (A[0] & B[0]) || ((A[0] ^ B[0]) & Cin);
    assign C[1] = (A[1] & B[1]) || ((A[1] ^ B[1]) & C[0]);
    assign C[2] = (A[2] & B[2]) || ((A[2] ^ B[2]) & C[1]);
    assign C[3] = (A[3] & B[3]) || ((A[3] ^ B[3]) & C[2]);
    assign Cout = C[3];

endmodule
