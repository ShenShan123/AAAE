module CLA16bit (
    input [15:0] A, 
    input [15:0] B, 
    input Cin, 
    output [15:0] Sum, 
    output Cout 
);

wire c1, c2, c3;

CLA4bit cla1
(
    .A(A[3:0]),
    .B(B[3:0]),
    .Cin(Cin),
    .Sum(Sum[3:0]),
    .Cout(c1)
);

CLA4bit cla2
(
    .A(A[7:4]),
    .B(B[7:4]),
    .Cin(c1),
    .Sum(Sum[7:4]),
    .Cout(c2)
);

CLA4bit cla3
(
    .A(A[11:8]),
    .B(B[11:8]),
    .Cin(c2),
    .Sum(Sum[11:8]),
    .Cout(c3)
);

CLA4bit cla4
(
    .A(A[15:12]),
    .B(B[15:12]),
    .Cin(c3),
    .Sum(Sum[15:12]),
    .Cout(Cout)
);


endmodule
