`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/04/07 13:56:56
// Design Name: 
// Module Name: Pro
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module antonio(
    input a,
    input b,
    input c,
    input d,
    output S,
    output C
    );
    
    wire w1;
    wire w2;
    wire w3;
    
    assign w1 = a | b;
    assign w2 = c | (a & b);
    assign w3 = d;
    
    Full_adder Full_adder(
        .a(w1),
        .b(w2),
        .CIN(w3),
        .S(S),
        .C(C)
    );
endmodule
