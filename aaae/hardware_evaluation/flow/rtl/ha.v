`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/08/05 11:01:53
// Design Name: 
// Module Name: ACCI3
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


module ha(
    input a,
    input b,
    input c,
    input d,
    output S,
    output C
    );
    
    wire w1;
    wire w2;
    assign w1 = a ^ b;
    assign w2 = c | d;
    assign S = w1 ^ w2;
    assign C = (a & b) | (w1 & w2);
endmodule
