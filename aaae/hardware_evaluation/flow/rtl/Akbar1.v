`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/11/08 10:18:50
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


module akbar1(
    input a,
    input b,
    input c,
    input d,
    output S,
    output C
    );
    assign S = ~((~(a ^ b)) & (~(c ^ d)));
    assign C = d;
endmodule
