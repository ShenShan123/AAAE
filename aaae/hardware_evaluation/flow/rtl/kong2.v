`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/04/04 10:47:35
// Design Name: 
// Module Name: Pro3
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


module kong2(
   input a,
   input b,
   input c,
   input d,
   output S,
   output C
    );
    wire part1;
    wire part2;
    wire part3;
    wire part4;
    wire part5;
    wire part6;
    
    assign part1 = a ^ b;
    assign part2 = c ^ d;
    assign part3 = a | b;
    assign part4 = c | d;
    assign part5 = a & b;
    assign part6 = c & d;
    assign S = (part1 & ~part2) | (~part1 & part4);
    assign C =  ~((~part3 & ~part6) | (~part4 & ~part5));
endmodule
