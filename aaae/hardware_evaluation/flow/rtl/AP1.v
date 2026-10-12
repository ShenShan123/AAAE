`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/23 15:05:27
// Design Name: 
// Module Name: AP1
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


module AP1(
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
 
    assign part1 = a & b;
    assign part2 = c & d;
    assign part3 = a | b;
    assign part4 = c | d;
    assign S = ((((part3 & ~part1)) | ((part4 & ~part2))) & ~(((part3 & ~part1)) & ((part4 & ~part2)))) | (part1 & part2);
    assign C = part1 | part2 | (part3 & part4); 
endmodule
