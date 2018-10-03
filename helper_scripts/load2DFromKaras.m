%
% load2DFromKaras.m
%
% Loads a matrix stored in a .karas file.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
function A = load2DFromKaras(fn)
    f = fopen(fn);
    rows = fread(f, 1, 'uint32');
    cols = fread(f, 1, 'uint32');

    A = fread(f, [cols rows], 'double');    
    A = A';  % Data are stored in row-major order.
    fclose(f);
end