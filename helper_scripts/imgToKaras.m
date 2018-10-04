%
% imgToKaras.m
%
% Converts a standard image format to a normalized grayscale image stored in
% a .karas container.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
function imgToKaras(in_fn, out_fn)
    img = imread(in_fn);    
    if (size(img, 3) == 3)    
        gray = im2double(rgb2gray(img));
    else
        gray = im2double(img);
    end
    
    % Normalize img.
    normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));    
    gray = normImg(gray);    
    
    store2DToKaras(out_fn, gray);    
end