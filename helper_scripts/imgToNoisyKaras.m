%
% imgToNoisyKaras.m
%
% Applies noise to a standard image format and stores it to a normalized 
% grayscale image stored in a .karas container.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
% Based on code from demo_non_local_means.m written by:
%   Dimitris Floros (fcdimitr@auth.gr)
%
function imgToNoisyKaras(in_fn, out_fn)
    img = imread(in_fn);    
    if (size(img, 3) == 3)    
        gray = im2double(rgb2gray(img));
    else
        gray = im2double(img);
    end
    
    % Normalize img.
    normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));    
    gray = normImg(gray);    
    
    % Apply noise.
    noiseParams = {'gaussian', ...
                 0,...
                 0.001};
    gray = imnoise( gray, noiseParams{:} );    
    
    store2DToKaras(out_fn, gray);
end