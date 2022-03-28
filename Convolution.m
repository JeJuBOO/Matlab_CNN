function out = Convolution(gradient,kernel)
arguments
    gradient (:,:,:,:) double
    kernel (:,:,:,:) double
end

[p_col,p_row,~,in_n] = size(gradient);
[k_col,k_row,k_ch,k_n] = size(kernel);
j = p_col+k_col-1;
i = p_row+k_row-1;
out = zeros(j,i,k_n,in_n);

for mini = 1:in_n
    for k = 1:k_ch 
        for d = 1:k_n 
            out(:,:,k,mini) = out(:,:,k,mini) ...
                + convn(gradient(:,:,d,mini),kernel(:,:,k,d),'full');
        end
    end
end


end
