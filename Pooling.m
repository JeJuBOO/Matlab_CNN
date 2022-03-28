function [out,out_grad] = Pooling(in,poolDim,method,stride)
arguments
    in (:,:,:,:) double
    poolDim (1,1) double
    method = 'mean'
    stride (1,1) double = 1
    
end
[col,row,ch,num] = size(in);
out = zeros(col/poolDim,row/poolDim,ch,num);
out_grad = zeros(col,row,ch,num);
pool_grad = zeros(2,2,ch,num);
j_out = 0;i_out = 0;

if method == "max"
for j = 1 :stride: col/poolDim
    j_out = j_out+1;
    for i = 1 :stride: row/poolDim
        i_out = i_out+1;
        [out(j_out,i_out,:,:),pool_idx] = ...
            max(in((j-1)*poolDim+1:poolDim*j,(i-1)*poolDim+1:poolDim*i,:,:),[],[1,2],'linear');
        pool_grad(pool_idx) = 1;
        out_grad((j-1)*poolDim+1:poolDim*j,(i-1)*poolDim+1:poolDim*i,:,:) = pool_grad;
        pool_grad = zeros(2,2,ch,num);
    end
    i_out = 0;
end
elseif method == "mean"
    for n = 1:num
        for c = 1:ch
            pooledFeaturemap = conv2(in(:,:,c,n),ones(poolDim)/(poolDim^2),'valid');
            out(:,:,c,n) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
        end
    end
end
end
