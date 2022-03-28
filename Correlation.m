function out = Correlation(in,kernel,bi,padding)
arguments
    in (:,:,:,:) double
    kernel (:,:,:,:) double
    bi (:,:) double
    padding (1,1) double = 0 %zero padding
end

[col,row,~,in_n] = size(in);
if padding ~= 0
    pad = zeros(col+padding*2,row+padding*2,1,in_n);
    pad(padding+1:end-padding,padding+1:end-padding,:,:) = in;
    in = pad;
end

[p_col,p_row,~,in_n] = size(in);
[k_col,k_row,k_ch,k_n] = size(kernel);
j = p_col-k_col+1;
i = p_row-k_row+1;
out = zeros(j,i,k_n,in_n);

for n = 1:in_n %전체 이미지 개수
    for k = 1:k_n %전체 커널 개수
        out1 = zeros(j,i);
        for d = 1:k_ch %커널 채널 개수
            kern = squeeze(kernel(:,:,d,k));
            kern = rot90(squeeze(kern),2);
            im = squeeze(in(:,:,d,n));
            out1 = out1 + conv2(im,kern,'valid');
        end
        out(:,:,k,n) = out1+bi(k);
    end
end
end
