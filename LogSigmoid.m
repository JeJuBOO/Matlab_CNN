function z = LogSigmoid(z)
z = 1 ./ (1+exp(-z));
end