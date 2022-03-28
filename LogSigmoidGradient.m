function z = LogSigmoidGradient(z)
z = LogSigmoid(z).*(1 - LogSigmoid(z));
end