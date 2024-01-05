function [result] = PowerSpectrum( x  )
%   時系列xのパワースペクトルを生成

N = length(x); % 系列の長さ
xdft = fft(x); % 時系列からフーリエ変換
freq = 2/N : 1/N: 1/2;

xdft = abs(xdft).^2;

psdx= xdft(2 : N/2);

p = polyfit(log10(freq),log10(psdx)',1);

fitx = log10(freq);
fity = polyval(p,fitx);


%plot(log10(freq),log10(psdx),fitx,fity);

result = p(1);

end

