function [result] = PowerSpectrum( x  )
%   ���n��x�̃p���[�X�y�N�g���𐶐�

N = length(x); % �n��̒���
xdft = fft(x); % ���n�񂩂�t�[���G�ϊ�
freq = 2/N : 1/N: 1/2;

xdft = abs(xdft).^2;

psdx= xdft(2 : N/2);

p = polyfit(log10(freq),log10(psdx)',1);

fitx = log10(freq);
fity = polyval(p,fitx);


%plot(log10(freq),log10(psdx),fitx,fity);

result = p(1);

end

