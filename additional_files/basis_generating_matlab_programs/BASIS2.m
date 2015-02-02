clear all;
N=1001;
Rm = fix(N/2)+1;
NBF=500;
I=(1:N)';
R2  = ((I-Rm).^2);
R  = I-Rm;
M = zeros(N,NBF);
Mc =zeros(N,NBF);
Mc(:,1)=exp(-R2);
M(:,1)=2*exp(-R2);
for n=1:(NBF-1)
    angn=exp(n^2-2*n^2*log(n)+sum(log(0.5:(n^2-0.5))));
    M(Rm,n+1)=2*angn;
    for r=1:(N-Rm)
        Mc(r+Rm,n+1)=exp(n^2-r^2+2*n^2*log(r/n));
        Mc(Rm-r,n+1)=Mc(r+Rm,n+1);
        aux=2*Mc(r+Rm,n+1)+2*angn*Mc(r+Rm,1);
        for l=max(1,r^2-100):min(n^2-1,r^2+100)
  aux=aux+2*exp(n^2-r^2-n^2*log(n^2)+l*log(r^2)+sum(log((l+1):n^2))+sum(log(0.5:(n^2-l-0.5)))-sum(log(1:(n^2-l))));
        end;
        M(r+Rm,n+1)=aux;
        M(Rm-r,n+1)=aux;
    end;
    n
end;

save('e:\basis1000_1.bst','Mc', '-ASCII');
save('e:\basis1000pr_1.bst','M', '-ASCII');
return
   