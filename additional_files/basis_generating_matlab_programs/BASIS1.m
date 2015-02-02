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
LOG1 = zeros(500^2,1);
LOG1(1) = 0; 
for m = 2:500^2
    LOG1(m) = LOG1(m-1)+log(m);
end;
LOG2 = zeros(500^2,1);
LOG2(1) = log(1/2); 
for m = 2:500^2
    LOG2(m) = LOG2(m-1)+log(m-0.5);
end;
delta = [ones(1,250)*4000 ones(1,250)*4000+ (1:250)*32];
for n=1:NBF-1
    n2 = n^2;
    logn2 = log(n2);
    angn=exp(n2-n2*logn2+LOG2(n2));
    M(Rm,n+1)=2*angn;
    for r=1:(N-Rm)
        Mc(r+Rm,n+1)=exp(n^2-r^2+2*n^2*log(r/n));
        Mc(Rm-r,n+1)=Mc(r+Rm,n+1);
        aux=2*Mc(r+Rm,n+1)+2*angn*Mc(r+Rm,1);
        mx = max(1,r^2-delta(n));
        mn = min(n^2-1,r^2+delta(n));
        r2 = r^2;
        logr2 = log(r2);
        for l=mx:mn
          aux=aux+2*exp(n2-r2-n2*logn2+l*logr2+LOG1(n2)-LOG1(l)+LOG2(n2-l)-LOG1(n2-l));
        end;
        M(r+Rm,n+1)=aux;
        M(Rm-r,n+1)=aux;
    end;
    n
end;
save('.\dan_basis1000_1.bst','Mc', '-ASCII');
save('.\dan_basis1000pr_1.bst','M', '-ASCII');
return
   