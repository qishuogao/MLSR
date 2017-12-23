function [W]=w_compute(xt,yt,T)

[mx,nx]=size(xt);
[gindex,groups] = grp2idx(yt);
ngroups = length(groups);
xc_pj=zeros(ngroups,nx);
xa_pj=mean(xt);
for i=1:ngroups
    for j=1:nx
        xc_pj(i,j)=mean(xt(gindex==i,j)); 
        
    end
end
ruc=zeros(mx,nx);
rdc=zeros(mx,nx);
for i=1:mx
    for j=1:nx
        ruc(i,j)=(xc_pj(gindex(i),j)-xa_pj(1,j))^2;
        rdc(i,j)=(xt(i,j)-xc_pj(gindex(i),j))^2;
    end
end
ru=sum(ruc);
rd=sum(rdc);
r=ru./rd;
R=r./max(r);
w=zeros(1,nx);
for j=1:nx
    w(1,j)=exp(T*R(1,j))/sum(exp(T.*R),2);
end
W=diag(w);
end