function [D]=d_compute(xt,xp,W)

[mx,nx]=size(xt);
[pmx,pnx]=size(xp);
D=zeros(pmx,mx);
for i=1:pmx
    for j=1:mx
        D(i,j)=sqrt((xt(j,:)-xp(i,:))*W*(xt(j,:)-xp(i,:))');
    end
end
end
