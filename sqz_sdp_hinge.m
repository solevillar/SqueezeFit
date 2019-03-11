%Delta:dxm is the set of constraints (Pi-Pj)
%epsilon is the target margin
%lambda is the hinge parameter
%we don't require the M<=Identity in this implementation
%returns M:dxd the solution of the SqueezeFit SDP. Uses CVX
function M= sqz_sdp_hinge(Delta, epsilon, lambda)


[d,m]=size(Delta);
cvx_begin sdp quiet
variable M(d,d) semidefinite
y=0;
for i=1:m
    v= Delta(:,i);
    y=y + max(epsilon - v'*M*v, 0);
end

minimize(trace(M)+ lambda*y)
%subject to
%M<=eye(d)
cvx_end

end