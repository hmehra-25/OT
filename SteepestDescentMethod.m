
format short 
clear all    
clc         

syms x1 x2
f1 = x1-x2+2*x1^2+2*x1*x2+x2^2;
fx = inline(f1); 
fobj = @(x) fx(x(:,1),x(:,2)); 
%% Phase 2- Compute gradient of f
grad = gradient(f1);
G = inline(grad);  % to convert grad into a function of x1 and x2.
gradx = @(x) G(x(:,1),x(:,2));  %convert the function in one variable 
%% Phase 3- Compute Hessian Matrix
H1 = hessian(f1);
Hx = inline(H1);  
%% Phase 4- Iterations
x0 = [1 1];  % given initial value of x1 x2
maxiter = 4; % maximum number of iteration
tol = 10^(-3);  % maximum tolerance or error in our answer
iter = 0;    % iteration counter 
X = [];      % initial vector array
while norm(gradx(x0))>tol && iter<maxiter
    X = [X;x0];      % save all vectors
    S = -gradx(x0);  % compute gradient at X
    H = Hx(x0);      % compute Hessian at X
    lambda = S'*S ./(S'*H*S);  % compute lambda
    Xnew = x0+lambda.*S';    % update X
    x0=Xnew;          % save new X
    iter = iter+1;    % update iteration number
end
%% Phase 5- Print the solution
fprintf('Optimal Solution x=[%f, %f]\n',x0(1),x0(2))
fprintf('Optimal Value f(x) = %f \n',fobj(x0))