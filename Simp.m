clc 
clear all
a=[1 2 1 0 0; 1 1 0 1 0; 1 -1 0 0 1];
b=[10;6;2];
A=[a b];
c=[2 1 0 0 0 0];
zjmcj=c([3 4 5])*A - c;
B=[3 4 5];
var={'x1' 'x2' 's1' 's2' 's3' 'soln'};
varr={'x1' 'x2' 's1' 's2' 's3' 'zj-cj'};
d=[B 6];
% array2table(,'rownames',{var([3 4 5]),'zjxcj'}

% 
array2table( [ A;zjmcj],'variablenames',var,'rownames',varr(d)) 
while true

zc=zjmcj(1:end-1);
if any  (zc<0)
    fprintf('bfs is not optimal')
    [enter_val,piv_col]=min(zc);
    pivot_column=A(:,piv_col);
    if all(pivot_column<=0)
        error('unbounded error');
    end
       soln=A(:,end);
      for i=1: size(A,1)
           if pivot_column(i)>0
               ratio(i)=soln(i)/pivot_column(i);
           else 
               ratio(i)=inf;
           end
      end
      [leaving_var,piv_ro]=min(ratio);
      B(piv_ro)=piv_col;
      d=[B 6];
      piv_ele=A(piv_ro,piv_col);
      A(piv_ro,:)=A(piv_ro,:)/piv_ele;
      for i=1:size(A,1)
          if i~=piv_ro              
              A(i,:)=A(i,:) - A(i,piv_col)*A(piv_ro,:);
          end
      end
       zjmcj=c(B)*A-c;
        next_table=[zjmcj; A];
        array2table(next_table,'variablenames',var,'rownames',varr(d))
else
        fprintf('final optimal value is %f\n', zjmcj(end))
        
        break;
end    
end