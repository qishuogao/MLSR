function [label]=MLSR_function(D,multi_level_test,sparsity_level, level_num,training_label,classnum)

% input arguments: D  : training data
%                  multi_level_test : multiscale training samples matrix
%                  sparsity_level : sparsity level
%                  level_num : number of level
%                  traing_label : labels of training samples
%                  classnum: number of labeled classes
% output arguments: label: the label of the test sample
%=============================================

K=size(D,2);
label_num=unique(training_label);

A = {};

Multiscale_residual = {};

for iii = 1: level_num
 Multiscale_residual{iii} =  multi_level_test{iii}; 
 A{iii} = zeros(size(D,2),size(multi_level_test{iii},2));
end

indx = [];
a = {};

for l=1:1:sparsity_level
    proj_t_whole = zeros(K,level_num);
    for ijj = 1: level_num
       proj = [];
       proj=D'*Multiscale_residual{ijj};
       proj_t =[];
       proj_t = max(abs(proj),[],2);
       proj_t_whole(:,ijj) =  proj_t;
    end
    
    max_value = [];
    max_index = [];
    for classi = 1: length(label_num)
     class_label = find( training_label == classi);   
     Class_proj_temp = proj_t_whole(class_label,:);   
     [max_value_temp, max_index_temp] = max(Class_proj_temp,[],1);       
     max_value(classi) = sum(max_value_temp);
     max_index(classi,:) = max_index_temp - 1 + class_label(1);
    end
    
    [max_m, index_m] = max(max_value);
    max_final = max_index(index_m,:);
    indx(l,:) = max_final;
    
    for tt = 1: level_num
       a{tt}=pinv(D(:,indx(1:l,tt)))*multi_level_test{tt};
       Multiscale_residual{tt}=multi_level_test{tt}-D(:,indx(1:l,tt))*a{tt};    
    end
end;
if (length(indx)>0)
    for ll = 1: level_num
    A{ll}(indx(:,ll),:) = a{ll};        
    
    end
end

label= [];
residual = zeros(1,classnum);     
for is = 1: level_num   

   res     = zeros(1,classnum);

    for j   =  1:classnum
    temp_s =  zeros(size(A{is}));
    class  =  label_num(j);
    Index = find(class==training_label);
    temp_s(Index,:)  =  A{is}(Index,:);
    zz = multi_level_test{is}-D*temp_s;
    gap(j) =  zz(:)'*zz(:); 
    end
    res =gap  ;
   
   res = res./size(A{is},2);
   residual = residual + res;     
end
  residual;  
index  =  find(residual==min(residual));
label =  [label label_num(index(1))];